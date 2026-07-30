#!/usr/bin/env python3
"""
Vector database search runner querying Chroma DB using natural language, filtering candidates, and performing Florence-2 VQA matching.

Usage examples:
    poetry run python scripts/query_vector_db.py "person in red jacket"
    poetry run python scripts/query_vector_db.py "man with backpack near camera 1" --top_k 5
    poetry run python scripts/query_vector_db.py "blue shirt" --top_k 10 --no-filters
"""

import argparse
import sys
from pathlib import Path

workspace_root = Path(__file__).resolve().parent.parent
sys.path.append(str(workspace_root))

from vlm_retrieval.retrieval.encoder import get_retrieval_encoder
from vlm_retrieval.retrieval.search import RetrievalEngine, RetrievalResult
from vlm_retrieval.retrieval.vector_store import VectorStore
import re
from shared.utils import setup_logger

logger = setup_logger("QueryDB")

# ANSI colours for terminal output
_BOLD = "\033[1m"
_CYAN = "\033[96m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_DIM = "\033[2m"
_RESET = "\033[0m"

FILENAME_REGEX = re.compile(
    r"^(clip\d+)_f(\d+)_t(\d+)_s([\d.]+)(?:_sim([\d.]+))?\.(jpg|jpeg|png)$", re.IGNORECASE
)


def camera_id_from_clip(clip_name: str) -> str:
    """Derive a camera identifier from the clip stem (e.g. 'clip1' -> 'cam_1')."""
    if clip_name.startswith("clip"):
        num = clip_name[4:]
        if num.isdigit():
            return f"cam_{num}"
    return clip_name


def find_crop_path(track_id: int, camera_id: str, camera_timestamp: float) -> str:
    """Locate the local crop path in the workspace matching the metadata."""
    crops_dir = workspace_root / "reid" / "v1" / str(track_id)
    if not crops_dir.exists():
        return "Not found"

    for p in crops_dir.iterdir():
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            match = FILENAME_REGEX.match(p.name)
            if match:
                clip_name = match.group(1)
                timestamp_seconds = float(match.group(4))
                if (
                    camera_id_from_clip(clip_name) == camera_id
                    and abs(timestamp_seconds - camera_timestamp) < 1e-4
                ):
                    try:
                        return str(p.relative_to(workspace_root))
                    except ValueError:
                        return str(p)
    return "Not found"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a text query against the prepopulated ChromaDB and show top-K results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "query",
        type=str,
        help="Natural-language search query (e.g. 'person in red jacket near cam_1')",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of results to retrieve (default: 10)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="track_events",
        help="ChromaDB collection name (default: track_events)",
    )
    parser.add_argument(
        "--no-filters",
        dest="filters",
        action="store_false",
        default=True,
        help="Disable metadata filtering (camera/time constraints are ignored)",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="Force-filter results to a specific camera ID (e.g. cam_1)",
    )
    parser.add_argument(
        "--retrieval_model",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="Retrieval encoder model to use (default: google/siglip2-so400m-patch14-384)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run encoder on (default: auto)",
    )
    return parser.parse_args()


def _similarity_bar(distance: float, width: int = 20) -> str:
    """Render a visual similarity bar. ChromaDB cosine distance: 0=identical, 1=orthogonal."""
    similarity = max(0.0, 1.0 - distance)
    filled = round(similarity * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"{bar} {similarity * 100:.1f}%"


def _fmt_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h:
        return f"{h:02d}:{m:02d}:{s:06.3f}"
    return f"{m:02d}:{s:06.3f}"


def print_results(query: str, parsed_query, results: list[RetrievalResult]) -> None:
    print()
    print(f"{_BOLD}{'─' * 70}{_RESET}")
    print(f"{_BOLD}  Query     :{_RESET} {_CYAN}{query}{_RESET}")
    print(f"{_BOLD}  Semantic  :{_RESET} {parsed_query.semantic_text}")
    if parsed_query.metadata_filters:
        print(f"{_BOLD}  Filters   :{_RESET} {parsed_query.metadata_filters}")
    print(f"{_BOLD}  Results   :{_RESET} {len(results)} returned")
    print(f"{_BOLD}{'─' * 70}{_RESET}")

    if not results:
        print(f"\n  {_YELLOW}No results found.{_RESET}\n")
        return

    for rank, r in enumerate(results, start=1):
        bar = _similarity_bar(r.distance)
        ts = _fmt_timestamp(r.camera_timestamp)
        bbox_str = f"  bbox={r.bbox}" if r.bbox else ""
        crop_path = find_crop_path(r.track_id, r.camera_id, r.camera_timestamp)

        print(f"\n  {_BOLD}#{rank:<3}{_RESET}  {_GREEN}ID:{_RESET} {r.id}")
        print(
            f"       {_GREEN}Camera:{_RESET} {r.camera_id:<10}"
            f"  {_GREEN}Track:{_RESET} {r.track_id:<6}"
            f"  {_GREEN}Time:{_RESET} {ts}"
            f"{_DIM}{bbox_str}{_RESET}"
        )
        print(f"       {_GREEN}Similarity:{_RESET} {bar}")
        print(f"       {_GREEN}Crop Path:{_RESET} {crop_path}")

    print(f"\n{_BOLD}{'─' * 70}{_RESET}\n")


def show_results_grid(results: list[RetrievalResult], query: str) -> None:
    """Create a grid of top-K crop images and display them, including the query prompt in the image."""
    if not results:
        return

    # Collect all valid image paths
    valid_paths = []
    for r in results:
        crop_path = find_crop_path(r.track_id, r.camera_id, r.camera_timestamp)
        if crop_path != "Not found":
            p = workspace_root / crop_path
            if p.exists():
                valid_paths.append((p, r))

    if not valid_paths:
        logger.warning("No crop images found to display in a grid.")
        return

    from PIL import Image, ImageDraw, ImageFont
    import math

    images = []
    for p, r in valid_paths:
        try:
            img = Image.open(p).convert("RGB")
            images.append((img, r))
        except Exception as e:
            logger.warning(f"Failed to load image {p}: {e}")

    if not images:
        return

    num_images = len(images)
    cols = min(5, num_images)
    rows = math.ceil(num_images / cols)

    cell_w, cell_h = 160, 240
    header_h = 50
    grid_img = Image.new("RGB", (cols * cell_w, rows * cell_h + header_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(grid_img)

    # Try to load a system font, fall back to default
    font = None
    title_font = None
    for font_name in ["Arial.ttf", "Helvetica.ttf", "sans-serif.ttf"]:
        try:
            font = ImageFont.truetype(font_name, 12)
            title_font = ImageFont.truetype(font_name, 14)
            break
        except IOError:
            continue
    if font is None:
        try:
            # Common macOS path
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
            title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except IOError:
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()

    # Draw header background
    draw.rectangle([(0, 0), (cols * cell_w, header_h)], fill=(20, 20, 20))
    # Subtle separator line
    draw.line([(0, header_h - 1), (cols * cell_w, header_h - 1)], fill=(60, 60, 60), width=1)

    # Draw the query text in the header
    display_query = f'Query: "{query}"'
    if hasattr(title_font, "getbbox"):
        bbox = title_font.getbbox(display_query)
        qw = bbox[2] - bbox[0]
        qh = bbox[3] - bbox[1]
    else:
        qw, qh = title_font.getsize(display_query)

    # Truncate text if it exceeds image width
    max_text_w = cols * cell_w - 20
    if max_text_w > 50 and qw > max_text_w:
        for i in range(len(display_query), 0, -1):
            test_str = display_query[:i] + "..."
            if hasattr(title_font, "getbbox"):
                t_bbox = title_font.getbbox(test_str)
                tw = t_bbox[2] - t_bbox[0]
                th = t_bbox[3] - t_bbox[1]
            else:
                tw, th = title_font.getsize(test_str)
            if tw <= max_text_w:
                display_query = test_str
                qw = tw
                qh = th
                break

    text_x = max(10, (cols * cell_w - qw) // 2)
    text_y = (header_h - qh) // 2
    draw.text((text_x, text_y), display_query, fill=(240, 240, 240), font=title_font)

    for idx, (img, r) in enumerate(images):
        c = idx % cols
        row = idx // cols

        x_offset = c * cell_w
        y_offset = row * cell_h + header_h

        # Resize crop image to fit 160x200 while preserving aspect ratio
        img_w, img_h = img.size
        ratio = min(cell_w / img_w, 200 / img_h)
        new_w = int(img_w * ratio)
        new_h = int(img_h * ratio)

        resample_filter = getattr(Image, "Resampling", None)
        if resample_filter:
            filter_mode = resample_filter.LANCZOS
        else:
            filter_mode = getattr(Image, "ANTIALIAS", Image.BICUBIC)

        resized_img = img.resize((new_w, new_h), filter_mode)

        # Center crop within 160x200 top region
        paste_x = x_offset + (cell_w - new_w) // 2
        paste_y = y_offset + (200 - new_h) // 2
        grid_img.paste(resized_img, (paste_x, paste_y))

        # Bottom label strip
        draw.rectangle(
            [(x_offset, y_offset + 200), (x_offset + cell_w, y_offset + cell_h)], fill=(45, 45, 45)
        )

        rank = idx + 1
        similarity_pct = max(0.0, 1.0 - r.distance) * 100.0
        text_line1 = f"#{rank} ({similarity_pct:.1f}%)"
        text_line2 = f"{r.camera_id} T:{r.track_id}"

        # Align text centered
        if hasattr(font, "getbbox"):
            w1 = font.getbbox(text_line1)[2] - font.getbbox(text_line1)[0]
            w2 = font.getbbox(text_line2)[2] - font.getbbox(text_line2)[0]
        else:
            w1, _ = font.getsize(text_line1)
            w2, _ = font.getsize(text_line2)

        draw.text(
            (x_offset + (cell_w - w1) // 2, y_offset + 203),
            text_line1,
            fill=(255, 255, 255),
            font=font,
        )
        draw.text(
            (x_offset + (cell_w - w2) // 2, y_offset + 220),
            text_line2,
            fill=(200, 200, 200),
            font=font,
        )

    # Save to workspace root
    grid_path = workspace_root / "query_results_grid.png"
    try:
        grid_img.save(grid_path)
        logger.info(f"Saved results grid to {grid_path}")
    except Exception as e:
        logger.error(f"Failed to save grid image: {e}")

    # Display image
    try:
        try:
            input("\nPress Enter to show the results grid image...")
        except (EOFError, OSError):
            pass
        logger.info("Opening the results grid image...")
        grid_img.show()
    except Exception as e:
        logger.warning(f"Could not open image viewer automatically: {e}")


def main():
    args = parse_args()

    logger.info(f"Initializing retrieval encoder '{args.retrieval_model}'...")
    encoder = get_retrieval_encoder(model_name=args.retrieval_model, device=args.device)

    logger.info(f"Connecting to ChromaDB collection '{args.collection}'...")
    vector_store = VectorStore(collection_name=args.collection)

    logger.info(f"Collection has {vector_store.get_event_count()} indexed embeddings.")

    engine = RetrievalEngine(
        encoder=encoder,
        vector_store=vector_store,
        metadata_filter_enabled=args.filters,
    )

    logger.info(f"Running query: '{args.query}' (top_k={args.top_k})")
    parsed, results = engine.search(
        query=args.query,
        top_k=args.top_k,
        camera_id=args.camera,
    )

    print_results(args.query, parsed, results)
    show_results_grid(results, args.query)


if __name__ == "__main__":
    main()
