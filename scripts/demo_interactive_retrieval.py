#!/usr/bin/env python3
"""
Interactive terminal demo application to test SigLIP2 semantic search and Florence-2 VQA crop reranking.
Loads cropped identity images, encodes them in memory, and provides an interactive search interface for testing queries.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Add workspace root to python path to import app modules
workspace_root = Path(__file__).resolve().parent.parent
sys.path.append(str(workspace_root))

from vlm_retrieval.retrieval.encoder import get_retrieval_encoder, BaseRetrievalEncoder
from vlm_retrieval.vqa import get_vqa_reasoner, BaseVQAReasoner, CandidateImage
from vlm_retrieval.retrieval.query_parser import parse_query
from shared.utils import setup_logger, compute_cosine_similarity

logger = setup_logger("DryRunInference")

# ANSI colors for beautiful terminal output
_BOLD = "\033[1m"
_CYAN = "\033[96m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_MAGENTA = "\033[95m"
_DIM = "\033[2m"
_RESET = "\033[0m"

# Filename pattern expected inside each global-ID subdirectory:
#   clip1_f000001_t1_s0.04_sim1.0000.jpg (or clip1_f000001_t1_s0.04.jpg)
FILENAME_REGEX = re.compile(
    r"^(clip\d+)_f(\d+)_t(\d+)_s([\d.]+)(?:_sim([\d.]+))?\.(jpg|jpeg|png)$", re.IGNORECASE
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dry-run script to test SigLIP2 retrieval and Florence-2 VQA in memory."
    )
    parser.add_argument(
        "--crops_dir",
        type=str,
        default=str(workspace_root / "reid" / "v1"),
        help="Root directory of ReID crops.",
    )
    parser.add_argument(
        "--reid_json",
        type=str,
        default=str(workspace_root / "reid" / "cleaned_reid.json"),
        help="Path to cleaned_reid.json for bbox / class label lookup.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cpu", "cuda", "mps", "auto"],
        help="Device to use for PyTorch models.",
    )
    parser.add_argument(
        "--retrieval_model",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="Retrieval encoder model (e.g. google/siglip2-base-patch16-224, openclip-vit, openclip:ViT-B-32/laion2b_s34b_b79k)",
    )
    parser.add_argument(
        "--retrieval_top_k",
        type=int,
        default=10,
        help="Number of candidates to retrieve using SigLIP2 (passed to Florence-2).",
    )
    parser.add_argument(
        "--rerank_top_k",
        type=int,
        default=5,
        help="Number of final candidates to return after Florence-2 reranking.",
    )
    parser.add_argument(
        "--reasoning_model",
        type=str,
        default="microsoft/Florence-2-large",
        help="VQA reasoning model to use (e.g. microsoft/Florence-2-large or Qwen/Qwen2-VL-2B-Instruct).",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run a single query and exit. If not provided, enters interactive mode.",
    )
    return parser.parse_args()


def camera_id_from_clip(clip_name: str) -> str:
    """Derive a camera identifier from the clip stem (e.g. 'clip1' -> 'cam_1')."""
    if clip_name.startswith("clip"):
        num = clip_name[4:]
        if num.isdigit():
            return f"cam_{num}"
    return clip_name


def collect_crop_files(crops_dir: Path) -> list:
    """Recursively collect all valid ReID crop images under crops_dir."""
    crop_files = []
    for p in crops_dir.glob("**/*"):
        if not (p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}):
            continue
        match = FILENAME_REGEX.match(p.name)
        if not match:
            continue
        try:
            global_id = int(p.parent.name)
        except ValueError:
            continue
        crop_files.append((p, global_id, match))
    return crop_files


def load_reid_metadata(reid_json_path: str) -> dict:
    """Load metadata lookup dictionary from cleaned_reid.json."""
    if not os.path.exists(reid_json_path):
        logger.warning(
            f"ReID JSON not found at {reid_json_path}. "
            "Proceeding without bbox/class_label enrichment."
        )
        return {}

    logger.info(f"Loading ReID metadata from {reid_json_path}...")
    with open(reid_json_path, "r") as f:
        data = json.load(f)

    lookup = {}
    for global_id_str, entries in data.items():
        try:
            global_id = int(global_id_str)
        except ValueError:
            continue
        for entry in entries:
            video = entry.get("video", "")
            frame = entry.get("frame")
            if video and frame is not None:
                lookup[(global_id, video, int(frame))] = entry

    logger.info(f"Loaded {len(lookup)} ReID metadata entries.")
    return lookup


def filter_candidates(candidates: list[dict], filters: dict) -> list[dict]:
    """Filter candidates using metadata filters parsed from the query."""
    filtered = []
    for c in candidates:
        keep = True
        meta = c["metadata"]

        if "camera_id" in filters:
            # We support flexible matching (e.g. 'cam_1' vs 'cam1' vs '1')
            q_cam = str(filters["camera_id"]).lower().replace("_", "").replace("cam", "")
            c_cam = str(meta["camera_id"]).lower().replace("_", "").replace("cam", "")
            if q_cam != c_cam:
                keep = False

        if "camera_timestamp_gte" in filters:
            if meta["camera_timestamp"] < float(filters["camera_timestamp_gte"]):
                keep = False

        if "camera_timestamp_lt" in filters:
            if meta["camera_timestamp"] >= float(filters["camera_timestamp_lt"]):
                keep = False

        if keep:
            filtered.append(c)

    return filtered


def run_search_pipeline(
    query_str: str,
    candidates_db: list[dict],
    encoder: BaseRetrievalEncoder,
    reasoner: BaseVQAReasoner,
    retrieval_k: int,
    rerank_k: int,
):
    """Executes search and VQA reranking for a single query."""
    # 1. Parse query for filters
    parsed_query = parse_query(query_str)
    semantic_text = parsed_query.semantic_text or query_str

    print(f"\n{_BOLD}{_CYAN}🔍 SEARCH PIPELINE RUN{_RESET}")
    print(f"{_CYAN}╪{'═' * 70}╪{_RESET}")
    print(f"  {_BOLD}Raw Query      :{_RESET} {_CYAN}{query_str}{_RESET}")
    print(f"  {_BOLD}Semantic Query :{_RESET} {semantic_text}")
    if parsed_query.metadata_filters:
        print(f"  {_BOLD}Filters        :{_RESET} {parsed_query.metadata_filters}")
    print(f"{_CYAN}╪{'═' * 70}╪{_RESET}\n")

    # 2. Encode text query
    try:
        query_embedding = encoder.encode_text(semantic_text)
    except Exception as e:
        logger.error(f"Failed to encode query text: {e}")
        return

    # 3. Compute cosine similarity & filter candidates
    scored_candidates = []
    for c in candidates_db:
        sim = compute_cosine_similarity(query_embedding, c["embedding"])
        scored_candidates.append({**c, "similarity": sim})

    # Apply parsed metadata filters
    if parsed_query.metadata_filters:
        filtered_scored = filter_candidates(scored_candidates, parsed_query.metadata_filters)
        print(
            f"⚙️  {_BOLD}Applied filters:{_RESET} {len(scored_candidates)} -> {_GREEN}{len(filtered_scored)}{_RESET} candidates remaining."
        )
        scored_candidates = filtered_scored

    # Sort and take top retrieval_k
    scored_candidates.sort(key=lambda x: x["similarity"], reverse=True)
    top_retrieved = scored_candidates[:retrieval_k]

    if not top_retrieved:
        print(f"⚠️  {_YELLOW}No candidates matched the query or filters.{_RESET}")
        return

    print(f"📡 {_BOLD}{_CYAN}Top {len(top_retrieved)} SigLIP2 Retrieval Candidates:{_RESET}")
    print(f"{_DIM}  {'─' * 72}{_RESET}")
    for idx, c in enumerate(top_retrieved, start=1):
        similarity = c["similarity"]
        filled = max(0, min(10, round(similarity * 10)))
        bar = "█" * filled + "░" * (10 - filled)
        sim_str = f"{bar} {similarity * 100:.1f}%"
        print(
            f"  {_BOLD}#{idx:<2}{_RESET} "
            f"[{_GREEN}Cam:{_RESET} {c['metadata']['camera_id']:<6} | "
            f"{_GREEN}Track ID:{_RESET} {c['metadata']['track_id']:<4} | "
            f"{_GREEN}Time:{_RESET} {c['metadata']['camera_timestamp']:.2f}s] "
            f"{_DIM}Score: {sim_str}{_RESET}\n"
            f"      {_DIM}File: {c['filepath'].name}{_RESET}"
        )
    print(f"{_DIM}  {'─' * 72}{_RESET}\n")

    # 4. Prep and run VQA reasoning
    print(f"🧠 {_BOLD}{_MAGENTA}Running {reasoner.__class__.__name__} VQA Reranking...{_RESET}")
    reasoning_candidates = []
    for c in top_retrieved:
        try:
            # Load the frame
            frame = Image.open(c["filepath"]).convert("RGB")
            bbox = c["metadata"].get("bbox")
            bbox_list = [float(v) for v in bbox.split(",")] if bbox else None

            reasoning_candidates.append(
                CandidateImage(
                    camera_id=c["metadata"]["camera_id"],
                    camera_timestamp=c["metadata"]["camera_timestamp"],
                    video_pos_ms=c["metadata"]["video_pos_ms"],
                    track_id=c["metadata"]["track_id"],
                    bbox=bbox_list,
                    frame=frame,
                    retrieval_distance=1.0 - c["similarity"],
                )
            )
        except Exception as e:
            logger.error(f"Failed to load image candidate {c['filepath']}: {e}")

    if not reasoning_candidates:
        print(
            f"❌ {_RED}Failed to prepare any candidate images for {reasoner.__class__.__name__}.{_RESET}"
        )
        return

    # Run VQA Reranking
    ranked_results = reasoner.answer(
        query=semantic_text,
        candidates=reasoning_candidates,
        top_k=rerank_k,
    )

    # Print final Ranked Results
    print(f"\n{_BOLD}{_GREEN}🏆 FINAL RANKED RESULTS ({reasoner.__class__.__name__}){_RESET}")
    print(f"{_GREEN}╪{'═' * 70}╪{_RESET}")
    if not ranked_results:
        print(f"  {_YELLOW}No candidate verified by {reasoner.__class__.__name__}.{_RESET}")
    else:
        for idx, res in enumerate(ranked_results, start=1):
            color = _GREEN if res.vlm_score > 0 else _YELLOW
            status = "VERIFIED" if res.vlm_score > 0 else "REJECTED"
            score_bar = (
                "★" * int(res.vlm_score) + "☆" * (5 - int(res.vlm_score))
                if 0 <= res.vlm_score <= 5
                else f"Score: {res.vlm_score}"
            )

            print(
                f"  {_BOLD}#{idx:<2}{_RESET} "
                f"[{color}{status:<8}{_RESET}]  "
                f"({_CYAN}{score_bar}{_RESET})  "
                f"Cam: {_BOLD}{res.camera_id:<6}{_RESET} "
                f"Track: {_BOLD}{res.track_id:<4}{_RESET} "
                f"Time: {_BOLD}{res.camera_timestamp:.2f}s{_RESET}"
            )
            print(f"       {_DIM}Reasoning: {res.vlm_explanation}{_RESET}\n")
    print(f"{_GREEN}╪{'═' * 70}╪{_RESET}\n")


def main():
    args = parse_args()

    print(f"\n{_BOLD}{_CYAN}┌────────────────────────────────────────────────────────────┐{_RESET}")
    print(f"{_BOLD}{_CYAN}│         CCTV Semantic Search & VQA Inference Node          │{_RESET}")
    print(f"{_BOLD}{_CYAN}└────────────────────────────────────────────────────────────┘{_RESET}\n")

    # Load ReID metadata lookup
    metadata_lookup = load_reid_metadata(args.reid_json)

    # 1. Initialize retrieval encoder
    print(
        f"🔧 {_BOLD}Initializing retrieval encoder:{_RESET} {_CYAN}{args.retrieval_model}{_RESET} on {_MAGENTA}{args.device}{_RESET}..."
    )
    encoder = get_retrieval_encoder(model_name=args.retrieval_model, device=args.device)

    # 2. Collect crop files
    crops_dir = Path(args.crops_dir)
    if not crops_dir.exists():
        logger.error(f"Crops directory does not exist: {crops_dir}")
        sys.exit(1)

    print(f"🔍 {_BOLD}Scanning crops directory:{_RESET} {crops_dir}...")
    crop_files = collect_crop_files(crops_dir)
    print(
        f"📊 {_BOLD}Found:{_RESET} {_GREEN}{len(crop_files)}{_RESET} crop images under the directory."
    )

    if not crop_files:
        logger.error("No valid crops found. Exiting.")
        sys.exit(1)

    # 3. Load & Encode all crops in-memory
    print(f"\n🚀 {_BOLD}Encoding all crop images into memory embeddings...{_RESET}\n")
    candidates_db = []
    for filepath, global_id, match in tqdm(
        crop_files,
        desc="🧬 Encoding crops",
        bar_format="{l_bar}{bar:30}{r_bar}{bar:-10b}",
    ):
        clip_name = match.group(1)
        frame_idx = int(match.group(2))
        timestamp_seconds = float(match.group(4))
        camera_id = camera_id_from_clip(clip_name)

        # Build metadata
        metadata = {
            "camera_id": camera_id,
            "track_id": global_id,
            "camera_timestamp": timestamp_seconds,
            "video_pos_ms": timestamp_seconds * 1000.0,
        }

        # Enrich with cleaned_reid.json
        video_filename = f"{clip_name}.mp4"
        entry = metadata_lookup.get((global_id, video_filename, frame_idx), {})
        if entry.get("bbox"):
            metadata["bbox"] = ",".join(str(v) for v in entry["bbox"])
        if entry.get("class_label"):
            metadata["class_label"] = entry["class_label"]

        # Encode image
        try:
            with Image.open(filepath) as img:
                embedding = encoder.encode_image(img)
        except Exception as e:
            logger.error(f"Failed to encode {filepath}: {e}")
            continue

        candidates_db.append({"filepath": filepath, "embedding": embedding, "metadata": metadata})

    print(
        f"\n✅ {_BOLD}{_GREEN}Successfully encoded {len(candidates_db)} crops into memory.{_RESET}"
    )

    # 4. Initialize VQA Reasoner
    print(
        f"🤖 {_BOLD}Initializing VQA Reasoner:{_RESET} {_MAGENTA}{args.reasoning_model}{_RESET}..."
    )
    reasoner = get_vqa_reasoner(model_name=args.reasoning_model, device=args.device)

    # 5. Handle Query input
    if args.query:
        # Run single query mode
        run_search_pipeline(
            query_str=args.query,
            candidates_db=candidates_db,
            encoder=encoder,
            reasoner=reasoner,
            retrieval_k=args.retrieval_top_k,
            rerank_k=args.rerank_top_k,
        )
    else:
        # Interactive mode
        print(f"\n{_BOLD}{_GREEN}✨ In-Memory Inference Pipeline Ready{_RESET}")
        print(
            f"{_DIM}Type your search query and press Enter. Type 'exit' or 'quit' to close.{_RESET}"
        )
        while True:
            try:
                query_str = input(f"\n{_BOLD}Query > {_RESET}").strip()
                if not query_str:
                    continue
                if query_str.lower() in ("exit", "quit"):
                    print("Exiting dry-run inference pipeline. Goodbye!")
                    break
                run_search_pipeline(
                    query_str=query_str,
                    candidates_db=candidates_db,
                    encoder=encoder,
                    reasoner=reasoner,
                    retrieval_k=args.retrieval_top_k,
                    rerank_k=args.rerank_top_k,
                )
            except KeyboardInterrupt:
                print("\nExiting dry-run inference pipeline. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error during query execution: {e}")


if __name__ == "__main__":
    main()
