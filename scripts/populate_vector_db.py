#!/usr/bin/env python3
"""
Database prepopulation coordinator that encodes extracted crop images using SigLIP2 and stores embeddings in the Chroma vector store.
Imports tracking metadata and saves associated embeddings into database collections.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from PIL import Image
import chromadb
from tqdm import tqdm

# Add workspace root to python path to import app modules
workspace_root = Path(__file__).resolve().parent.parent
sys.path.append(str(workspace_root))

from vlm_retrieval.retrieval.encoder import get_retrieval_encoder
from vlm_retrieval.retrieval.vector_store import VectorStore
from shared.utils import setup_logger

logger = setup_logger("PrepopulateDB")

# Filename pattern expected inside each global-ID subdirectory:
#   clip1_f000001_t1_s0.04_sim1.0000.jpg (or clip1_f000001_t1_s0.04.jpg)
FILENAME_REGEX = re.compile(
    r"^(clip\d+)_f(\d+)_t(\d+)_s([\d.]+)(?:_sim([\d.]+))?\.(jpg|jpeg|png)$", re.IGNORECASE
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prepopulate Chroma DB with SigLIP2 embeddings computed from the "
            "precomputed ReID crops stored in v1. "
            "Each subdirectory of v1 is named after the global person ID "
            "and contains bounding-box crop images from the CCTV footage."
        )
    )
    parser.add_argument(
        "--crops_dir",
        type=str,
        default=str(workspace_root / "reid" / "v1"),
        help=(
            "Root directory of ReID crops. Must contain one subdirectory per global person ID "
            "(e.g. v1/42/clip1_f000001_t1_s0.04.jpg)."
        ),
    )
    parser.add_argument(
        "--reid_json",
        type=str,
        default=str(workspace_root / "reid" / "cleaned_reid.json"),
        help=(
            "Optional path to cleaned_reid.json for enriching metadata with bbox and class_label. "
            "If the file is absent, metadata is derived entirely from the directory/filename."
        ),
    )
    parser.add_argument(
        "--chroma_mode",
        type=str,
        choices=["cloud", "local"],
        default="cloud",
        help="Chroma DB connection mode: cloud (via VectorStore client) or local (via HttpClient)",
    )
    parser.add_argument(
        "--chroma_host",
        type=str,
        default="localhost",
        help="Chroma local host (only used when --chroma_mode=local)",
    )
    parser.add_argument(
        "--chroma_port",
        type=int,
        default=8200,
        help="Chroma local port (only used when --chroma_mode=local)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="track_events",
        help="Chroma DB collection name",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Number of embeddings to upsert per Chroma DB call",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of crops to process (useful for dry runs)",
    )
    parser.add_argument(
        "--reset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete all existing entries from the collection before indexing",
    )
    parser.add_argument(
        "--retrieval_model",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="Retrieval encoder model (e.g. google/siglip2-base-patch16-224, openclip-vit, openclip:ViT-B-32/laion2b_s34b_b79k)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run inference on (auto, cuda, mps, cpu)",
    )
    return parser.parse_args()


def load_reid_metadata(reid_json_path: str) -> dict:
    """Optionally load bbox / class_label enrichment from cleaned_reid.json.

    Returns a lookup dict keyed by ``(global_id: int, video_filename: str, frame: int)``.
    Returns an empty dict when the file does not exist.
    """
    if not os.path.exists(reid_json_path):
        logger.warning(
            f"ReID JSON not found at {reid_json_path}. "
            "Proceeding without bbox/class_label enrichment."
        )
        return {}

    logger.info(f"Loading ReID metadata from {reid_json_path}...")
    with open(reid_json_path, "r") as f:
        data = json.load(f)

    lookup: dict = {}
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


def camera_id_from_clip(clip_name: str) -> str:
    """Derive a camera identifier from the clip stem (e.g. 'clip1' -> 'cam_1')."""
    if clip_name.startswith("clip"):
        num = clip_name[4:]
        if num.isdigit():
            return f"cam_{num}"
    return clip_name


def collect_crop_files(crops_dir: Path) -> list:
    """Recursively collect all valid ReID crop images under *crops_dir*.

    Returns a list of ``(path, global_id, regex_match)`` tuples where:
    - ``path``      – absolute path to the crop image
    - ``global_id`` – integer person ID parsed from the parent directory name
    - ``match``     – :class:`re.Match` object against :data:`FILENAME_REGEX`
    """
    crop_files = []
    for p in crops_dir.glob("**/*"):
        if not (p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}):
            continue
        match = FILENAME_REGEX.match(p.name)
        if not match:
            logger.debug(f"Skipping file with unexpected name format: {p}")
            continue
        try:
            global_id = int(p.parent.name)
        except ValueError:
            logger.warning(
                f"Skipping {p}: parent directory '{p.parent.name}' is not an integer global ID"
            )
            continue
        crop_files.append((p, global_id, match))
    return crop_files


# ANSI colours for terminal output
_BOLD = "\033[1m"
_CYAN = "\033[96m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_MAGENTA = "\033[95m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def connect_collection(args):
    """Connect to Chroma DB, optionally reset the collection, and return its handle."""
    logger.info(f"Connecting to Chroma DB in '{args.chroma_mode}' mode...")
    if args.chroma_mode == "local":
        client = chromadb.HttpClient(host=args.chroma_host, port=args.chroma_port)
        collection = client.get_or_create_collection(
            name=args.collection,
            metadata={"hnsw:space": "cosine"},
        )
    else:
        store = VectorStore(collection_name=args.collection)
        collection = store.collection

    if args.reset:
        logger.info(f"Resetting collection '{args.collection}'...")
        try:
            existing = collection.get()
            if existing and existing.get("ids"):
                collection.delete(ids=existing["ids"])
                logger.info(f"Cleared {len(existing['ids'])} existing entries.")
        except Exception as e:
            logger.warning(f"Could not clear collection: {e}")

    logger.info(f"Target collection: '{args.collection}' (current count: {collection.count()})")
    return collection


def main():
    args = parse_args()

    print(f"\n{_BOLD}{_CYAN}┌────────────────────────────────────────────────────────────┐{_RESET}")
    print(f"{_BOLD}{_CYAN}│        CCTV Database Prepopulation Pipeline                │{_RESET}")
    print(f"{_BOLD}{_CYAN}└────────────────────────────────────────────────────────────┘{_RESET}\n")

    collection = connect_collection(args)
    metadata_lookup = load_reid_metadata(args.reid_json)

    print(
        f"🔧 {_BOLD}Initializing retrieval encoder:{_RESET} {_CYAN}{args.retrieval_model}{_RESET} on {_MAGENTA}{args.device}{_RESET}..."
    )
    encoder = get_retrieval_encoder(model_name=args.retrieval_model, device=args.device)

    crops_dir = Path(args.crops_dir)
    if not crops_dir.exists():
        logger.error(f"Crops directory does not exist: {crops_dir}")
        sys.exit(1)

    print(f"🔍 {_BOLD}Scanning crops directory:{_RESET} {crops_dir}...")
    crop_files = collect_crop_files(crops_dir)
    print(
        f"📊 {_BOLD}Found:{_RESET} {_GREEN}{len(crop_files)}{_RESET} crop images across all global IDs."
    )

    if args.limit:
        crop_files = crop_files[: args.limit]
        print(f"⚠️  {_BOLD}{_YELLOW}Limiting prepopulation to first {args.limit} crops.{_RESET}")

    batch_ids: list = []
    batch_embeddings: list = []
    batch_metadatas: list = []
    processed_count = 0

    print(f"\n🚀 {_BOLD}Encoding & indexing crops to vector store...{_RESET}\n")
    for filepath, global_id, match in tqdm(
        crop_files,
        desc="🧬 Indexing",
        bar_format="{l_bar}{bar:30}{r_bar}{bar:-10b}",
    ):
        clip_name = match.group(1)  # e.g. "clip1"
        frame_idx = int(match.group(2))  # e.g. 1
        timestamp_seconds = float(match.group(4))  # e.g. 0.04

        camera_id = camera_id_from_clip(clip_name)

        # Core metadata derived entirely from the directory/filename structure
        metadata: dict = {
            "camera_id": camera_id,
            "track_id": global_id,
            "camera_timestamp": timestamp_seconds,
            "video_pos_ms": timestamp_seconds * 1000.0,
        }

        # Optionally enrich with bbox and class_label from the ReID JSON
        video_filename = f"{clip_name}.mp4"
        entry = metadata_lookup.get((global_id, video_filename, frame_idx), {})
        if entry.get("bbox"):
            metadata["bbox"] = ",".join(str(v) for v in entry["bbox"])
        if entry.get("class_label"):
            metadata["class_label"] = entry["class_label"]

        # Unique document ID consistent with the EventStore convention
        event_id = f"{camera_id}_{global_id}_{timestamp_seconds:.4f}"

        # Encode the crop image
        try:
            with Image.open(filepath) as img:
                embedding = encoder.encode_image(img)
        except Exception as e:
            logger.error(f"Failed to encode {filepath}: {e}")
            continue

        batch_ids.append(event_id)
        batch_embeddings.append(embedding.tolist())
        batch_metadatas.append(metadata)

        if len(batch_ids) >= args.batch_size:
            try:
                collection.upsert(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                )
                processed_count += len(batch_ids)
            except Exception as e:
                logger.error(f"Failed to upsert batch: {e}")
            finally:
                batch_ids.clear()
                batch_embeddings.clear()
                batch_metadatas.clear()

    # Flush any remaining entries
    if batch_ids:
        try:
            collection.upsert(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
            )
            processed_count += len(batch_ids)
        except Exception as e:
            logger.error(f"Failed to upsert final batch: {e}")

    print(
        f"\n{_BOLD}{_GREEN}┌────────────────────────────────────────────────────────────┐{_RESET}"
    )
    print(f"{_BOLD}{_GREEN}│              PREPOPULATION COMPLETE 🎉                     │{_RESET}")
    print(f"{_BOLD}{_GREEN}├────────────────────────────────────────────────────────────┤{_RESET}")
    print(
        f"{_BOLD}{_GREEN}│{_RESET}  Processed Crops : {_BOLD}{processed_count:<40}{_RESET}{_BOLD}{_GREEN}│{_RESET}"
    )
    print(
        f"{_BOLD}{_GREEN}│{_RESET}  Total DB Items  : {_BOLD}{collection.count():<40}{_RESET}{_BOLD}{_GREEN}│{_RESET}"
    )
    print(
        f"{_BOLD}{_GREEN}│{_RESET}  Collection Name : {_BOLD}{args.collection:<40}{_RESET}{_BOLD}{_GREEN}│{_RESET}"
    )
    print(
        f"{_BOLD}{_GREEN}└────────────────────────────────────────────────────────────┘{_RESET}\n"
    )


if __name__ == "__main__":
    main()
