#!/usr/bin/env python3
"""
Build CCTV VLM Database Script

Ingests all workspace pipeline artifacts (NPZ embedding files, JSON registries, tracks, crops)
and outputs a single, self-contained SQL database file (`cctv_vlm.db`).

The generated database serves as the sole single point of truth for VLM retrieval and reasoning.
Timestamps use ISO 8601 strings, starting with Video 0 at '2026-08-02T00:00:00Z' (0.0s baseline)
and subsequent videos spaced exactly 2 minutes (120.0 seconds) apart.

Usage:
    python scripts/build_cctv_database.py --artifacts_dir artifacts --output_db artifacts/cctv_vlm.db
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add workspace root to sys.path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from shared.utils import setup_logger
from tracking.compression.builder import CompressedTrackBuilder
from tracking.serialization.json_serializer import JsonSerializer

logger = setup_logger("BuildCCTVDatabase")

BASE_ISO_DATETIME = datetime.datetime(2026, 8, 2, 0, 0, 0, tzinfo=datetime.timezone.utc)


def format_iso_timestamp(offset_sec: float) -> str:
    """Format a relative timestamp in seconds into an ISO 8601 string."""
    dt = BASE_ISO_DATETIME + datetime.timedelta(seconds=float(offset_sec))
    return dt.isoformat()


def camera_id_from_clip(video_name: str) -> str:
    """Derive camera identifier from clip stem or name (e.g. 'c001.mp4' -> 'c001', 'c006_vdo.avi' -> 'c006')."""
    stem = video_name.replace(".mp4", "").replace(".avi", "")
    match_c = re.search(r"(c\d{3}|c\d+)", stem)
    if match_c:
        return match_c.group(1)
    match_clip = re.search(r"clip(\d+)", stem)
    if match_clip:
        num = int(match_clip.group(1))
        return f"c{num:03d}"
    return stem


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build CCTV VLM SQLite database file from pipeline artifact files."
    )
    parser.add_argument(
        "--artifacts_dir",
        type=str,
        default="artifacts",
        help="Path to directory containing artifact files (NPZ, JSON, crops).",
    )
    parser.add_argument(
        "--output_db",
        type=str,
        default="artifacts/cctv_vlm.db",
        help="Target output SQLite database file path (default: artifacts/cctv_vlm.db).",
    )
    parser.add_argument(
        "--test_tracks_json",
        type=str,
        default=None,
        help="Optional path to dataset ground truth tracks JSON file. Omit for videos without ground truths.",
    )
    parser.add_argument(
        "--retrieval_model",
        type=str,
        default="google/siglip2-so400m-patch14-384",
        help="Default retrieval model version tag (e.g. 'google/siglip2-so400m-patch14-384').",
    )
    parser.add_argument(
        "--reid_model",
        type=str,
        default="resnetibn",
        help="Default ReID model version tag (e.g. 'resnetibn').",
    )
    return parser.parse_args()


def init_database_schema(conn: sqlite3.Connection) -> None:
    """Initialize videos, tracks, and embeddings SQL tables and indexes."""
    cursor = conn.cursor()

    cursor.executescript(
        """
        CREATE TABLE IF NOT EXISTS videos (
            video_name TEXT PRIMARY KEY,
            camera_id TEXT NOT NULL,
            video_index INTEGER NOT NULL,
            start_timestamp_iso TEXT NOT NULL,
            start_timestamp_sec REAL NOT NULL,
            duration_sec REAL DEFAULT 0.0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS tracks (
            id TEXT PRIMARY KEY,
            global_id TEXT,
            track_id INTEGER NOT NULL,
            camera_id TEXT NOT NULL,
            video_name TEXT NOT NULL,
            sequence_id TEXT,
            start_time_iso TEXT NOT NULL,
            end_time_iso TEXT NOT NULL,
            camera_timestamp_iso TEXT NOT NULL,
            camera_timestamp_sec REAL NOT NULL,
            start_time REAL NOT NULL,
            end_time REAL NOT NULL,
            class_label TEXT NOT NULL,
            trajectory TEXT,
            occurrences TEXT,
            compressed_track TEXT,
            raw_frames TEXT,
            raw_boxes TEXT,
            metadata TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(video_name) REFERENCES videos(video_name)
        );

        CREATE TABLE IF NOT EXISTS embeddings (
            id TEXT PRIMARY KEY,
            global_id TEXT,
            track_id INTEGER NOT NULL,
            camera_id TEXT NOT NULL,
            video_name TEXT NOT NULL,
            embedding_type TEXT NOT NULL,
            model_name TEXT NOT NULL,
            vector_dim INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            camera_timestamp_iso TEXT NOT NULL,
            camera_timestamp_sec REAL NOT NULL,
            video_pos_ms REAL NOT NULL,
            start_time_iso TEXT NOT NULL,
            end_time_iso TEXT NOT NULL,
            class_label TEXT NOT NULL,
            bbox TEXT,
            crop_path TEXT,
            metadata TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(video_name) REFERENCES videos(video_name)
        );

        CREATE INDEX IF NOT EXISTS idx_embeddings_type_model ON embeddings(embedding_type, model_name);
        CREATE INDEX IF NOT EXISTS idx_embeddings_cam_time ON embeddings(camera_id, camera_timestamp_sec);
        CREATE INDEX IF NOT EXISTS idx_embeddings_class ON embeddings(class_label);
        CREATE INDEX IF NOT EXISTS idx_tracks_cam_time ON tracks(camera_id, camera_timestamp_sec);
        CREATE INDEX IF NOT EXISTS idx_tracks_global_id ON tracks(global_id);
        CREATE INDEX IF NOT EXISTS idx_tracks_sequence_id ON tracks(sequence_id);
        """
    )
    conn.commit()


def process_video_clips(
    artifacts_dir: Path, conn: sqlite3.Connection
) -> Dict[str, Tuple[int, str, float]]:
    """Scan and register all video clips with 2-minute (120s) timestamp spacing.

    Returns mapping: video_name -> (video_index, start_timestamp_iso, start_timestamp_sec)
    """
    video_names = set()

    # Discover videos from input_vids
    input_vids_dir = workspace_root / "input_vids"
    if input_vids_dir.exists():
        for f in input_vids_dir.glob("*.*"):
            if f.suffix.lower() in [".mp4", ".avi", ".mkv", ".mov"]:
                video_names.add(f.name)

    # Check models json file for feed keys
    models_json = artifacts_dir / "registry.tracks.models.json"
    if models_json.exists():
        try:
            with open(models_json, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    for k in data.keys():
                        video_names.add(k)
        except Exception as e:
            logger.warning(f"Error parsing models JSON for videos: {e}")

    # Check identities json file for video clip names
    identities_json = artifacts_dir / "registry.tracks.identities.json"
    if identities_json.exists():
        try:
            with open(identities_json, "r") as f:
                data = json.load(f)
                for item in data.values():
                    if isinstance(item, dict):
                        for tid in item.get("track_ids", []):
                            if isinstance(tid, str) and "_" in tid:
                                v_name = tid.rsplit("_", 1)[0]
                                video_names.add(v_name)
        except Exception as e:
            logger.warning(f"Error parsing identities JSON for videos: {e}")

    # Discover crops directories under artifacts/crops/
    crops_dir = artifacts_dir / "crops"
    if crops_dir.exists():
        for d in crops_dir.iterdir():
            if d.is_dir() and "_" in d.name:
                v_name = d.name.rsplit("_", 1)[0]
                video_names.add(v_name)

    # Fallback to default clip names if none discovered
    if not video_names:
        video_names = {"c001.mp4", "c002.mp4"}

    sorted_videos = sorted(list(video_names))
    video_offsets: Dict[str, Tuple[int, str, float]] = {}

    cursor = conn.cursor()
    for idx, v_name in enumerate(sorted_videos):
        start_sec = idx * 120.0  # 2 minutes apart per subsequent video
        start_iso = format_iso_timestamp(start_sec)
        cam_id = camera_id_from_clip(v_name)

        video_offsets[v_name] = (idx, start_iso, start_sec)
        cursor.execute(
            """
            INSERT OR REPLACE INTO videos (video_name, camera_id, video_index, start_timestamp_iso, start_timestamp_sec, duration_sec)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (v_name, cam_id, idx, start_iso, start_sec, 0.0),
        )

    conn.commit()
    logger.info(f"Registered {len(video_offsets)} video clips with 2-minute spacing.")
    return video_offsets


def ingest_test_tracks_json(
    test_tracks_path: Path,
    conn: sqlite3.Connection,
) -> int:
    """Ingest test-tracks.json ground truth dataset tracks into SQLite database.

    Bridges dataset frame/bbox presentation with compressed polynomial/spline track models.
    NOTE: Ground truth queries are explicitly NOT ingested.
    """
    target_path = test_tracks_path
    if not target_path.exists():
        logger.info(f"Ground truth tracks file '{target_path}' not found; skipping dataset ground truth ingestion.")
        return 0

    try:
        with open(target_path, "r") as f:
            test_tracks_data = json.load(f)
    except Exception as err:
        logger.error(f"Failed loading tracks JSON from '{target_path}': {err}")
        return 0

    logger.info(f"Loaded {len(test_tracks_data)} track entries from '{target_path}'")
    cursor = conn.cursor()
    count = 0

    for uuid_key, track_info in test_tracks_data.items():
        if not isinstance(track_info, dict):
            continue

        frames = track_info.get("frames", [])
        boxes = track_info.get("boxes", [])

        if not frames or not boxes or len(frames) != len(boxes):
            continue

        # Extract sequence_id and camera_id from metadata or first frame path string
        first_frame = frames[0]
        seq_id = str(track_info.get("sequence", track_info.get("sequence_id", "S02")))
        cam_id = str(track_info.get("camera", track_info.get("camera_id", "c006")))

        if isinstance(first_frame, str):
            match_seq = re.search(r"/(S\d+)/", first_frame)
            if match_seq:
                seq_id = match_seq.group(1)
            match_cam = re.search(r"/(c\d+)/", first_frame)
            if match_cam:
                cam_id = match_cam.group(1)

        v_name = f"{seq_id}_{cam_id}.mp4"

        # Build compressed track model using CompressedTrackBuilder
        builder = CompressedTrackBuilder()
        builder.set_metadata(
            track_id=abs(hash(uuid_key)) % 10000000,
            camera_id=cam_id,
            class_label="vehicle",
        )

        obs_frames = []
        obs_timestamps = []
        obs_bboxes = []
        formatted_frame_paths = []

        for idx, (f_item, box) in enumerate(zip(frames, boxes)):
            # Box format: [x, y, w, h] -> convert to xyxy: [x, y, x + w, y + h]
            x, y, w, h = box
            bbox_xyxy = (float(x), float(y), float(x + w), float(y + h))

            if isinstance(f_item, int):
                f_num = f_item
                f_path = f"./dataset/{seq_id}/{cam_id}/img1/{f_num:06d}.jpg"
            elif isinstance(f_item, str):
                f_path = f_item
                f_match = re.search(r"(\d+)\.(jpg|png|jpeg)$", f_item, re.IGNORECASE)
                f_num = int(f_match.group(1)) if f_match else idx
            else:
                f_num = idx
                f_path = f"./dataset/{seq_id}/{cam_id}/img1/{f_num:06d}.jpg"

            ts = float(f_num) / 10.0  # Assumed 10.0 FPS timeline

            obs_frames.append(f_num)
            obs_timestamps.append(ts)
            obs_bboxes.append(bbox_xyxy)
            formatted_frame_paths.append(f_path)

            builder.add_observation(
                frame=f_num,
                timestamp=ts,
                bbox=bbox_xyxy,
                confidence=1.0,
            )

        compressed_dict = {}
        try:
            compressed_track_obj = builder.build()
            compressed_dict = JsonSerializer.serialize_to_dict(compressed_track_obj)
        except Exception as err:
            logger.debug(f"Track compression note for {uuid_key}: {err}")

        st_rel = obs_timestamps[0]
        end_rel = obs_timestamps[-1]

        st_iso = format_iso_timestamp(st_rel)
        end_iso = format_iso_timestamp(end_rel)

        # Center trajectory points [[cx, cy], ...]
        traj_points = []
        for (x, y, w, h) in boxes:
            traj_points.append([round(x + w / 2.0, 2), round(y + h / 2.0, 2)])

        occurrences = [
            {
                "frame": f_num,
                "frame_path": f_path,
                "timestamp": ts,
                "bbox": box,
            }
            for f_num, f_path, ts, box in zip(obs_frames, formatted_frame_paths, obs_timestamps, boxes)
        ]

        meta_payload = {
            "uuid": uuid_key,
            "sequence_id": seq_id,
            "camera_id": cam_id,
            "video_name": v_name,
            "num_frames": len(frames),
            "start_frame": obs_frames[0],
            "end_frame": obs_frames[-1],
        }

        cursor.execute(
            """
            INSERT OR REPLACE INTO tracks (
                id, global_id, track_id, camera_id, video_name, sequence_id,
                start_time_iso, end_time_iso, camera_timestamp_iso, camera_timestamp_sec,
                start_time, end_time, class_label, trajectory, occurrences,
                compressed_track, raw_frames, raw_boxes, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                uuid_key,
                uuid_key,
                abs(hash(uuid_key)) % 10000000,
                cam_id,
                v_name,
                seq_id,
                st_iso,
                end_iso,
                st_iso,
                st_rel,
                st_rel,
                end_rel,
                "vehicle",
                json.dumps(traj_points),
                json.dumps(occurrences),
                json.dumps(compressed_dict),
                json.dumps(frames),
                json.dumps(boxes),
                json.dumps(meta_payload),
            ),
        )
        count += 1

    conn.commit()
    logger.info(f"Ingested {count} dataset tracks from test-tracks.json into 'tracks' table.")
    return count


def build_cctv_database(
    artifacts_dir: Path,
    db_path: Path,
    retrieval_model: str,
    reid_model: str,
    test_tracks_json: Optional[Path] = None,
):
    """Compile all pipeline artifacts into the destination SQLite .db file."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        logger.info(f"Overwriting existing database at '{db_path}'")
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    init_database_schema(conn)

    video_offsets = process_video_clips(artifacts_dir, conn)

    # Ingest dataset tracks if test_tracks_json is explicitly provided and existing
    if test_tracks_json is not None:
        t_path = Path(test_tracks_json)
        if t_path.exists():
            ingest_test_tracks_json(t_path, conn)
        else:
            logger.info(f"Ground truth tracks file '{t_path}' does not exist; skipping dataset ground truth ingestion.")
    else:
        logger.info("No ground truth tracks JSON specified; skipping dataset ground truth ingestion.")

    # Helper function to compute absolute timestamps
    def get_abs_timestamps(video_name: str, rel_sec: float) -> Tuple[str, float]:
        v_meta = video_offsets.get(video_name)
        v_start_sec = v_meta[2] if v_meta else 0.0
        abs_sec = v_start_sec + float(rel_sec)
        return format_iso_timestamp(abs_sec), abs_sec

    cursor = conn.cursor()

    # ------------------------------------------------------------------
    # 1. Ingest Track Registries (JSON)
    # ------------------------------------------------------------------
    identities_path = artifacts_dir / "registry.tracks.identities.json"
    models_path = artifacts_dir / "registry.tracks.models.json"

    tracks_inserted = 0
    if identities_path.exists():
        try:
            with open(identities_path, "r") as f:
                id_data = json.load(f)
                for gid, prof in id_data.items():
                    if not isinstance(prof, dict):
                        continue
                    cam_ids = prof.get("camera_ids", ["cam_1"])
                    track_ids = prof.get("track_ids", [])
                    st_rel = float(prof.get("start_time", 0.0))
                    end_rel = float(prof.get("end_time", st_rel))
                    cls_lbl = str(prof.get("class_label", "vehicle"))

                    for raw_tid in track_ids:
                        raw_str = str(raw_tid)
                        v_name = "c001.mp4"
                        tid = 0
                        if "_" in raw_str:
                            parts = raw_str.rsplit("_", 1)
                            v_name = parts[0]
                            if len(parts) > 1 and parts[1].isdigit():
                                tid = int(parts[1])
                        elif isinstance(raw_tid, int):
                            tid = raw_tid

                        cam_id = camera_id_from_clip(v_name)
                        st_iso, st_sec = get_abs_timestamps(v_name, st_rel)
                        end_iso, end_sec = get_abs_timestamps(v_name, end_rel)

                        t_doc_id = raw_str
                        cursor.execute(
                            """
                            INSERT OR REPLACE INTO tracks (
                                id, global_id, track_id, camera_id, video_name,
                                start_time_iso, end_time_iso, camera_timestamp_iso, camera_timestamp_sec,
                                start_time, end_time, class_label, trajectory, occurrences, compressed_track, metadata
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                t_doc_id,
                                gid,
                                tid,
                                cam_id,
                                v_name,
                                st_iso,
                                end_iso,
                                st_iso,
                                st_sec,
                                st_rel,
                                end_rel,
                                cls_lbl,
                                json.dumps([]),
                                json.dumps([]),
                                json.dumps(prof),
                                json.dumps(prof),
                            ),
                        )
                        tracks_inserted += 1
        except Exception as e:
            logger.error(f"Failed loading track identities JSON: {e}")

    logger.info(f"Ingested {tracks_inserted} track records into 'tracks' table.")

    # ------------------------------------------------------------------
    # 2. Ingest Retrieval Embeddings (NPZ)
    # ------------------------------------------------------------------
    retrieval_npz = artifacts_dir / "registry.retrieval.embeddings.npz"
    retrieval_count = 0

    if retrieval_npz.exists():
        try:
            npz_data = np.load(retrieval_npz, allow_pickle=True)
            embs = npz_data.get("retrieval_embeddings", npz_data.get("embeddings"))
            metas = npz_data.get("metadatas", npz_data.get("metadata"))
            ids = npz_data.get("ids")

            if embs is not None:
                for i in range(len(embs)):
                    vec = np.array(embs[i], dtype=np.float32)
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                    vec_blob = vec.tobytes()

                    meta_dict = {}
                    if metas is not None and i < len(metas):
                        m_raw = metas[i]
                        if isinstance(m_raw, str):
                            try:
                                meta_dict = json.loads(m_raw)
                            except Exception:
                                meta_dict = {"raw": m_raw}
                        elif isinstance(m_raw, dict):
                            meta_dict = m_raw

                    doc_id = str(ids[i]) if ids is not None and i < len(ids) else f"retrieval_{i}"
                    gid = meta_dict.get("global_id")
                    v_name = meta_dict.get("video_name", "clip1.mp4")
                    raw_tid = meta_dict.get("track_id", i)

                    if isinstance(raw_tid, str) and "_" in raw_tid:
                        v_name = raw_tid.split("_")[0]
                        match = re.search(r"(\d+)$", raw_tid)
                        tid = int(match.group(1)) if match else i
                    else:
                        try:
                            tid = int(raw_tid)
                        except (ValueError, TypeError):
                            tid = i

                    cam_id = str(meta_dict.get("camera_id", camera_id_from_clip(v_name)))
                    st_rel = float(meta_dict.get("start_time", meta_dict.get("camera_timestamp", 0.0)))
                    end_rel = float(meta_dict.get("end_time", st_rel))
                    cls_lbl = str(meta_dict.get("class_label", "vehicle"))
                    crop_p = meta_dict.get("crop_path", "")
                    bbox = meta_dict.get("bbox")

                    st_iso, st_sec = get_abs_timestamps(v_name, st_rel)
                    end_iso, end_sec = get_abs_timestamps(v_name, end_rel)
                    video_pos_ms = float(meta_dict.get("video_pos_ms", st_rel * 1000.0))

                    # Auto-detect retrieval model name based on embedding vector dimension
                    v_dim = len(vec)
                    actual_model = retrieval_model
                    if retrieval_model == "google/siglip2-so400m-patch14-384":
                        if v_dim == 768:
                            actual_model = "openai/clip-vit-large-patch14"
                        elif v_dim == 512:
                            actual_model = "openai/clip-vit-base-patch32"
                        elif v_dim == 1152:
                            actual_model = "google/siglip2-so400m-patch14-384"

                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO embeddings (
                            id, global_id, track_id, camera_id, video_name,
                            embedding_type, model_name, vector_dim, embedding,
                            camera_timestamp_iso, camera_timestamp_sec, video_pos_ms,
                            start_time_iso, end_time_iso, class_label, bbox, crop_path, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            doc_id,
                            gid,
                            tid,
                            cam_id,
                            v_name,
                            "retrieval",
                            actual_model,
                            v_dim,
                            vec_blob,
                            st_iso,
                            st_sec,
                            video_pos_ms,
                            st_iso,
                            end_iso,
                            cls_lbl,
                            json.dumps(bbox) if bbox else None,
                            str(crop_p),
                            json.dumps(meta_dict),
                        ),
                    )
                    retrieval_count += 1
        except Exception as e:
            logger.error(f"Error processing retrieval embeddings NPZ: {e}")

    logger.info(f"Ingested {retrieval_count} retrieval embedding records.")

    # ------------------------------------------------------------------
    # 3. Ingest ReID Embeddings (NPZ)
    # ------------------------------------------------------------------
    reid_npz = artifacts_dir / "registry.reid.embeddings.npz"
    reid_count = 0

    if reid_npz.exists():
        try:
            npz_data = np.load(reid_npz, allow_pickle=True)
            for key in npz_data.files:
                raw_arr = npz_data[key]
                if raw_arr.ndim == 2:
                    vec = np.mean(raw_arr, axis=0)
                else:
                    vec = raw_arr

                vec = np.array(vec, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                vec_blob = vec.tobytes()

                v_name = "clip1.mp4"
                tid = 0
                if "_" in key:
                    parts = key.split("_")
                    v_name = parts[0]
                    for p in parts[1:]:
                        if p.isdigit():
                            tid = int(p)
                            break

                cam_id = camera_id_from_clip(v_name)
                st_iso, st_sec = get_abs_timestamps(v_name, 0.0)

                rec_meta = {
                    "key": key,
                    "video_name": v_name,
                    "camera_id": cam_id,
                    "track_id": tid,
                    "embedding_type": "reid",
                    "model_name": reid_model,
                }

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO embeddings (
                        id, global_id, track_id, camera_id, video_name,
                        embedding_type, model_name, vector_dim, embedding,
                        camera_timestamp_iso, camera_timestamp_sec, video_pos_ms,
                        start_time_iso, end_time_iso, class_label, bbox, crop_path, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"reid_{key}",
                        None,
                        tid,
                        cam_id,
                        v_name,
                        "reid",
                        reid_model,
                        len(vec),
                        vec_blob,
                        st_iso,
                        st_sec,
                        0.0,
                        st_iso,
                        st_iso,
                        "vehicle",
                        None,
                        None,
                        json.dumps(rec_meta),
                    ),
                )
                reid_count += 1
        except Exception as e:
            logger.error(f"Error processing ReID embeddings NPZ: {e}")

    logger.info(f"Ingested {reid_count} ReID embedding records.")

    conn.commit()
    conn.close()
    logger.info(f"Successfully compiled CCTV VLM SQL database to '{db_path}'!")


def main():
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir).resolve()
    output_db = Path(args.output_db).resolve()
    test_tracks_json = Path(args.test_tracks_json).resolve() if args.test_tracks_json else None

    logger.info(f"Building CCTV VLM SQLite Database at '{output_db}'...")
    build_cctv_database(
        artifacts_dir=artifacts_dir,
        db_path=output_db,
        retrieval_model=args.retrieval_model,
        reid_model=args.reid_model,
        test_tracks_json=test_tracks_json,
    )


if __name__ == "__main__":
    main()
