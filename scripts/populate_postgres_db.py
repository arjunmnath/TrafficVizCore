#!/usr/bin/env python3
"""
Database population script to set up PostgreSQL pgvector schema and populate
track events from temp.noinclude.json and temp.noinclude.npz.

Computes SigLIP2 768-dim retrieval embeddings from track candidate crops,
stores 2048-dim appearance embeddings from NPZ, and indexes metadata in PostgreSQL / Supabase
or local SQLite fallback database.
"""

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from dotenv import load_dotenv

# Add workspace root to python path to import app modules
workspace_root = Path(__file__).resolve().parent.parent
sys.path.append(str(workspace_root))

from inference_node.retrieval.encoder import get_retrieval_encoder
from shared.utils import setup_logger

logger = setup_logger("PopulatePostgresDB")

load_dotenv()

SQL_SCHEMA_DDL = """
-- PostgreSQL pgvector DDL for CCTV track events
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS track_events (
    id VARCHAR(255) PRIMARY KEY,
    camera_id TEXT NOT NULL,
    track_id INT NOT NULL,
    camera_timestamp DOUBLE PRECISION NOT NULL,
    video_pos_ms DOUBLE PRECISION NOT NULL,
    bbox TEXT,
    class_label TEXT,
    start_time DOUBLE PRECISION,
    end_time DOUBLE PRECISION,
    trajectory JSONB,
    appearance_embedding vector(2048),
    retrieval_embedding vector(768),
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS track_events_retrieval_idx 
ON track_events USING hnsw (retrieval_embedding vector_cosine_ops);
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Populate PostgreSQL pgvector database with track events and SigLIP2 embeddings."
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default=str(workspace_root / "temp.noinclude.json"),
        help="Path to temp.noinclude.json file containing track trajectories and metadata.",
    )
    parser.add_argument(
        "--npz_path",
        type=str,
        default=str(workspace_root / "temp.noinclude.npz"),
        help="Path to temp.noinclude.npz file containing appearance embeddings.",
    )
    parser.add_argument(
        "--crops_dir",
        type=str,
        default=str(workspace_root / "crops.noinclude"),
        help="Directory containing candidate crop images (e.g. clip1_track_1.jpg).",
    )
    parser.add_argument(
        "--reid_crops_dir",
        type=str,
        default=str(workspace_root / "reid_crops_cleaned"),
        help="Fallback directory for ReID crops organised by track_id.",
    )
    parser.add_argument(
        "--table_name",
        type=str,
        default="track_events",
        help="Target PostgreSQL table name.",
    )
    parser.add_argument(
        "--postgres_url",
        type=str,
        default=os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL"),
        help="PostgreSQL connection string (e.g. postgresql://user:password@host:port/dbname).",
    )
    parser.add_argument(
        "--supabase_url",
        type=str,
        default=os.environ.get("SUPABASE_URL"),
        help="Supabase project URL.",
    )
    parser.add_argument(
        "--supabase_key",
        type=str,
        default=os.environ.get("SUPABASE_KEY"),
        help="Supabase API key.",
    )
    parser.add_argument(
        "--local_db_path",
        type=str,
        default=str(workspace_root / "cctv_vector.db"),
        help="Local SQLite fallback database path.",
    )
    parser.add_argument(
        "--retrieval_model",
        type=str,
        default="google/siglip2-base-patch16-224",
        help="Retrieval encoder model name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run inference on (auto, cuda, mps, cpu).",
    )
    parser.add_argument(
        "--reset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop and recreate target table before indexing.",
    )
    return parser.parse_args()


def camera_id_from_clip(clip_name: str) -> str:
    """Derive camera identifier from clip filename (e.g. 'clip1.mp4' -> 'cam_1')."""
    stem = clip_name.replace(".mp4", "")
    if stem.startswith("clip"):
        num = stem[4:]
        if num.isdigit():
            return f"cam_{num}"
    return stem


def find_candidate_crop(
    video_name: str, track_id: int, crops_dir: Path, reid_crops_dir: Path
) -> Path | None:
    """Find candidate crop image file for a given track."""
    clip_stem = video_name.replace(".mp4", "")

    # Primary check: crops.noinclude/clip1_track_1.jpg
    for ext in [".jpg", ".jpeg", ".png"]:
        p = crops_dir / f"{clip_stem}_track_{track_id}{ext}"
        if p.exists():
            return p

    # Secondary check: reid_crops_cleaned/{track_id}/
    sub = reid_crops_dir / str(track_id)
    if sub.exists():
        clip_files = sorted(list(sub.glob(f"{clip_stem}_*")))
        if clip_files:
            return clip_files[0]
        all_files = sorted(list(sub.glob("*")))
        if all_files:
            return all_files[0]

    return None


def get_db_connection(args):
    """Establish connection to PostgreSQL via psycopg2, Supabase SDK, or SQLite fallback."""
    if args.postgres_url:
        try:
            import psycopg2

            logger.info("Connecting to PostgreSQL database via psycopg2...")
            conn = psycopg2.connect(args.postgres_url)
            conn.autocommit = True
            return ("psycopg2", conn)
        except Exception as e:
            logger.warning(f"Failed to connect via psycopg2: {e}")

    if args.supabase_url and args.supabase_key:
        try:
            from supabase import create_client

            logger.info(f"Connecting to Supabase at {args.supabase_url}...")
            client = create_client(args.supabase_url, args.supabase_key)
            return ("supabase", client)
        except Exception as e:
            logger.warning(f"Failed to connect via Supabase client: {e}")

    logger.info(f"Using local SQLite database fallback at {args.local_db_path}...")
    conn = sqlite3.connect(args.local_db_path)
    return ("sqlite", conn)


def setup_schema_psycopg2(conn, table_name: str, reset: bool):
    """Create pgvector extension and table schema using raw SQL connection."""
    with conn.cursor() as cur:
        logger.info("Enabling pgvector extension...")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        if reset:
            logger.info(f"Resetting table '{table_name}'...")
            cur.execute(f"DROP TABLE IF EXISTS {table_name};")

        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id VARCHAR(255) PRIMARY KEY,
            camera_id TEXT NOT NULL,
            track_id INT NOT NULL,
            camera_timestamp DOUBLE PRECISION NOT NULL,
            video_pos_ms DOUBLE PRECISION NOT NULL,
            bbox TEXT,
            class_label TEXT,
            start_time DOUBLE PRECISION,
            end_time DOUBLE PRECISION,
            trajectory JSONB,
            appearance_embedding vector(2048),
            retrieval_embedding vector(768),
            metadata JSONB
        );
        """
        cur.execute(create_table_sql)
        try:
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {table_name}_retrieval_idx ON {table_name} USING hnsw (retrieval_embedding vector_cosine_ops);"
            )
        except Exception as e:
            logger.warning(f"Could not create HNSW index: {e}")

    logger.info(f"Schema setup complete for table '{table_name}'.")


def setup_schema_sqlite(conn, table_name: str, reset: bool):
    """Create SQLite table schema for local vector database fallback."""
    cur = conn.cursor()
    if reset:
        cur.execute(f"DROP TABLE IF EXISTS {table_name};")

    cur.execute(
        f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id TEXT PRIMARY KEY,
        camera_id TEXT NOT NULL,
        track_id INTEGER NOT NULL,
        camera_timestamp REAL NOT NULL,
        video_pos_ms REAL NOT NULL,
        bbox TEXT,
        class_label TEXT,
        start_time REAL,
        end_time REAL,
        trajectory TEXT,
        appearance_embedding TEXT,
        retrieval_embedding TEXT,
        metadata TEXT
    );
    """
    )
    conn.commit()
    logger.info(f"SQLite schema setup complete for table '{table_name}'.")


def populate_db():
    args = parse_args()

    print("\n┌────────────────────────────────────────────────────────────┐")
    print("│         PostgreSQL pgvector Prepopulation Pipeline         │")
    print("└────────────────────────────────────────────────────────────┘\n")

    conn_type, db = get_db_connection(args)

    if conn_type == "psycopg2":
        setup_schema_psycopg2(db, args.table_name, args.reset)
    elif conn_type == "sqlite":
        setup_schema_sqlite(db, args.table_name, args.reset)

    logger.info(f"Loading track metadata from {args.json_path}...")
    with open(args.json_path, "r") as f:
        json_data = json.load(f)

    logger.info(f"Loading appearance embeddings from {args.npz_path}...")
    npz_data = np.load(args.npz_path)

    logger.info(f"Initializing SigLIP2 retrieval encoder ({args.retrieval_model})...")
    encoder = get_retrieval_encoder(model_name=args.retrieval_model, device=args.device)

    crops_dir = Path(args.crops_dir)
    reid_crops_dir = Path(args.reid_crops_dir)

    total_inserted = 0

    for video_name, tracks in json_data.items():
        camera_id = camera_id_from_clip(video_name)
        logger.info(f"Processing video {video_name} ({camera_id}) with {len(tracks)} tracks...")

        for item in tracks:
            track_id = item.get("track_id")
            compressed_track = item.get("compressed_track") or {}
            if not isinstance(compressed_track, dict):
                compressed_track = {}
            class_label = compressed_track.get("class", item.get("class", "object"))
            start_time = float(compressed_track.get("start_time", item.get("start_time", 0.0)))
            end_time = float(compressed_track.get("end_time", item.get("end_time", 0.0)))
            trajectory = compressed_track.get("trajectory") or item.get("trajectory") or {}

            event_id = f"{camera_id}_{track_id}_{start_time:.4f}"

            # 1. Appearance embedding from NPZ
            npz_key = f"{video_name}_app_{track_id}"
            app_emb = None
            if npz_key in npz_data:
                raw_app = npz_data[npz_key]
                if raw_app.ndim == 2:
                    app_emb = np.mean(raw_app, axis=0)
                else:
                    app_emb = raw_app
                norm = np.linalg.norm(app_emb)
                if norm > 0:
                    app_emb = app_emb / norm

            # 2. Candidate crop and SigLIP2 retrieval embedding
            crop_path = find_candidate_crop(video_name, track_id, crops_dir, reid_crops_dir)
            retrieval_emb = None

            if crop_path:
                try:
                    with Image.open(crop_path) as img:
                        retrieval_emb = encoder.encode_image(img)
                except Exception as e:
                    logger.error(f"Failed to encode crop {crop_path}: {e}")

            if retrieval_emb is None:
                logger.warning(f"No candidate crop found for {video_name} track {track_id}, fallback to zero vector.")
                retrieval_emb = np.zeros(768, dtype=np.float32)

            metadata_dict = {
                "camera_id": camera_id,
                "track_id": track_id,
                "camera_timestamp": start_time,
                "video_pos_ms": start_time * 1000.0,
                "class_label": class_label,
                "start_time": start_time,
                "end_time": end_time,
                "video_name": video_name,
            }

            app_emb_list = app_emb.tolist() if app_emb is not None else None
            retrieval_emb_list = retrieval_emb.tolist()

            if conn_type == "psycopg2":
                with db.cursor() as cur:
                    insert_sql = f"""
                    INSERT INTO {args.table_name} (
                        id, camera_id, track_id, camera_timestamp, video_pos_ms,
                        class_label, start_time, end_time, trajectory,
                        appearance_embedding, retrieval_embedding, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        camera_id = EXCLUDED.camera_id,
                        track_id = EXCLUDED.track_id,
                        camera_timestamp = EXCLUDED.camera_timestamp,
                        video_pos_ms = EXCLUDED.video_pos_ms,
                        class_label = EXCLUDED.class_label,
                        start_time = EXCLUDED.start_time,
                        end_time = EXCLUDED.end_time,
                        trajectory = EXCLUDED.trajectory,
                        appearance_embedding = EXCLUDED.appearance_embedding,
                        retrieval_embedding = EXCLUDED.retrieval_embedding,
                        metadata = EXCLUDED.metadata;
                    """
                    cur.execute(
                        insert_sql,
                        (
                            event_id,
                            camera_id,
                            track_id,
                            start_time,
                            start_time * 1000.0,
                            class_label,
                            start_time,
                            end_time,
                            json.dumps(trajectory),
                            app_emb_list,
                            retrieval_emb_list,
                            json.dumps(metadata_dict),
                        ),
                    )
            elif conn_type == "sqlite":
                cur = db.cursor()
                insert_sql = f"""
                INSERT OR REPLACE INTO {args.table_name} (
                    id, camera_id, track_id, camera_timestamp, video_pos_ms,
                    class_label, start_time, end_time, trajectory,
                    appearance_embedding, retrieval_embedding, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """
                cur.execute(
                    insert_sql,
                    (
                        event_id,
                        camera_id,
                        track_id,
                        start_time,
                        start_time * 1000.0,
                        class_label,
                        start_time,
                        end_time,
                        json.dumps(trajectory),
                        json.dumps(app_emb_list),
                        json.dumps(retrieval_emb_list),
                        json.dumps(metadata_dict),
                    ),
                )
                db.commit()
            else:
                row_data = {
                    "id": event_id,
                    "camera_id": camera_id,
                    "track_id": track_id,
                    "camera_timestamp": start_time,
                    "video_pos_ms": start_time * 1000.0,
                    "class_label": class_label,
                    "start_time": start_time,
                    "end_time": end_time,
                    "trajectory": trajectory,
                    "appearance_embedding": app_emb_list,
                    "retrieval_embedding": retrieval_emb_list,
                    "metadata": metadata_dict,
                }
                try:
                    db.table(args.table_name).upsert(row_data).execute()
                except Exception as err:
                    logger.error(
                        f"Supabase upsert error for event '{event_id}': {err}\n"
                        "Note: Ensure table 'track_events' exists in Supabase SQL Editor:\n"
                        f"{SQL_SCHEMA_DDL}"
                    )
                    break

            total_inserted += 1

    print("\n┌────────────────────────────────────────────────────────────┐")
    print("│             PREPOPULATION COMPLETE 🎉                      │")
    print(f"│  Total Inserted Records: {total_inserted:<33} │")
    print("└────────────────────────────────────────────────────────────┘\n")


if __name__ == "__main__":
    populate_db()
