"""SQL database vector store client for retrieval encoder embedding retrieval and track metadata queries.

Supports local SQLite `.db` files (`artifacts/cctv_vlm.db`) with pgvector/cosine distance math,
serving as the sole single point of truth for VLM querying.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from shared.utils import setup_logger


def cosine_distance_func(blob1: bytes, blob2: bytes) -> float:
    """SQL custom function computing cosine distance between two float32 binary vectors."""
    if not blob1 or not blob2:
        return 1.0
    v1 = np.frombuffer(blob1, dtype=np.float32)
    v2 = np.frombuffer(blob2, dtype=np.float32)
    if len(v1) != len(v2) or len(v1) == 0:
        return 1.0
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 1.0
    sim = float(np.dot(v1, v2) / (n1 * n2))
    return max(0.0, 1.0 - sim)


class VectorStore:
    """SQL database vector store client for CCTV VLM retrieval and metadata queries."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        encoder: Optional[Any] = None,
    ) -> None:
        self.logger = setup_logger("VectorStore")
        self.encoder = encoder

        # Prefer explicit db_path or environment variable
        self.db_path = db_path or os.environ.get("DB_PATH")
        self.conn_type: str = "sql"
        self._conn: Optional[sqlite3.Connection] = None

        self._connect()

    def _connect(self) -> None:
        """Establish SQL database connection to .db file."""
        target_path: Optional[Path] = None

        if self.db_path:
            target_path = Path(self.db_path)
        else:
            workspace_root = Path(__file__).resolve().parent.parent.parent
            default_db = workspace_root / "artifacts" / "cctv_vlm.db"
            if default_db.exists():
                target_path = default_db

        if target_path and target_path.exists():
            try:
                self._conn = sqlite3.connect(str(target_path), check_same_thread=False)
                self._conn.row_factory = sqlite3.Row
                # Register cosine distance function for vector queries
                self._conn.create_function("cosine_distance", 2, cosine_distance_func)
                self.conn_type = "sqlite"
                self.db_path = str(target_path)
                self.logger.info(f"Connected to SQL database store: '{target_path.name}'")
                return
            except Exception as err:
                self.logger.error(f"Error opening SQLite database '{target_path}': {err}")

        self.logger.warning("SQL database file not found. Initializing empty in-memory SQLite database.")
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.create_function("cosine_distance", 2, cosine_distance_func)
        self.conn_type = "memory"

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20,
        where: Optional[Dict[str, Any]] = None,
        embedding_type: str = "retrieval",
        model_name: Optional[str] = None,
    ) -> List[dict]:
        """Search embedding vectors by cosine similarity using SQL queries.

        Returns candidate dicts with keys: id, metadata, distance, camera_id, track_id, camera_timestamp, video_pos_ms, bbox.
        """
        if self._conn is None:
            return []

        query_vec = np.array(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm
        query_blob = query_vec.tobytes()

        # Build SQL query conditions
        sql_conditions = ["embedding_type = ?", "vector_dim = ?"]
        params: List[Any] = [embedding_type, len(query_vec)]

        if model_name:
            sql_conditions.append("model_name = ?")
            params.append(model_name)

        if where:
            where_sql, where_params = self._build_where_sql(where)
            if where_sql:
                sql_conditions.append(f"({where_sql})")
                params.extend(where_params)

        where_clause = " AND ".join(sql_conditions)

        query_sql = f"""
            SELECT 
                id, global_id, track_id, camera_id, video_name, embedding_type, model_name,
                camera_timestamp_iso, camera_timestamp_sec, video_pos_ms, start_time_iso, end_time_iso,
                class_label, bbox, crop_path, metadata, embedding,
                cosine_distance(embedding, ?) AS distance
            FROM embeddings
            WHERE {where_clause}
            ORDER BY distance ASC
            LIMIT ?
        """
        exec_params = [query_blob] + params + [top_k]

        cursor = self._conn.cursor()
        try:
            cursor.execute(query_sql, exec_params)
            rows = cursor.fetchall()
        except Exception as err:
            self.logger.error(f"SQL search execution failed: {err}")
            return []

        candidates: List[dict] = []
        for r in rows:
            meta_dict = {}
            if r["metadata"]:
                try:
                    meta_dict = json.loads(r["metadata"])
                except Exception:
                    meta_dict = {"raw": r["metadata"]}

            # Ensure essential metadata fields are present
            meta_dict["camera_id"] = r["camera_id"]
            meta_dict["track_id"] = r["track_id"]
            meta_dict["global_id"] = r["global_id"] or r["track_id"]
            meta_dict["camera_timestamp_iso"] = r["camera_timestamp_iso"]
            meta_dict["camera_timestamp"] = r["camera_timestamp_sec"]
            meta_dict["video_pos_ms"] = r["video_pos_ms"]
            meta_dict["class_label"] = r["class_label"]
            meta_dict["crop_path"] = r["crop_path"]

            bbox_val = None
            if r["bbox"]:
                try:
                    bbox_val = json.loads(r["bbox"])
                except Exception:
                    bbox_val = None

            emb_arr = None
            if r["embedding"]:
                emb_arr = np.frombuffer(r["embedding"], dtype=np.float32).tolist()

            candidates.append({
                "id": r["id"],
                "global_id": r["global_id"],
                "track_id": r["track_id"],
                "camera_id": r["camera_id"],
                "video_name": r["video_name"],
                "camera_timestamp_iso": r["camera_timestamp_iso"],
                "camera_timestamp": r["camera_timestamp_sec"],
                "video_pos_ms": r["video_pos_ms"],
                "bbox": bbox_val,
                "class_label": r["class_label"],
                "crop_path": r["crop_path"],
                "distance": float(r["distance"]),
                "retrieval_embedding": emb_arr if r["embedding_type"] == "retrieval" else None,
                "appearance_embedding": emb_arr if r["embedding_type"] == "reid" else None,
                "metadata": meta_dict,
            })

        return candidates

    def query_metadata(
        self,
        where: Optional[Dict[str, Any]] = None,
        limit: int = 20,
    ) -> List[dict]:
        """Query metadata records from the database without vector similarity search."""
        if self._conn is None:
            return []

        sql_conditions = []
        params: List[Any] = []

        if where:
            where_sql, where_params = self._build_where_sql(where)
            if where_sql:
                sql_conditions.append(where_sql)
                params.extend(where_params)

        where_clause = " WHERE " + " AND ".join(sql_conditions) if sql_conditions else ""

        query_sql = f"""
            SELECT metadata, camera_id, track_id, global_id, camera_timestamp_iso, camera_timestamp_sec, video_pos_ms, class_label
            FROM tracks
            {where_clause}
            LIMIT ?
        """
        params.append(limit)

        cursor = self._conn.cursor()
        try:
            cursor.execute(query_sql, params)
            rows = cursor.fetchall()
        except Exception as err:
            self.logger.error(f"SQL metadata query failed: {err}")
            rows = []

        # Fallback to querying embeddings table if tracks table empty
        if not rows:
            query_sql_emb = f"""
                SELECT metadata, camera_id, track_id, global_id, camera_timestamp_iso, camera_timestamp_sec, video_pos_ms, class_label
                FROM embeddings
                {where_clause}
                LIMIT ?
            """
            try:
                cursor.execute(query_sql_emb, params)
                rows = cursor.fetchall()
            except Exception as err:
                self.logger.error(f"SQL embeddings metadata query failed: {err}")
                rows = []

        results: List[dict] = []
        for r in rows:
            meta_dict = {}
            if r["metadata"]:
                try:
                    meta_dict = json.loads(r["metadata"])
                except Exception:
                    meta_dict = {"raw": r["metadata"]}

            meta_dict["camera_id"] = r["camera_id"]
            meta_dict["track_id"] = r["track_id"]
            meta_dict["global_id"] = r["global_id"] or r["track_id"]
            meta_dict["camera_timestamp_iso"] = r["camera_timestamp_iso"]
            meta_dict["camera_timestamp"] = r["camera_timestamp_sec"]
            meta_dict["video_pos_ms"] = r["video_pos_ms"]
            meta_dict["class_label"] = r["class_label"]
            results.append(meta_dict)

        return results

    def get_event_count(self) -> int:
        """Return total number of embedding records in the database store."""
        if self._conn is None:
            return 0
        try:
            cursor = self._conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM embeddings")
            row = cursor.fetchone()
            return row[0] if row else 0
        except Exception as e:
            self.logger.error(f"Failed getting event count: {e}")
            return 0

    def _build_where_sql(self, where: Dict[str, Any]) -> Tuple[str, List[Any]]:
        """Translate structured dictionary where filters into parameterized SQL clauses."""
        clauses: List[str] = []
        params: List[Any] = []

        field_map = {
            "camera_id": "camera_id",
            "track_id": "track_id",
            "global_id": "global_id",
            "class_label": "class_label",
            "camera_timestamp": "camera_timestamp_sec",
            "camera_timestamp_sec": "camera_timestamp_sec",
            "video_name": "video_name",
            "video_pos_ms": "video_pos_ms",
        }

        for k, v in where.items():
            if k == "$and" and isinstance(v, list):
                sub_clauses = []
                for sub in v:
                    sc, sp = self._build_where_sql(sub)
                    if sc:
                        sub_clauses.append(f"({sc})")
                        params.extend(sp)
                if sub_clauses:
                    clauses.append(" AND ".join(sub_clauses))
                continue

            col_name = field_map.get(k, k)

            if isinstance(v, dict):
                for op, op_val in v.items():
                    if op in ("$gte", ">="):
                        clauses.append(f"{col_name} >= ?")
                        params.append(op_val)
                    elif op in ("$lte", "<="):
                        clauses.append(f"{col_name} <= ?")
                        params.append(op_val)
                    elif op in ("$gt", ">"):
                        clauses.append(f"{col_name} > ?")
                        params.append(op_val)
                    elif op in ("$lt", "<"):
                        clauses.append(f"{col_name} < ?")
                        params.append(op_val)
                    elif op in ("$eq", "="):
                        clauses.append(f"{col_name} = ?")
                        params.append(op_val)
            else:
                clauses.append(f"{col_name} = ?")
                params.append(v)

        return " AND ".join(clauses), params
