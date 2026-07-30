"""PostgreSQL pgvector wrapper for SigLIP2 embedding search with metadata filtering."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from shared.utils import setup_logger


class VectorStore:
    """PostgreSQL pgvector client for SigLIP2 embedding retrieval and track metadata queries."""

    def __init__(
        self,
        table_name: str = "track_events",
        postgres_url: Optional[str] = None,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        local_db_path: Optional[str] = None,
    ) -> None:
        self.logger = setup_logger("VectorStore")
        self.table_name = table_name

        self.postgres_url = (
            postgres_url
            or os.environ.get("POSTGRES_URL")
            or os.environ.get("DATABASE_URL")
        )
        self.supabase_url = supabase_url or os.environ.get("SUPABASE_URL")
        self.supabase_key = supabase_key or os.environ.get("SUPABASE_KEY")
        self.local_db_path = local_db_path or str(
            Path(__file__).resolve().parent.parent.parent / "cctv_vector.db"
        )

        self.conn_type = None
        self.conn = None
        self.supabase_client = None

        self._connect()

    def _connect(self) -> None:
        if self.postgres_url:
            try:
                import psycopg2

                self.conn = psycopg2.connect(self.postgres_url)
                self.conn.autocommit = True
                self.conn_type = "psycopg2"
                self.logger.info(
                    f"Connected to PostgreSQL pgvector: table='{self.table_name}'"
                )
                return
            except Exception as err:
                self.logger.warning(f"Could not connect via psycopg2: {err}")

        if self.supabase_url and self.supabase_key:
            try:
                from supabase import create_client

                self.supabase_client = create_client(self.supabase_url, self.supabase_key)
                self.conn_type = "supabase"
                self.logger.info(
                    f"Connected to Supabase at '{self.supabase_url}': table='{self.table_name}'"
                )
                return
            except Exception as err:
                self.logger.warning(f"Could not connect via Supabase client: {err}")

        if os.path.exists(self.local_db_path):
            try:
                self.conn = sqlite3.connect(self.local_db_path)
                self.conn_type = "sqlite"
                self.logger.info(
                    f"Connected to local vector database fallback: path='{self.local_db_path}' table='{self.table_name}'"
                )
                return
            except Exception as err:
                self.logger.warning(f"Could not connect to local SQLite database: {err}")

        self.logger.warning("Initializing empty in-memory vector store connection.")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[dict]:
        """Search by embedding vector using cosine similarity with optional metadata filters.

        Returns dicts with keys: id, metadata, distance.
        """
        emb_list = query_embedding.tolist()
        candidates: List[dict] = []

        if self.conn_type == "psycopg2" and self.conn:
            where_sql, params = self._parse_where_to_sql(where)
            query_sql = f"""
            SELECT id, camera_id, track_id, camera_timestamp, video_pos_ms,
                   bbox, class_label, start_time, end_time, trajectory, metadata,
                   (retrieval_embedding <=> %s::vector) AS distance
            FROM {self.table_name}
            {where_sql}
            ORDER BY retrieval_embedding <=> %s::vector
            LIMIT %s;
            """
            full_params = [emb_list] + params + [emb_list, top_k]

            try:
                with self.conn.cursor() as cur:
                    cur.execute(query_sql, full_params)
                    rows = cur.fetchall()

                for row in rows:
                    (
                        doc_id,
                        camera_id,
                        track_id,
                        camera_timestamp,
                        video_pos_ms,
                        bbox,
                        class_label,
                        start_time,
                        end_time,
                        trajectory,
                        metadata,
                        distance,
                    ) = row

                    meta_dict = metadata if isinstance(metadata, dict) else {}
                    meta_dict.update(
                        {
                            "camera_id": camera_id,
                            "track_id": track_id,
                            "camera_timestamp": camera_timestamp,
                            "video_pos_ms": video_pos_ms,
                            "bbox": bbox,
                            "class_label": class_label,
                            "start_time": start_time,
                            "end_time": end_time,
                            "trajectory": trajectory,
                        }
                    )

                    candidates.append(
                        {
                            "id": doc_id,
                            "metadata": meta_dict,
                            "distance": float(distance) if distance is not None else 1.0,
                        }
                    )
            except Exception as err:
                self.logger.error(f"Error performing vector search via psycopg2: {err}")

        elif self.conn_type == "sqlite" and self.conn:
            where_sql, params = self._parse_where_to_sql_sqlite(where)
            query_sql = f"""
            SELECT id, camera_id, track_id, camera_timestamp, video_pos_ms,
                   bbox, class_label, start_time, end_time, trajectory, retrieval_embedding, metadata
            FROM {self.table_name}
            {where_sql};
            """
            try:
                cur = self.conn.cursor()
                cur.execute(query_sql, params)
                rows = cur.fetchall()

                scored_rows = []
                query_norm = np.linalg.norm(query_embedding)
                if query_norm == 0:
                    query_norm = 1.0

                for row in rows:
                    (
                        doc_id,
                        camera_id,
                        track_id,
                        camera_timestamp,
                        video_pos_ms,
                        bbox,
                        class_label,
                        start_time,
                        end_time,
                        trajectory,
                        retrieval_emb_str,
                        metadata,
                    ) = row

                    db_emb = None
                    if retrieval_emb_str:
                        db_emb = np.array(json.loads(retrieval_emb_str), dtype=np.float32)

                    if db_emb is not None and db_emb.size > 0:
                        db_norm = np.linalg.norm(db_emb)
                        if db_norm == 0:
                            db_norm = 1.0
                        similarity = float(np.dot(query_embedding, db_emb) / (query_norm * db_norm))
                        distance = 1.0 - similarity
                    else:
                        distance = 1.0

                    meta_dict = json.loads(metadata) if isinstance(metadata, str) else (metadata or {})
                    meta_dict.update(
                        {
                            "camera_id": camera_id,
                            "track_id": track_id,
                            "camera_timestamp": camera_timestamp,
                            "video_pos_ms": video_pos_ms,
                            "bbox": bbox,
                            "class_label": class_label,
                            "start_time": start_time,
                            "end_time": end_time,
                        }
                    )

                    scored_rows.append(
                        {
                            "id": doc_id,
                            "metadata": meta_dict,
                            "distance": distance,
                        }
                    )

                scored_rows.sort(key=lambda x: x["distance"])
                candidates = scored_rows[:top_k]
            except Exception as err:
                self.logger.error(f"Error searching via SQLite fallback: {err}")

        elif self.conn_type == "supabase" and self.supabase_client:
            try:
                rpc_res = self.supabase_client.rpc(
                    "match_track_events",
                    {
                        "query_embedding": emb_list,
                        "match_count": top_k,
                        "filter_camera_id": where.get("camera_id") if where else None,
                    },
                ).execute()

                if rpc_res.data:
                    for row in rpc_res.data:
                        meta = row.get("metadata", {})
                        meta.update(row)
                        candidates.append(
                            {
                                "id": row.get("id"),
                                "metadata": meta,
                                "distance": float(row.get("distance", 0.0)),
                            }
                        )
                else:
                    q = self.supabase_client.table(self.table_name).select("*").limit(top_k)
                    res = q.execute()
                    for row in res.data or []:
                        meta = row.get("metadata", {})
                        meta.update(row)
                        candidates.append({"id": row.get("id"), "metadata": meta, "distance": 0.0})
            except Exception as err:
                self.logger.error(f"Error searching via Supabase client: {err}")

        return candidates

    def query_metadata(
        self,
        where: Optional[Dict[str, Any]] = None,
        limit: int = 20,
    ) -> List[dict]:
        """Query metadata records from the database without vector search."""
        results: List[dict] = []

        if self.conn_type == "psycopg2" and self.conn:
            where_sql, params = self._parse_where_to_sql(where)
            query_sql = f"""
            SELECT id, camera_id, track_id, camera_timestamp, video_pos_ms,
                   bbox, class_label, start_time, end_time, metadata
            FROM {self.table_name}
            {where_sql}
            LIMIT %s;
            """
            full_params = params + [limit]
            try:
                with self.conn.cursor() as cur:
                    cur.execute(query_sql, full_params)
                    rows = cur.fetchall()
                for row in rows:
                    (
                        doc_id,
                        camera_id,
                        track_id,
                        camera_timestamp,
                        video_pos_ms,
                        bbox,
                        class_label,
                        start_time,
                        end_time,
                        metadata,
                    ) = row
                    meta_dict = metadata if isinstance(metadata, dict) else {}
                    meta_dict.update(
                        {
                            "id": doc_id,
                            "camera_id": camera_id,
                            "track_id": track_id,
                            "camera_timestamp": camera_timestamp,
                            "video_pos_ms": video_pos_ms,
                            "bbox": bbox,
                            "class_label": class_label,
                            "start_time": start_time,
                            "end_time": end_time,
                        }
                    )
                    results.append(meta_dict)
            except Exception as err:
                self.logger.error(f"Error querying metadata via psycopg2: {err}")

        elif self.conn_type == "sqlite" and self.conn:
            where_sql, params = self._parse_where_to_sql_sqlite(where)
            query_sql = f"""
            SELECT id, camera_id, track_id, camera_timestamp, video_pos_ms,
                   bbox, class_label, start_time, end_time, metadata
            FROM {self.table_name}
            {where_sql}
            LIMIT ?;
            """
            full_params = params + [limit]
            try:
                cur = self.conn.cursor()
                cur.execute(query_sql, full_params)
                rows = cur.fetchall()
                for row in rows:
                    (
                        doc_id,
                        camera_id,
                        track_id,
                        camera_timestamp,
                        video_pos_ms,
                        bbox,
                        class_label,
                        start_time,
                        end_time,
                        metadata,
                    ) = row
                    meta_dict = json.loads(metadata) if isinstance(metadata, str) else (metadata or {})
                    meta_dict.update(
                        {
                            "id": doc_id,
                            "camera_id": camera_id,
                            "track_id": track_id,
                            "camera_timestamp": camera_timestamp,
                            "video_pos_ms": video_pos_ms,
                            "bbox": bbox,
                            "class_label": class_label,
                            "start_time": start_time,
                            "end_time": end_time,
                        }
                    )
                    results.append(meta_dict)
            except Exception as err:
                self.logger.error(f"Error querying metadata via SQLite: {err}")

        elif self.conn_type == "supabase" and self.supabase_client:
            try:
                q = self.supabase_client.table(self.table_name).select("*")
                if where:
                    for k, v in where.items():
                        if not k.startswith("$"):
                            q = q.eq(k, v)
                q = q.limit(limit)
                res = q.execute()
                for row in res.data or []:
                    meta = row.get("metadata", {})
                    meta.update(row)
                    results.append(meta)
            except Exception as err:
                self.logger.error(f"Error querying metadata via Supabase: {err}")

        return results

    def get_event_count(self) -> int:
        """Return total number of records stored in the track_events table."""
        if self.conn_type == "psycopg2" and self.conn:
            try:
                with self.conn.cursor() as cur:
                    cur.execute(f"SELECT COUNT(*) FROM {self.table_name};")
                    row = cur.fetchone()
                    return row[0] if row else 0
            except Exception as err:
                self.logger.error(f"Error getting count: {err}")
                return 0

        elif self.conn_type == "sqlite" and self.conn:
            try:
                cur = self.conn.cursor()
                cur.execute(f"SELECT COUNT(*) FROM {self.table_name};")
                row = cur.fetchone()
                return row[0] if row else 0
            except Exception as err:
                self.logger.error(f"Error getting count via SQLite: {err}")
                return 0

        elif self.conn_type == "supabase" and self.supabase_client:
            try:
                res = self.supabase_client.table(self.table_name).select("id", count="exact").execute()
                return res.count or 0
            except Exception as err:
                self.logger.error(f"Error getting count via Supabase: {err}")
                return 0

        return 0

    @staticmethod
    def _parse_where_to_sql(where: Optional[Dict[str, Any]]) -> tuple[str, list]:
        """Convert filter dictionary to SQL WHERE clause and parameter list (PostgreSQL %s format)."""
        if not where:
            return "", []

        conditions: List[str] = []
        params: List[Any] = []

        def process_dict(d: Dict[str, Any]) -> None:
            for k, v in d.items():
                if k == "$and" and isinstance(v, list):
                    for sub in v:
                        if isinstance(sub, dict):
                            process_dict(sub)
                elif isinstance(v, dict):
                    for op, val in v.items():
                        if op == "$gte":
                            conditions.append(f"{k} >= %s")
                            params.append(val)
                        elif op == "$lte":
                            conditions.append(f"{k} <= %s")
                            params.append(val)
                        elif op == "$gt":
                            conditions.append(f"{k} > %s")
                            params.append(val)
                        elif op == "$lt":
                            conditions.append(f"{k} < %s")
                            params.append(val)
                        elif op == "$eq":
                            conditions.append(f"{k} = %s")
                            params.append(val)
                else:
                    conditions.append(f"{k} = %s")
                    params.append(v)

        process_dict(where)
        if not conditions:
            return "", []
        return "WHERE " + " AND ".join(conditions), params

    @staticmethod
    def _parse_where_to_sql_sqlite(where: Optional[Dict[str, Any]]) -> tuple[str, list]:
        """Convert filter dictionary to SQLite WHERE clause and parameter list (? format)."""
        if not where:
            return "", []

        conditions: List[str] = []
        params: List[Any] = []

        def process_dict(d: Dict[str, Any]) -> None:
            for k, v in d.items():
                if k == "$and" and isinstance(v, list):
                    for sub in v:
                        if isinstance(sub, dict):
                            process_dict(sub)
                elif isinstance(v, dict):
                    for op, val in v.items():
                        if op == "$gte":
                            conditions.append(f"{k} >= ?")
                            params.append(val)
                        elif op == "$lte":
                            conditions.append(f"{k} <= ?")
                            params.append(val)
                        elif op == "$gt":
                            conditions.append(f"{k} > ?")
                            params.append(val)
                        elif op == "$lt":
                            conditions.append(f"{k} < ?")
                            params.append(val)
                        elif op == "$eq":
                            conditions.append(f"{k} = ?")
                            params.append(val)
                else:
                    conditions.append(f"{k} = ?")
                    params.append(v)

        process_dict(where)
        if not conditions:
            return "", []
        return "WHERE " + " AND ".join(conditions), params
