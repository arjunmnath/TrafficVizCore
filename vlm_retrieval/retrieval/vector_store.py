"""PostgreSQL pgvector wrapper for SigLIP2 embedding search with metadata filtering."""

from __future__ import annotations

import json
import os
import re
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
        npz_dir: Optional[str] = None,
        npz_path: Optional[str] = None,
        json_path: Optional[str] = None,
        encoder: Optional[Any] = None,
    ) -> None:
        self.logger = setup_logger("VectorStore")
        self.table_name = table_name
        self.encoder = encoder

        self.npz_dir = npz_dir or os.environ.get("NPZ_DIR")
        self.npz_path = npz_path or os.environ.get("NPZ_PATH")
        self.json_path = json_path or os.environ.get("JSON_PATH")

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
        self.npz_records: List[Dict[str, Any]] = []

        self._connect()

    def _connect(self) -> None:
        if self.npz_dir or self.npz_path:
            self._load_from_npz(self.npz_dir or self.npz_path, self.json_path)
            self.conn_type = "npz"
            self.logger.info(
                f"Connected to NPZ vector store: loaded {len(self.npz_records)} records"
            )
            return

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

        # Workspace fallback to NPZ if available
        workspace_root = Path(__file__).resolve().parent.parent.parent
        default_npz = workspace_root / "temp.noinclude.npz"
        default_json = workspace_root / "temp.noinclude.json"
        if default_npz.exists():
            self._load_from_npz(str(default_npz), str(default_json) if default_json.exists() else None)
            if self.npz_records:
                self.conn_type = "npz"
                self.logger.info(
                    f"Connected to NPZ workspace fallback: loaded {len(self.npz_records)} records from '{default_npz.name}'"
                )
                return

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

        elif self.conn_type == "npz":
            query_norm = np.linalg.norm(query_embedding)
            if query_norm == 0:
                query_norm = 1.0

            scored = []
            for rec in self.npz_records:
                if not self._matches_where(rec, where):
                    continue

                rec_emb = rec.get("embedding")
                if rec.get("retrieval_embedding") is not None:
                    r_arr = np.array(rec["retrieval_embedding"], dtype=np.float32)
                    if r_arr.shape[0] == query_embedding.shape[0]:
                        rec_emb = r_arr

                if rec_emb is None or rec_emb.shape[0] != query_embedding.shape[0]:
                    crop_emb = self._encode_crop_if_needed(rec, target_dim=query_embedding.shape[0])
                    if crop_emb is not None and crop_emb.shape[0] == query_embedding.shape[0]:
                        rec_emb = crop_emb
                        rec["retrieval_embedding"] = crop_emb.tolist()

                if rec_emb is None or rec_emb.ndim == 0 or rec_emb.shape[0] != query_embedding.shape[0]:
                    dist = 1.0
                else:
                    r_norm = np.linalg.norm(rec_emb)
                    if r_norm == 0:
                        r_norm = 1.0
                    sim = float(np.dot(query_embedding, rec_emb) / (query_norm * r_norm))
                    dist = 1.0 - sim

                scored.append({
                    "id": rec["id"],
                    "metadata": rec["metadata"],
                    "distance": dist,
                    "retrieval_embedding": rec.get("retrieval_embedding"),
                    "appearance_embedding": rec.get("appearance_embedding"),
                })

            scored.sort(key=lambda x: x["distance"])
            candidates = scored[:top_k]

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

        elif self.conn_type == "npz":
            for rec in self.npz_records:
                if self._matches_where(rec, where):
                    results.append(rec["metadata"])
                    if len(results) >= limit:
                        break

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

        elif self.conn_type == "npz":
            return len(self.npz_records)

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

    def _load_from_npz(self, npz_target: str, json_target: Optional[str] = None) -> None:
        """Load track embeddings and metadata from an .npz file or directory."""
        self.npz_records = []
        target_path = Path(npz_target)

        npz_files: List[Path] = []
        if target_path.is_dir():
            npz_files = sorted(list(target_path.glob("*.npz")))
        elif target_path.is_file() and target_path.suffix == ".npz":
            npz_files = [target_path]

        if not npz_files:
            self.logger.warning(f"No .npz files found at path '{npz_target}'")
            return

        json_data: Dict[str, Any] = {}
        json_paths: List[Path] = []
        if json_target and Path(json_target).exists():
            json_paths = [Path(json_target)]
        elif target_path.is_dir():
            json_paths = sorted(list(target_path.glob("*.json")))
        elif target_path.is_file():
            stem_json = target_path.with_suffix(".json")
            if stem_json.exists():
                json_paths = [stem_json]

        for jp in json_paths:
            try:
                with open(jp, "r") as f:
                    content = json.load(f)
                    if isinstance(content, dict):
                        json_data.update(content)
            except Exception as e:
                self.logger.warning(f"Failed loading JSON metadata from {jp}: {e}")

        def camera_id_from_clip(clip_name: str) -> str:
            stem = clip_name.replace(".mp4", "")
            if stem.startswith("clip") and stem[4:].isdigit():
                return f"cam_{stem[4:]}"
            return stem

        for npz_file in npz_files:
            try:
                npz_data = np.load(npz_file, allow_pickle=True)
            except Exception as e:
                self.logger.error(f"Error reading NPZ file {npz_file}: {e}")
                continue

            if "retrieval_embeddings" in npz_data or "embeddings" in npz_data:
                embs = npz_data.get("retrieval_embeddings", npz_data.get("embeddings"))
                metas = npz_data.get("metadata", npz_data.get("metadatas"))
                ids = npz_data.get("ids")
                if embs is not None:
                    n_samples = len(embs)
                    for i in range(n_samples):
                        vec = np.array(embs[i], dtype=np.float32)
                        norm = np.linalg.norm(vec)
                        if norm > 0:
                            vec = vec / norm

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

                        doc_id = str(ids[i]) if ids is not None and i < len(ids) else f"npz_{npz_file.stem}_{i}"
                        cam_id = str(meta_dict.get("camera_id", "cam_1"))
                        raw_tid = meta_dict.get("track_id", i)
                        try:
                            track_id = int(raw_tid)
                        except (ValueError, TypeError):
                            # Extract numeric digits if present, else keep raw_tid as string
                            match = re.search(r"(\d+)$", str(raw_tid))
                            track_id = int(match.group(1)) if match else str(raw_tid)
                        start_time = float(meta_dict.get("start_time", meta_dict.get("camera_timestamp", 0.0)))

                        rec = {
                            "id": doc_id,
                            "camera_id": cam_id,
                            "track_id": track_id,
                            "camera_timestamp": start_time,
                            "video_pos_ms": float(meta_dict.get("video_pos_ms", start_time * 1000.0)),
                            "bbox": meta_dict.get("bbox"),
                            "class_label": str(meta_dict.get("class_label", "object")),
                            "start_time": start_time,
                            "end_time": float(meta_dict.get("end_time", start_time)),
                            "trajectory": meta_dict.get("trajectory"),
                            "embedding": vec,
                            "metadata": meta_dict,
                        }
                        self.npz_records.append(rec)
                continue

            if json_data:
                for video_name, tracks in json_data.items():
                    if not isinstance(tracks, list):
                        continue
                    cam_id = camera_id_from_clip(video_name)
                    for item in tracks:
                        if not isinstance(item, dict):
                            continue
                        tid = item.get("track_id")
                        comp_track = item.get("compressed_track") or {}
                        if not isinstance(comp_track, dict):
                            comp_track = {}

                        cls_lbl = comp_track.get("class", item.get("class", "object"))
                        st_time = float(comp_track.get("start_time", item.get("start_time", 0.0)))
                        end_t = float(comp_track.get("end_time", item.get("end_time", 0.0)))
                        traj = comp_track.get("trajectory") or item.get("trajectory") or {}

                        app_candidate_keys = [
                            f"{video_name}_app_{tid}",
                            f"{video_name}_smooth_{tid}",
                            f"{video_name}_occ_{tid}",
                            f"{video_name}_{tid}",
                            f"{cam_id}_{tid}",
                        ]
                        retrieval_candidate_keys = [
                            f"{video_name}_retrieval_{tid}",
                            f"{video_name}_siglip_{tid}",
                            f"{cam_id}_retrieval_{tid}",
                        ]

                        app_vec = None
                        for ck in app_candidate_keys:
                            if ck in npz_data:
                                raw_arr = npz_data[ck]
                                if raw_arr.ndim == 2:
                                    app_vec = np.mean(raw_arr, axis=0)
                                else:
                                    app_vec = raw_arr
                                break

                        retrieval_vec = None
                        for ck in retrieval_candidate_keys:
                            if ck in npz_data:
                                raw_arr = npz_data[ck]
                                if raw_arr.ndim == 2:
                                    retrieval_vec = np.mean(raw_arr, axis=0)
                                else:
                                    retrieval_vec = raw_arr
                                break

                        if retrieval_vec is None:
                            retrieval_vec = app_vec
                        if app_vec is None:
                            app_vec = retrieval_vec

                        if app_vec is not None:
                            app_vec = np.array(app_vec, dtype=np.float32)
                            norm = np.linalg.norm(app_vec)
                            if norm > 0:
                                app_vec = app_vec / norm

                        if retrieval_vec is not None:
                            retrieval_vec = np.array(retrieval_vec, dtype=np.float32)
                            norm = np.linalg.norm(retrieval_vec)
                            if norm > 0:
                                retrieval_vec = retrieval_vec / norm

                        if retrieval_vec is not None or app_vec is not None:
                            primary_vec = retrieval_vec if retrieval_vec is not None else app_vec
                            evt_id = f"{cam_id}_{tid}_{st_time:.4f}"
                            meta_dict = {
                                "camera_id": cam_id,
                                "track_id": tid,
                                "global_id": item.get("global_id", tid),
                                "camera_timestamp": st_time,
                                "video_pos_ms": st_time * 1000.0,
                                "class_label": cls_lbl,
                                "start_time": st_time,
                                "end_time": end_t,
                                "video_name": video_name,
                                "trajectory": traj,
                                "compressed_track": comp_track,
                                "occurrences": item.get("occurrences"),
                                "track_details": item,  # Full track metadata record from registry JSON
                            }
                            rec = {
                                "id": evt_id,
                                "camera_id": cam_id,
                                "track_id": tid,
                                "global_id": item.get("global_id", tid),
                                "camera_timestamp": st_time,
                                "video_pos_ms": st_time * 1000.0,
                                "bbox": None,
                                "class_label": cls_lbl,
                                "start_time": st_time,
                                "end_time": end_t,
                                "trajectory": traj,
                                "embedding": primary_vec,
                                "retrieval_embedding": retrieval_vec.tolist() if retrieval_vec is not None else None,
                                "appearance_embedding": app_vec.tolist() if app_vec is not None else None,
                                "metadata": meta_dict,
                            }
                            self.npz_records.append(rec)
            else:
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

                    cam_id = "cam_1"
                    tid = 0
                    parts = key.split("_")
                    for p in parts:
                        if p.isdigit():
                            tid = int(p)
                            break
                    meta_dict = {
                        "camera_id": cam_id,
                        "track_id": tid,
                        "camera_timestamp": 0.0,
                        "video_pos_ms": 0.0,
                        "class_label": "object",
                        "start_time": 0.0,
                        "end_time": 0.0,
                    }
                    self.npz_records.append({
                        "id": key,
                        "camera_id": cam_id,
                        "track_id": tid,
                        "camera_timestamp": 0.0,
                        "video_pos_ms": 0.0,
                        "bbox": None,
                        "class_label": "object",
                        "start_time": 0.0,
                        "end_time": 0.0,
                        "trajectory": None,
                        "embedding": vec,
                        "metadata": meta_dict,
                    })

    @staticmethod
    def _matches_where(record: dict, where: Optional[Dict[str, Any]]) -> bool:
        if not where:
            return True

        meta = record.get("metadata", {})

        def check_clause(k: str, v: Any) -> bool:
            if k == "$and" and isinstance(v, list):
                return all(VectorStore._matches_where(record, sub) for sub in v)

            val = record.get(k)
            if val is None:
                val = meta.get(k)

            if isinstance(v, dict):
                for op, op_val in v.items():
                    if op == "$gte" and (val is None or val < op_val):
                        return False
                    elif op == "$lte" and (val is None or val > op_val):
                        return False
                    elif op == "$gt" and (val is None or val <= op_val):
                        return False
                    elif op == "$lt" and (val is None or val >= op_val):
                        return False
                    elif op == "$eq" and val != op_val:
                        return False
                return True
            else:
                return val == v

        return all(check_clause(k, v) for k, v in where.items())

    def _encode_crop_if_needed(self, rec: dict, target_dim: int) -> np.ndarray | None:
        """Encode track crop image with encoder on-demand if dimension mismatch occurs."""
        meta = rec.get("metadata", {})
        video_name = meta.get("video_name", "")
        track_id = rec.get("track_id", 0)

        workspace_root = Path(__file__).resolve().parent.parent.parent
        crops_dir = workspace_root / "crops.noinclude"

        crop_path = None
        # Primary check: crops.noinclude/{video_name}_{track_id}/
        sub_dir = crops_dir / f"{video_name}_{track_id}"
        if sub_dir.exists() and sub_dir.is_dir():
            files = sorted(list(sub_dir.glob("*.jpg")) + list(sub_dir.glob("*.png")))
            if files:
                crop_path = files[0]

        if not crop_path:
            clip_stem = video_name.replace(".mp4", "")
            for ext in [".jpg", ".jpeg", ".png"]:
                p = crops_dir / f"{clip_stem}_track_{track_id}{ext}"
                if p.exists():
                    crop_path = p
                    break

        if crop_path and crop_path.exists():
            try:
                from PIL import Image
                if self.encoder is None:
                    from vlm_retrieval.retrieval.encoder import get_retrieval_encoder
                    self.encoder = get_retrieval_encoder(model_name="google/siglip2-base-patch16-224")
                with Image.open(crop_path) as img:
                    emb = self.encoder.encode_image(img)
                    return emb
            except Exception as err:
                self.logger.warning(f"Could not encode candidate crop {crop_path}: {err}")

        return None
