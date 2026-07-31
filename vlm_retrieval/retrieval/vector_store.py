"""NPZ vector store wrapper for SigLIP2 embedding search with metadata filtering."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from shared.utils import setup_logger


class VectorStore:
    """Local NPZ file vector store client for SigLIP2 embedding retrieval and track metadata queries."""

    def __init__(
        self,
        npz_dir: Optional[str] = None,
        npz_path: Optional[str] = None,
        json_path: Optional[str] = None,
        encoder: Optional[Any] = None,
    ) -> None:
        self.logger = setup_logger("VectorStore")
        self.encoder = encoder

        self.npz_dir = npz_dir or os.environ.get("NPZ_DIR")
        self.npz_path = npz_path or os.environ.get("NPZ_PATH")
        self.json_path = json_path or os.environ.get("JSON_PATH")

        self.conn_type = None
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

        # Workspace fallback to NPZ if available
        workspace_root = Path(__file__).resolve().parent.parent.parent
        default_npz = workspace_root / "artifacts" / "registry.retrieval.embeddings.npz"
        default_json = workspace_root / "artifacts" / "registry.tracks.identities.json"
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
        candidates: List[dict] = []

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
        """Query metadata records from the vector store without vector search."""
        results: List[dict] = []

        for rec in self.npz_records:
            if self._matches_where(rec, where):
                results.append(rec["metadata"])
                if len(results) >= limit:
                    break

        return results

    def get_event_count(self) -> int:
        """Return total number of records stored in the vector store."""
        return len(self.npz_records)

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
                                "track_details": item,
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
        crops_dir = workspace_root / "artifacts" / "crops"

        crop_path = None
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
                    self.encoder = get_retrieval_encoder(model_name="google/siglip2-so400m-patch14-384")
                with Image.open(crop_path) as img:
                    emb = self.encoder.encode_image(img)
                    return emb
            except Exception as err:
                self.logger.warning(f"Could not encode candidate crop {crop_path}: {err}")

        return None
