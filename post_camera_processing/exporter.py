"""Exporter module for Post-Camera Processing.

Exports semantic profiles and representative crop embeddings to compressed `.npz` and `.json` files
compatible with `VectorStore` and `scripts/run_standalone_inference.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from post_camera_processing.aggregator import SemanticProfile
from shared.utils import setup_logger


class EmbeddingExporter:
    """Exports semantic identity profiles to `.npz` embedding files and JSON registry files."""

    def __init__(self, logger: Optional[Any] = None) -> None:
        self.logger = logger or setup_logger("EmbeddingExporter")

    def export_profiles(
        self,
        profiles: List[SemanticProfile],
        output_npz_path: Union[str, Path],
        output_json_path: Optional[Union[str, Path]] = None,
        export_mode: str = "multi_view",  # "multi_view" (all rep views) or "mean" (one vector per ID)
    ) -> tuple[Path, Optional[Path]]:
        """Exports semantic profiles to `.npz` and matching `.json` registry."""
        npz_path = Path(output_npz_path)
        npz_path.parent.mkdir(parents=True, exist_ok=True)

        json_path: Optional[Path] = None
        if output_json_path:
            json_path = Path(output_json_path)
            json_path.parent.mkdir(parents=True, exist_ok=True)

        embeddings_list: List[np.ndarray] = []
        ids_list: List[str] = []
        metadatas_list: List[str] = []  # JSON string serialized metadata per entry
        registry_json_data: Dict[str, Any] = {}

        for prof in profiles:
            gid = prof.global_id
            registry_json_data[gid] = prof.metadata

            if export_mode == "mean" and prof.aggregated_embedding is not None:
                embeddings_list.append(prof.aggregated_embedding)
                ids_list.append(gid)
                meta_item = dict(prof.metadata)
                meta_item["vector_type"] = "mean_aggregated"
                metadatas_list.append(json.dumps(meta_item))

            else:
                # "multi_view" mode: export each representative view vector
                for view in prof.representative_views:
                    embeddings_list.append(view.embedding)
                    ids_list.append(view.doc_id)

                    view_meta = dict(prof.metadata)
                    view_meta.update(
                        {
                            "view_idx": view.view_idx,
                            "camera_id": view.camera_id,
                            "track_id": view.track_id,
                            "frame_idx": view.frame_idx,
                            "camera_timestamp": view.timestamp_sec,
                            "start_time": view.timestamp_sec,
                            "end_time": view.timestamp_sec,
                            "quality_score": view.quality_score,
                            "crop_path": view.source_path,
                            "vector_type": "representative_view",
                        }
                    )
                    metadatas_list.append(json.dumps(view_meta))

        if not embeddings_list:
            self.logger.warning("No embeddings to export.")
            return npz_path, json_path

        emb_matrix = np.array(embeddings_list, dtype=np.float32)
        ids_array = np.array(ids_list, dtype=object)
        metas_array = np.array(metadatas_list, dtype=object)

        # Save to NPZ
        np.savez_compressed(
            npz_path,
            retrieval_embeddings=emb_matrix,
            ids=ids_array,
            metadatas=metas_array,
        )
        self.logger.info(
            f"Successfully exported {len(emb_matrix)} semantic embeddings to '{npz_path}' (shape: {emb_matrix.shape})"
        )

        # Save to JSON registry
        if json_path:
            with open(json_path, "w") as f:
                json.dump(registry_json_data, f, indent=2)
            self.logger.info(f"Successfully exported registry metadata to '{json_path}'")

        return npz_path, json_path
