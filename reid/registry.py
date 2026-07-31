import numpy as np
from typing import Dict, List, Optional, Any


class SimpleRegistry:
    """Identity registry that maps local track IDs to appearance vectors.

    No embedding matching is performed. Each local_track_id is used directly as the identity key.
    The registry stores per track:
      - appearance_embeddings: per-frame raw detection embeddings from FrameData.features
      - compressed_track: serialised CompressedTrack dict (set via add_compressed_track)
    """

    def __init__(self) -> None:
        # local_track_id -> {
        #   "appearance_embeddings": list[ndarray], # raw detection feature per frame
        #   "compressed_track": dict | None         # serialised CompressedTrack
        # }
        self.identities: Dict[int, Dict[str, Any]] = {}
        # secondary_track_id -> master_track_id
        self.merged_track_map: Dict[int, int] = {}

    def get_master_track_id(self, local_track_id: int) -> int:
        """Resolve a local track ID to its root master track ID if it was merged."""
        curr = local_track_id
        visited = set()
        while curr in self.merged_track_map and curr not in visited:
            visited.add(curr)
            curr = self.merged_track_map[curr]
        return curr

    def update_track(
        self,
        local_track_id: int,
        appearance_embedding: np.ndarray[Any, Any],
        class_label: str = "unknown",
        feed_name: str = "",
        frame_number: int = 0,
        timestamp: float = 0.0,
        bbox: Optional[List[float]] = None,
    ) -> int:
        """Register or update a track with a new frame observation.

        Args:
            local_track_id: The tracker-assigned track ID, used directly as identity key.
            appearance_embedding: The raw detection embedding from FrameData.features for this frame.
            class_label: Detected class name.
            feed_name: Source video feed identifier.
            frame_number: Current frame index.
            timestamp: Current timestamp in seconds.
            bbox: Bounding box [x1, y1, x2, y2].

        Returns:
            The local_track_id (identity key).
        """
        if local_track_id not in self.identities:
            self.identities[local_track_id] = {
                "appearance_embeddings": [],
                "class_label": class_label,
                "feed_name": feed_name,
                "compressed_track": None,
            }

        entry = self.identities[local_track_id]
        entry["appearance_embeddings"].append(appearance_embedding)
        entry["class_label"] = class_label
        entry["feed_name"] = feed_name

        return local_track_id

    def add_compressed_track(
        self, local_track_id: int, compressed_track_dict: Dict[str, Any]
    ) -> None:
        """Associate a serialized compressed track representation with the identity."""
        target_id = self.get_master_track_id(local_track_id)
        if target_id not in self.identities:
            self.identities[target_id] = {
                "appearance_embeddings": [],
                "class_label": "unknown",
                "feed_name": "",
                "compressed_track": None,
            }
        self.identities[target_id]["compressed_track"] = compressed_track_dict

    def merge_tracks(self, primary_track_id: int, secondary_track_id: int) -> None:
        """Consolidate secondary_track_id into primary_track_id.

        Merges appearance_embeddings and updates compressed_track, removing secondary_track_id from registry.
        """
        if primary_track_id == secondary_track_id:
            return

        # Resolve primary_track_id to root master track ID if primary was previously merged
        root_primary = self.get_master_track_id(primary_track_id)

        # Record secondary mapping
        self.merged_track_map[secondary_track_id] = root_primary

        # Update any previous tracks that pointed to secondary_track_id to point to root_primary
        for s_id, p_id in list(self.merged_track_map.items()):
            if p_id == secondary_track_id:
                self.merged_track_map[s_id] = root_primary

        if secondary_track_id not in self.identities:
            return

        if root_primary not in self.identities:
            self.identities[root_primary] = self.identities.pop(secondary_track_id)
            return

        primary = self.identities[root_primary]
        secondary = self.identities.pop(secondary_track_id)

        primary["appearance_embeddings"].extend(secondary.get("appearance_embeddings", []))
        if secondary.get("compressed_track") is not None:
            primary["compressed_track"] = secondary["compressed_track"]

    def get_results_summary(self) -> List[Dict[str, Any]]:
        """Return a JSON-serialisable summary of all track identities and their compressed track details."""
        return [
            {
                "track_id": track_id,
                "compressed_track": data.get("compressed_track"),
            }
            for track_id, data in self.identities.items()
        ]

    def get_embeddings_dict(self) -> Dict[str, np.ndarray[Any, Any]]:
        """Return per-track stacked embeddings suitable for np.savez.

        Keys follow the pattern:
          - ``app_{track_id}``    — stacked raw appearance embeddings, shape (N, D)

        Returns:
            Flat dict of str -> ndarray ready for ``np.savez(**result)``.
        """
        result: Dict[str, np.ndarray[Any, Any]] = {}
        for track_id, data in self.identities.items():
            result[f"app_{track_id}"] = np.array(data["appearance_embeddings"], dtype=np.float32)
        return result
