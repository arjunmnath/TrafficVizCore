import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from reid.stages.base import PipelineStage
from reid.utils import FrameData
from reid.postprocessing.pipeline import TerminatedTrack
from shared.utils import compute_cosine_similarity


@dataclass
class BufferedTrack:
    """Represents the complete state and details of a track in memory."""

    track_id: int
    class_label: str
    feed_name: str
    appearance_embeddings: List[np.ndarray] = field(default_factory=list)
    bboxes: List[List[float]] = field(default_factory=list)
    frames: List[int] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)

    # Postprocessed details (populated on termination)
    fused_embedding: Optional[np.ndarray] = None
    compressed_track: Optional[Dict[str, Any]] = None

    # Timestamps when the track was terminated (used for eviction)
    termination_timestamp: Optional[float] = None
    terminated_at_wall_clock: Optional[float] = None

    @property
    def appearance_embeddings_array(self) -> Optional[np.ndarray]:
        """Convert appearance embeddings list to a stacked numpy array (shape (N, D))."""
        if not self.appearance_embeddings:
            return None
        return np.array(self.appearance_embeddings, dtype=np.float32)


class ReIDBufferStage(PipelineStage):
    """In-memory buffer stage that caches ReID active and terminated track results.

    Active tracks are updated frame-by-frame. Terminated tracks are moved to a
    terminated tracks cache and held for `retention_seconds` to allow intra-camera
    matching/re-identification.
    """

    def __init__(self, retention_seconds: float = 60.0):
        """Constructor.

        Args:
            retention_seconds: How long to keep a terminated track in memory (in stream/elapsed seconds).
        """
        self.retention_seconds = retention_seconds
        self.active_tracks: Dict[int, BufferedTrack] = {}
        self.terminated_tracks: Dict[int, BufferedTrack] = {}
        self._last_stream_timestamp: float = 0.0

    def initialize(self, listener: Any = None) -> None:
        """Clear the buffer state."""
        self.active_tracks.clear()
        self.terminated_tracks.clear()
        self._last_stream_timestamp = 0.0

    def process(self, data: FrameData, pipeline: Any) -> FrameData:
        """Process active tracks on the current frame and perform eviction on expired tracks."""
        if data.skip or data.end_of_stream:
            return data

        self._last_stream_timestamp = data.timestamp

        if data.tracks is None or len(data.tracks) == 0:
            # Perform temporal eviction even on empty frames
            self._evict_expired_tracks(data.timestamp)
            return data

        # Resolve feed name
        feed_name = getattr(data, "feed_name", "")
        assert feed_name != "", f"Feed name not found for frame {data.frame_count}"

        feat_dim = (
            data.features.shape[1] if data.features is not None and len(data.features) > 0 else 2048
        )

        for t in data.tracks:
            # track layout: [x1, y1, x2, y2, track_id, score, class_id, detection_idx]
            bbox = t[0:4].tolist()
            track_id = int(t[4])
            class_id = int(t[6])
            det_idx = int(t[7])

            class_label = getattr(pipeline, "coco_classes", {}).get(class_id, "unknown")

            # Raw appearance embedding from FrameData.features for this specific detection
            if data.features is not None and det_idx < len(data.features):
                appearance_embedding = data.features[det_idx].copy()
            else:
                appearance_embedding = np.zeros(feat_dim, dtype=np.float32)

            if track_id not in self.active_tracks:
                self.active_tracks[track_id] = BufferedTrack(
                    track_id=track_id,
                    class_label=class_label,
                    feed_name=feed_name,
                )

            bt = self.active_tracks[track_id]
            bt.appearance_embeddings.append(appearance_embedding)
            bt.bboxes.append(bbox)
            bt.frames.append(data.frame_count)
            bt.timestamps.append(data.timestamp)

        # Evict expired terminated tracks
        self._evict_expired_tracks(data.timestamp)

        return data

    def handle_track_terminated(
        self, track_id: int, terminated_track: TerminatedTrack, timestamp: Optional[float] = None
    ) -> None:
        """Moves a terminated track from the active buffer to the terminated cache, enriching it."""
        term_time = timestamp if timestamp is not None else self._last_stream_timestamp
        wall_clock = time.time()

        if track_id in self.active_tracks:
            bt = self.active_tracks.pop(track_id)
        else:
            # If not already present in active buffer, build a new one from the terminated track details
            bt = BufferedTrack(
                track_id=track_id,
                class_label=terminated_track.class_label,
                feed_name=terminated_track.feed_name,
            )
            if terminated_track.appearance_embeddings is not None:
                bt.appearance_embeddings = [emb for emb in terminated_track.appearance_embeddings]
            if terminated_track.history is not None:
                h = terminated_track.history
                bt.bboxes = h.get("bboxes", [])
                bt.frames = h.get("frames", [])
                bt.timestamps = h.get("timestamps", [])

        # Enrich with terminated track details from postprocessing pipeline
        bt.fused_embedding = terminated_track.fused_embedding
        if terminated_track.compressed_track is not None:
            # Convert CompressedTrack structure to dictionary
            from tracking.serialization import JsonSerializer

            bt.compressed_track = JsonSerializer.serialize_to_dict(
                terminated_track.compressed_track
            )

        bt.termination_timestamp = term_time
        bt.terminated_at_wall_clock = wall_clock

        # Store in terminated cache
        self.terminated_tracks[track_id] = bt

    def _evict_expired_tracks(self, current_timestamp: float) -> None:
        """Evict terminated tracks that exceed the retention duration window."""
        now_wall = time.time()
        expired_ids = []

        for track_id, bt in self.terminated_tracks.items():
            if bt.termination_timestamp is not None and current_timestamp > 0.0:
                if (current_timestamp - bt.termination_timestamp) > self.retention_seconds:
                    expired_ids.append(track_id)
            elif bt.terminated_at_wall_clock is not None:
                if (now_wall - bt.terminated_at_wall_clock) > self.retention_seconds:
                    expired_ids.append(track_id)

        for track_id in expired_ids:
            self.terminated_tracks.pop(track_id, None)

    def get_track(self, track_id: int) -> Optional[BufferedTrack]:
        """Get the buffered track details (active or terminated) by track ID."""
        if track_id in self.active_tracks:
            return self.active_tracks[track_id]
        return self.terminated_tracks.get(track_id)

    def get_all_tracks(self) -> List[BufferedTrack]:
        """Get all active and terminated tracks currently stored in the buffer."""
        return list(self.active_tracks.values()) + list(self.terminated_tracks.values())

    def get_active_tracks(self) -> List[BufferedTrack]:
        """Get all active tracks currently stored in the buffer."""
        return list(self.active_tracks.values())

    def get_terminated_tracks(self) -> List[BufferedTrack]:
        """Get all terminated tracks currently stored in the buffer."""
        return list(self.terminated_tracks.values())

    def match_track(
        self, embedding: np.ndarray, class_label: str, threshold: float = 0.8
    ) -> List[Tuple[int, float]]:
        """Matches a query embedding against the buffered tracks of the same class label.

        Args:
            embedding: 1D query appearance vector.
            class_label: Class label of the target (e.g. 'person', 'car').
            threshold: Minimum cosine similarity required to consider a match.

        Returns:
            List of (track_id, similarity) sorted by similarity in descending order.
        """
        matches = []
        for track in self.get_all_tracks():
            if track.class_label != class_label:
                continue

            # Prioritize fused_embedding, and fall back to the average of appearances
            track_emb = None
            if track.fused_embedding is not None:
                track_emb = track.fused_embedding
            elif track.appearance_embeddings:
                track_emb = np.mean(track.appearance_embeddings, axis=0)

            if track_emb is not None:
                similarity = compute_cosine_similarity(embedding, track_emb)
                if similarity >= threshold:
                    matches.append((track.track_id, similarity))

        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
