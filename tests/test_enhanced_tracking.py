import numpy as np
from typing import Any, List

from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYAH
from ultralytics.trackers.basetrack import TrackState
from reid.tracking.enhanced_bytetrack import EnhancedSTrack, EnhancedByteTracker
from reid.tracking.quality import TrackQuality
from reid.tracking.tracker import Tracker, Detections


class MockArgs:
    """Mock configuration arguments for EnhancedByteTracker."""

    def __init__(self) -> None:
        self.tracker_type = "enhanced_bytetrack"
        self.track_high_thresh = 0.25
        self.track_low_thresh = 0.1
        self.new_track_thresh = 0.25
        self.track_buffer = 30
        self.match_thresh = 0.2  # Lower match threshold to fail normal association for recalls
        self.fuse_score = True
        self.iou_weight = 0.7
        self.appearance_weight = 0.3
        self.quality_enabled = True
        self.quality_weights = {
            "detector_confidence": 0.3,
            "track_duration": 0.2,
            "embedding_stability": 0.3,
            "association_consistency": 0.2,
        }
        self.lifecycle_events_enabled = True





def test_recall_support() -> None:
    kf = KalmanFilterXYAH()
    feat1 = np.array([1.0, 0.0, 0.0])
    track = EnhancedSTrack(np.array([100, 100, 50, 50, 0]), 0.9, 0, feat1)
    track.activate(kf, frame_id=1)
    original_id = track.track_id

    # Make it lost
    track.mark_lost()
    assert track.state == TrackState.Lost

    # Create new detection track
    feat2 = np.array([0.95, 0.05, 0.0])
    new_det = EnhancedSTrack(np.array([105, 105, 50, 50, 0]), 0.95, 0, feat2)

    # Recall the track
    track.recall(new_det, frame_id=2)

    assert track.track_id == original_id
    assert track.state == TrackState.Tracked
    assert track.is_activated is True
    assert track.frame_id == 2
    assert track.score == 0.95
    assert track.recall_count == 1
    assert track.lost_recovered_count == 1
    # Check appearance smoothed
    assert track.curr_feat is not None
    assert abs(np.dot(track.smooth_feat, feat2) - 1.0) < 0.1


def test_track_quality() -> None:
    feat = np.array([1.0, 0.0])
    track = EnhancedSTrack(np.array([100, 100, 50, 50, 0]), 0.8, 0, feat)

    # Add detection updates to improve quality
    kf = KalmanFilterXYAH()
    track.activate(kf, frame_id=1)

    t2 = EnhancedSTrack(np.array([101, 101, 50, 50, 0]), 0.9, 0, feat)
    track.update(t2, frame_id=2)

    assert track.num_detections == 2
    assert abs(track.total_conf - 1.7) < 1e-6
    assert track.consecutive_associations == 2
    assert track.embedding_stability_sum == 2.0  # perfect stability (similarity 1.0 twice)

    score = TrackQuality.evaluate(track)
    assert 0.0 <= score <= 1.0
    assert track.quality_score == score


def test_event_dispatcher() -> None:
    args = MockArgs()
    tracker = EnhancedByteTracker(args)

    events: List[Any] = []

    def on_event(event_type: str, track_id: int, frame_id: int, timestamp: float, track: Any) -> None:
        events.append((event_type, track_id, timestamp))

    tracker.subscribe(on_event)

    # Setup frame detections mock
    results = Detections(
        xywh=np.array([[100.0, 100.0, 50.0, 50.0, 0.0]]),
        conf=np.array([0.9]),
        cls=np.array([0]),
    )
    feat1 = np.array([[1.0, 0.0]])

    # Frame 1: Create Track
    tracker.update(results, feats=feat1, timestamp=1.0)
    assert len(events) == 1
    assert events[0] == ("created", 1, 1.0)

    # Frame 2: Update Track
    results_updated = Detections(
        xywh=np.array([[102.0, 102.0, 50.0, 50.0, 0.0]]),
        conf=np.array([0.95]),
        cls=np.array([0]),
    )
    feat2 = np.array([[0.95, 0.05]])
    tracker.update(results_updated, feats=feat2, timestamp=2.0)
    assert len(events) == 2
    assert events[1] == ("updated", 1, 2.0)

    # Frame 3: Track Lost
    empty = Detections(
        xywh=np.empty((0, 4)),
        conf=np.empty((0,)),
        cls=np.empty((0,)),
    )
    tracker.update(empty, feats=np.empty((0, 2)), timestamp=3.0)
    # The track went lost!
    assert len(events) == 3
    assert events[2] == ("lost", 1, 3.0)




def test_tracker_wrapper_event_wiring() -> None:
    # Test that the Wrapper Tracker class correctly subscribes and populates internal buffers
    config_dict = {
        "tracker_type": "enhanced_bytetrack",
        "track_high_thresh": 0.25,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.25,
        "track_buffer": 30,
        "match_thresh": 0.2,  # Match threshold to fail normal association on recalls
        "fuse_score": True,
        "iou_weight": 0.7,
        "appearance_weight": 0.3,
        "quality_enabled": True,
        "quality_weights": {
            "detector_confidence": 0.3,
            "track_duration": 0.2,
            "embedding_stability": 0.3,
            "association_consistency": 0.2,
        },
        "lifecycle_events_enabled": True,
    }

    t_wrapper = Tracker(config_dict)

    terminated_ids: List[int] = []

    def on_terminated(track: Any) -> None:
        terminated_ids.append(track.track_id)

    t_wrapper.on_track_terminated = on_terminated

    boxes = np.array([[100.0, 100.0, 150.0, 150.0]])
    scores = np.array([0.9])
    classes = np.array([0])
    feats = np.ones((1, 128), dtype=np.float32)

    # Frame 1: Create
    t_wrapper.update(boxes, scores, classes, features=feats, frame_count=1, timestamp=1.0)
    assert 1 in t_wrapper.track_history
    assert 1 in t_wrapper.track_embeddings

    # Make the lost track buffer tiny so it gets removed/terminated quickly
    t_wrapper.tracker.max_frames_lost = 1

    # Frame 2: Lost
    t_wrapper.update(np.empty((0, 4)), np.empty((0,)), np.empty((0,)), frame_count=2, timestamp=2.0)

    # Frame 3: Stale -> Terminated
    t_wrapper.update(np.empty((0, 4)), np.empty((0,)), np.empty((0,)), frame_count=3, timestamp=3.0)

    # Track 1 should be terminated and termination hook should have fired
    assert 1 in terminated_ids
    assert 1 not in t_wrapper.track_history
    assert 1 not in t_wrapper.track_embeddings


def test_tracking_stage_unconfirmed_ignored() -> None:
    from reid.stages.tracking import TrackingStage
    from reid.postprocessing.pipeline import PostProcessingPipeline
    from reid.registry import SimpleRegistry

    # Mock pipeline object
    class MockPipeline:
        def __init__(self) -> None:
            self.registry = SimpleRegistry()
            self.stages = []

    pipeline = MockPipeline()

    class MockStage:
        def process(self, track: Any) -> Any:
            from tracking.compression.builder import CompressedTrackBuilder
            builder = CompressedTrackBuilder()
            builder.set_metadata(track_id=track.track_id, camera_id="cam1", class_label="car")
            builder.add_observation(0, 0.0, (10, 10, 20, 20))
            track.compressed_track = builder.build()
            return track

    stage = TrackingStage(
        tracker_config="bytetrack.yaml",
        postprocessing_pipeline=PostProcessingPipeline([MockStage()]),
    )

    # Initialize stage
    stage.initialize()
    pipeline.stages.append(stage)

    # Wire termination hook
    stage._wire_termination_hook(pipeline)

    # Mock track objects
    class MockTrack:
        def __init__(self, track_id: int, is_activated: bool) -> None:
            self.track_id = track_id
            self.is_activated = is_activated
            self.history = {"bboxes": [[10, 10, 20, 20]], "frames": [1], "timestamps": [1.0]}
            self.fused_embedding = None
            self.compressed_track = None

    # Test unconfirmed track
    unconfirmed = MockTrack(track_id=99, is_activated=False)
    stage.manual_tracker.on_track_terminated(unconfirmed)

    # Verify unconfirmed track is NOT in registry
    assert 99 not in pipeline.registry.identities

    # Test confirmed track
    confirmed = MockTrack(track_id=100, is_activated=True)
    # Put it in registry frame-by-frame first to mimic normal tracking
    pipeline.registry.update_track(local_track_id=100, appearance_embedding=np.zeros(2048, dtype=np.float32))

    stage.manual_tracker.on_track_terminated(confirmed)

    # Verify confirmed track IS in registry and has compressed track associated
    assert 100 in pipeline.registry.identities
    assert pipeline.registry.identities[100]["compressed_track"] is not None
