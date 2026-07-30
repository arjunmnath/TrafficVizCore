import time
from typing import Any, Optional
from reid.stages.base import PipelineStage
from reid.utils import ReIDPipelineListener, FrameData
from reid.tracking.tracker import Tracker


class TrackingStage(PipelineStage):
    """Performs manual track association. Owns the Tracker.

    When a track is terminated the ``on_track_terminated`` hook fires.
    If a ``postprocessing_pipeline`` is provided it is executed immediately
    on that track before any other downstream logic.
    """

    def __init__(
        self,
        tracker_config: str,
        postprocessing_pipeline: Optional[Any] = None,
    ):
        """Constructor.

        Args:
            tracker_config (str): Path to tracker configuration YAML.
            postprocessing_pipeline: Optional PostProcessingPipeline to run on
                each terminated track. If None, no postprocessing is performed.
        """
        self.tracker_config: str = tracker_config
        self.manual_tracker: Tracker | None = None
        self.postprocessing_pipeline: Optional[Any] = postprocessing_pipeline

    def initialize(self, listener: ReIDPipelineListener | None = None) -> None:
        if listener:
            listener.on_init_status("Loading manual Tracker and configuration...")
        self.manual_tracker = Tracker(self.tracker_config)

    def _wire_termination_hook(self, pipeline: Any) -> None:
        """Wire the on_track_terminated hook to run the postprocessing pipeline."""
        from reid.postprocessing.pipeline import TerminatedTrack

        postprocessing_pipeline = self.postprocessing_pipeline

        def _on_terminated(track: Any) -> None:
            # Skip unconfirmed/non-activated tracks to prevent registry pollution
            if not getattr(track, "is_activated", False):
                return

            # Resolve class label from track history if available
            class_label = "unknown"
            feed_name = ""

            # Pull appearance_embeddings from the registry for this track
            appearance_embeddings = None
            if hasattr(pipeline, "registry") and pipeline.registry is not None:
                entry = pipeline.registry.identities.get(track.track_id)
                if entry is not None:
                    app_list = entry.get("appearance_embeddings", [])
                    if app_list:
                        import numpy as np

                        appearance_embeddings = np.array(app_list, dtype=np.float32)
                    # Pull feed_name and class_label from registry attributes directly
                    feed_name = entry.get("feed_name", "")
                    class_label = entry.get("class_label", "unknown")

            # Fallback: pull from ReIDBufferStage if registry didn't have it
            if appearance_embeddings is None:
                from reid.stages.buffer import ReIDBufferStage

                buffer_stage = next(
                    (s for s in pipeline.stages if isinstance(s, ReIDBufferStage)), None
                )
                if buffer_stage is not None and track.track_id in buffer_stage.active_tracks:
                    bt = buffer_stage.active_tracks[track.track_id]
                    if bt.appearance_embeddings:
                        import numpy as np

                        appearance_embeddings = np.array(bt.appearance_embeddings, dtype=np.float32)
                    feed_name = bt.feed_name
                    class_label = bt.class_label

            terminated = TerminatedTrack(
                track_id=track.track_id,
                class_label=class_label,
                feed_name=feed_name,
                appearance_embeddings=appearance_embeddings,
                history=getattr(track, "history", None),
            )

            if postprocessing_pipeline is not None:
                terminated = postprocessing_pipeline.run(terminated)

            # Store the postprocessed track back so downstream stages can read it
            track.postprocessed = terminated

            if hasattr(pipeline, "registry") and pipeline.registry is not None:
                master_id = getattr(terminated, "master_track_id", None)
                target_id = master_id if master_id is not None else track.track_id
                if master_id is not None and master_id != track.track_id:
                    pipeline.registry.merge_tracks(master_id, track.track_id)

                    # Live log intra-camera trajectory fusion event
                    sim = terminated.extra.get("fusion_similarity", 0.0)
                    t_str = time.strftime("%H:%M:%S")
                    fusion_msg = (
                        f"[{t_str}] [bold green][INTRA-CAM FUSION][/bold green] "
                        f"Merged Track [bold yellow]#{track.track_id:03d}[/bold yellow] into "
                        f"Master Track [bold yellow]#{master_id:03d}[/bold yellow] ({terminated.class_label}) "
                        f"| Sim: [cyan]{sim:.3f}[/cyan] | Feed: [magenta]{terminated.feed_name}[/magenta]"
                    )

                    listener = getattr(pipeline, "listener", None)
                    if listener is not None:
                        if hasattr(listener, "recent_logs"):
                            listener.recent_logs.append(fusion_msg)
                        if hasattr(listener, "on_frame_processed"):
                            listener.on_frame_processed(
                                video_name=terminated.feed_name,
                                video_idx=getattr(listener, "current_video_idx", 1),
                                total_videos=len(getattr(listener, "video_paths", [terminated.feed_name])),
                                frame_count=getattr(track, "end_frame", 0),
                                total_frames=0,
                                elapsed_time=0.0,
                                fps=0.0,
                                registry=pipeline.registry,
                                log_message=fusion_msg,
                            )

                compressed_track = getattr(terminated, "compressed_track", None)
                if compressed_track is not None:
                    from tracking.serialization import JsonSerializer

                    serialized_dict = JsonSerializer.serialize_to_dict(compressed_track)
                    pipeline.registry.add_compressed_track(target_id, serialized_dict)

            # Notify ReIDBufferStage of track termination
            from reid.stages.buffer import ReIDBufferStage

            buffer_stage = next(
                (s for s in pipeline.stages if isinstance(s, ReIDBufferStage)), None
            )
            if buffer_stage is not None:
                term_timestamp = None
                if track.history and track.history.get("timestamps"):
                    term_timestamp = track.history["timestamps"][-1]
                buffer_stage.handle_track_terminated(
                    track_id=track.track_id,
                    terminated_track=terminated,
                    timestamp=term_timestamp,
                )

        self.manual_tracker.on_track_terminated = _on_terminated

    def process(self, data: FrameData, pipeline: Any) -> FrameData:
        assert self.manual_tracker is not None, "tracker not initialized."

        # Wire hook on first call (lazy, after pipeline is fully set up)
        if not self.manual_tracker.hook_wired:
            self._wire_termination_hook(pipeline)
            self.manual_tracker.set_hook_wired(True)

        # Calculate dynamic processing speed (frames per second) instead of static video frame rate
        processing_fps = data.frame_count / data.elapsed_time if data.elapsed_time > 0.0 else 0.0

        if data.skip or data.end_of_stream:
            listener = data.listener
            if listener and not data.end_of_stream:
                listener.on_frame_processed(
                    video_name=data.feed_name,
                    video_idx=data.feed_idx,
                    total_videos=data.total_videos,
                    frame_count=data.frame_count,
                    total_frames=data.total_frames,
                    elapsed_time=data.elapsed_time,
                    fps=processing_fps,
                    registry=pipeline.registry,
                )
            return data

        assert data.boxes is not None and data.scores is not None and data.classes is not None, (
            "boxes, scores, and classes must not be None"
        )
        # Run manual tracker update
        tracks = self.manual_tracker.update(
            boxes=data.boxes,
            scores=data.scores,
            classes=data.classes,
            features=data.features,
            frame_count=data.frame_count,
            timestamp=data.timestamp,
        )
        data.tracks = tracks

        if hasattr(pipeline, "recorded_predictions") and pipeline.recorded_predictions is not None:
            for t in tracks:
                bbox = t[0:4].tolist()  # [x1, y1, x2, y2]
                track_id = int(t[4])
                pipeline.recorded_predictions.append(
                    {
                        "feed": data.feed_name,
                        "frame": data.frame_count,
                        "track_id": track_id,
                        "bbox": bbox,
                    }
                )

        listener = data.listener

        # Progress update with active track count
        if listener:
            active_ids = [int(t[4]) for t in tracks] if len(tracks) > 0 else []
            log_line = None
            if len(active_ids) > 0:
                t_str = time.strftime("%H:%M:%S")
                log_line = f"[{t_str}] Active tracks: {active_ids}"

            listener.on_frame_processed(
                video_name=data.feed_name,
                video_idx=data.feed_idx,
                total_videos=data.total_videos,
                frame_count=data.frame_count,
                total_frames=data.total_frames,
                elapsed_time=data.elapsed_time,
                fps=processing_fps,
                registry=pipeline.registry,
                log_message=log_line,
            )

        return data

    def finalize(self, pipeline: Any) -> None:
        if self.manual_tracker:
            self.manual_tracker.terminate_all_tracks()
            self.manual_tracker.reset()
