import cv2
from PIL import Image
from typing import Optional, Tuple
from shared.utils import setup_logger


from pathlib import Path

class FrameExtractor:
    """Extracts frames from video files given a camera ID and video position."""

    def __init__(self, video_sources: Optional[dict] = None):
        """
        Args:
            video_sources: Mapping of camera_id -> absolute path to video file.
        """
        self.logger = setup_logger("FrameExtractor")
        self.video_sources = dict(video_sources) if video_sources else {}
        self._auto_discover_video_sources()
        self.logger.info(f"Configured video sources: {self.video_sources}")

    def _auto_discover_video_sources(self) -> None:
        """Auto-discover videos in workspace input_vids/ and map to cam_1, cam_2, etc."""
        workspace_root = Path(__file__).resolve().parent.parent
        search_dirs = [workspace_root / "input_vids", workspace_root / "dataset"]

        for sdir in search_dirs:
            if not sdir.exists():
                continue
            # Sort files so primary clip1.mp4 is prioritized over clip1_tracks.mp4
            video_files = sorted(list(sdir.glob("*.mp4")) + list(sdir.glob("*.avi")) + list(sdir.glob("*.mov")))
            for vfile in video_files:
                name = vfile.name
                stem = vfile.stem

                if name not in self.video_sources:
                    self.video_sources[name] = str(vfile.resolve())
                if stem not in self.video_sources:
                    self.video_sources[stem] = str(vfile.resolve())

                # Map clip1.mp4 -> cam_1, clip2.mp4 -> cam_2
                if stem.startswith("clip") and stem[4:].isdigit():
                    cam_key = f"cam_{stem[4:]}"
                    if cam_key not in self.video_sources:
                        self.video_sources[cam_key] = str(vfile.resolve())

    def extract_frame(
        self,
        camera_id: str,
        video_pos_ms: float,
        bbox: Optional[list] = None,
    ) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
        """Extract a frame from the video file at the given position.

        Args:
            camera_id: Camera identifier.
            video_pos_ms: Position in the video in milliseconds.
            bbox: Optional [x1, y1, x2, y2] bounding box for cropping.

        Returns:
            Tuple of (full_frame, cropped_region) as PIL Images. Either may be None.
        """
        path = self.video_sources.get(camera_id)
        if not path:
            if camera_id.startswith("cam_") and camera_id[4:].isdigit():
                num = camera_id[4:]
                workspace_root = Path(__file__).resolve().parent.parent
                cand_path = workspace_root / "input_vids" / f"clip{num}.mp4"
                if cand_path.exists():
                    path = str(cand_path.resolve())
                    self.video_sources[camera_id] = path

        if not path:
            self.logger.warning(f"No video source for camera {camera_id}")
            return None, None

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self.logger.error(f"Cannot open video: {path}")
            return None, None

        cap.set(cv2.CAP_PROP_POS_MSEC, video_pos_ms)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            self.logger.warning(f"Failed to read frame at {video_pos_ms:.0f}ms from {path}")
            return None, None

        # Convert BGR -> RGB for PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        full_frame = Image.fromarray(frame_rgb)

        crop = None
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                crop_arr = frame_rgb[y1:y2, x1:x2]
                crop = Image.fromarray(crop_arr)

        return full_frame, crop
