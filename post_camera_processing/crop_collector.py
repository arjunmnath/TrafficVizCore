"""Crop Collector module for Post-Camera Processing.

Gathers all bounding box crops belonging to Global Vehicle Identities across multiple
local tracks, camera channels, and disk storage locations / video feeds.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
from PIL import Image

from shared.utils import setup_logger


@dataclass
class RawCropItem:
    """Represents a single collected bounding box crop before quality filtering."""

    image: Image.Image
    global_id: str
    track_id: str
    camera_id: str
    frame_idx: int = 0
    timestamp_sec: float = 0.0
    bbox: Optional[List[float]] = None  # [x1, y1, x2, y2]
    source_path: Optional[str] = None
    class_label: str = "vehicle"
    confidence: float = 1.0


@dataclass
class GlobalIdentityCrops:
    """Container for all raw crops associated with a single Global Vehicle Identity."""

    global_id: str
    crops: List[RawCropItem] = field(default_factory=list)
    metadata: Dict[str, Union[str, int, float, List[str]]] = field(default_factory=dict)


class CropCollector:
    """Gathers crops for each Global Vehicle Identity from disk directories or registry JSON."""

    def __init__(self, logger: Optional[Any] = None) -> None:
        self.logger = logger or setup_logger("CropCollector")

    def collect_from_crop_directory(
        self,
        crop_dir: Union[str, Path],
        global_match_json: Optional[Union[str, Path]] = None,
    ) -> Dict[str, GlobalIdentityCrops]:
        """Gathers crops from a directory structure.

        Directory structure options:
        1. Subdirectories per track: `crop_dir/<track_folder>/frame_XXXX.jpg`
        2. Flat image files: `crop_dir/cam1_track12_frame001.jpg`

        If `global_match_json` is provided, maps local track IDs to global identity clusters.
        """
        crop_dir = Path(crop_dir)
        if not crop_dir.exists():
            self.logger.warning(f"Crop directory does not exist: {crop_dir}")
            return {}

        track_to_global: Dict[str, str] = {}
        if global_match_json and Path(global_match_json).exists():
            track_to_global = self._load_global_mappings(Path(global_match_json))

        identity_map: Dict[str, GlobalIdentityCrops] = {}

        # Option A: Directory contains subdirectories per track
        subdirs = [d for d in crop_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        if subdirs:
            self.logger.info(f"Found {len(subdirs)} track subdirectories in '{crop_dir}'")
            for sdir in subdirs:
                local_track_name = sdir.name
                global_id = track_to_global.get(local_track_name, local_track_name)

                if global_id not in identity_map:
                    identity_map[global_id] = GlobalIdentityCrops(global_id=global_id)

                image_files = sorted(
                    [
                        f
                        for f in sdir.iterdir()
                        if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
                    ]
                )
                for img_path in image_files:
                    try:
                        pil_img = Image.open(img_path).convert("RGB")
                    except Exception as err:
                        self.logger.warning(f"Failed to load crop '{img_path}': {err}")
                        continue

                    camera_id, frame_idx, timestamp_sec = self._parse_filename_metadata(
                        img_path.name, local_track_name
                    )

                    crop_item = RawCropItem(
                        image=pil_img,
                        global_id=global_id,
                        track_id=local_track_name,
                        camera_id=camera_id,
                        frame_idx=frame_idx,
                        timestamp_sec=timestamp_sec,
                        source_path=str(img_path),
                    )
                    identity_map[global_id].crops.append(crop_item)

        # Option B: Flat directory of crop images
        else:
            image_files = sorted(
                [
                    f
                    for f in crop_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
                ]
            )
            self.logger.info(f"Found {len(image_files)} crop images in flat directory '{crop_dir}'")
            for img_path in image_files:
                try:
                    pil_img = Image.open(img_path).convert("RGB")
                except Exception as err:
                    self.logger.warning(f"Failed to load crop '{img_path}': {err}")
                    continue

                track_id = self._extract_track_id_from_filename(img_path.name)
                global_id = track_to_global.get(track_id, track_id)
                camera_id, frame_idx, timestamp_sec = self._parse_filename_metadata(
                    img_path.name, track_id
                )

                if global_id not in identity_map:
                    identity_map[global_id] = GlobalIdentityCrops(global_id=global_id)

                crop_item = RawCropItem(
                    image=pil_img,
                    global_id=global_id,
                    track_id=track_id,
                    camera_id=camera_id,
                    frame_idx=frame_idx,
                    timestamp_sec=timestamp_sec,
                    source_path=str(img_path),
                )
                identity_map[global_id].crops.append(crop_item)

        total_crops = sum(len(gid.crops) for gid in identity_map.values())
        self.logger.info(
            f"Collected {total_crops} total raw crops across {len(identity_map)} Global Vehicle Identities."
        )
        return identity_map

    def _load_global_mappings(self, json_path: Path) -> Dict[str, str]:
        """Loads local track -> global identity mapping from cross-camera association JSON."""
        track_to_global: Dict[str, str] = {}
        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            # Option 1: List of pairwise match dicts [{"feed_a": "clip1.mp4", "track_a": 114, ...}]
            if isinstance(data, list) and data and isinstance(data[0], dict) and "feed_a" in data[0]:
                parent: Dict[str, str] = {}

                def find(i: str) -> str:
                    if parent.setdefault(i, i) == i:
                        return i
                    parent[i] = find(parent[i])
                    return parent[i]

                def union(i: str, j: str) -> None:
                    root_i, root_j = find(i), find(j)
                    if root_i != root_j:
                        parent[root_i] = root_j

                for match in data:
                    t_a = f"{match['feed_a']}_{match['track_a']}"
                    t_b = f"{match['feed_b']}_{match['track_b']}"
                    union(t_a, t_b)

                clusters: Dict[str, List[str]] = {}
                for t in parent:
                    root = find(t)
                    if root not in clusters:
                        clusters[root] = []
                    clusters[root].append(t)

                for idx, (root, members) in enumerate(clusters.items()):
                    gid = f"global_veh_{idx+1}"
                    for m in members:
                        track_to_global[m] = gid

            elif isinstance(data, list):
                for idx, cluster in enumerate(data):
                    gid = f"global_veh_{idx+1}"
                    if isinstance(cluster, list):
                        for track_name in cluster:
                            track_to_global[str(track_name)] = gid
                    elif isinstance(cluster, dict) and "tracks" in cluster:
                        gid = str(cluster.get("global_id", gid))
                        for track_name in cluster["tracks"]:
                            track_to_global[str(track_name)] = gid

            elif isinstance(data, dict):
                if "global_matches" in data:
                    track_to_global = {str(k): str(v) for k, v in data["global_matches"].items()}
                else:
                    for gid, tracks in data.items():
                        if isinstance(tracks, list):
                            for t in tracks:
                                track_to_global[str(t)] = str(gid)
                        else:
                            track_to_global[str(gid)] = str(tracks)

            self.logger.info(
                f"Loaded {len(track_to_global)} track-to-global identity mappings from {json_path.name}"
            )
        except Exception as err:
            self.logger.error(f"Error parsing global match JSON '{json_path}': {err}")

        return track_to_global

    def _parse_filename_metadata(
        self, filename: str, fallback_track_id: str
    ) -> tuple[str, int, float]:
        """Parses camera ID, frame index, and timestamp from crop filename conventions."""
        camera_id = "cam_1"
        frame_idx = 0
        timestamp_sec = 0.0

        cam_match = re.search(r"clip(\d+)", fallback_track_id + "_" + filename)
        if cam_match:
            camera_id = f"cam_{cam_match.group(1)}"

        frame_match = re.search(r"frame_(\d+)", filename)
        if frame_match:
            frame_idx = int(frame_match.group(1))

        t_match = re.search(r"_t([\d]+(?:\.[\d]+)?)", filename)
        if t_match:
            try:
                timestamp_sec = float(t_match.group(1))
            except ValueError:
                timestamp_sec = frame_idx / 30.0
        elif frame_idx > 0:
            timestamp_sec = frame_idx / 30.0

        return camera_id, frame_idx, timestamp_sec

    def _extract_track_id_from_filename(self, filename: str) -> str:
        """Extracts track ID prefix from flat filenames like 'clip1.mp4_12_frame_001.jpg'."""
        match = re.search(r"^(clip\d+\.mp4_\d+)", filename)
        if match:
            return match.group(1)
        stem = Path(filename).stem
        if "_frame_" in stem:
            return stem.split("_frame_")[0]
        return stem
