"""
Convert Mosaic labels to FERAL label JSON format.

FERAL expects a JSON file with:
  {
    "is_multilabel": false,
    "splits": {
      "train": ["video1.mp4", ...],
      "val": ["video2.mp4", ...],
      "test": ["video3.mp4", ...],
      "inference": ["video4.mp4", ...]
    },
    "labels": {
      "video1.mp4": [0, 0, 1, 1, 0, ...],
      ...
    },
    "class_names": {
      "0": "no_behavior",
      "1": "trophallaxis"
    }
  }

Where ``labels`` values are per-frame integer class IDs and the video
filenames are relative to a common video directory (``prefix``).

This module provides utilities to:
1. Convert Mosaic's ``individual_pair_v1`` NPZ labels to FERAL JSON
2. Build the splits mapping from Mosaic's dataset structure
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def mosaic_labels_to_feral_json(
    label_dir: Path | str,
    video_dir: Path | str,
    output_path: Path | str,
    class_names: dict[str, str] | None = None,
    splits: dict[str, list[str]] | None = None,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    seed: int = 42,
    is_multilabel: bool = False,
) -> Path:
    """Convert Mosaic label NPZ files to FERAL label JSON.

    Parameters
    ----------
    label_dir : Path
        Directory containing Mosaic label NPZ files (e.g.
        ``labels/behavior/``).
    video_dir : Path
        Directory containing the interaction crop videos. Video filenames
        must match the label NPZ stems.
    output_path : Path
        Where to write the FERAL label JSON.
    class_names : dict, optional
        Mapping from class ID string to class name. Default:
        ``{"0": "no_behavior", "1": "trophallaxis"}``.
    splits : dict, optional
        Pre-defined splits ``{"train": [...], "val": [...], ...}``.
        If None, auto-splits by ``train_fraction`` / ``val_fraction``.
    train_fraction : float
        Fraction of videos for training (if auto-splitting).
    val_fraction : float
        Fraction for validation (rest goes to test).
    seed : int
        Random seed for auto-splitting.
    is_multilabel : bool
        Whether labels are multi-label (one-hot) or single-label (int).

    Returns
    -------
    Path
        Path to the written JSON file.
    """
    label_dir = Path(label_dir)
    video_dir = Path(video_dir)
    output_path = Path(output_path)

    if class_names is None:
        class_names = {"0": "no_behavior", "1": "trophallaxis"}

    # Collect all label files
    label_files = sorted(label_dir.glob("*.npz"))
    if not label_files:
        raise FileNotFoundError(f"No NPZ label files found in {label_dir}")

    # Build labels dict: video_filename -> per-frame class IDs
    labels = {}
    all_videos = []

    for npz_path in label_files:
        data = np.load(npz_path, allow_pickle=True)
        label_format = str(data.get("label_format", "dense"))

        if label_format == "individual_pair_v1":
            # Sparse event format -> dense per-frame
            frames = data["frames"]
            label_ids = data["labels"]
            # Need total frame count from video
            video_name = _find_matching_video(npz_path.stem, video_dir)
            if video_name is None:
                continue
            total_frames = _get_frame_count(video_dir / video_name)
            if total_frames is None:
                continue
            dense = np.zeros(total_frames, dtype=int)
            for f, l in zip(frames, label_ids):
                if 0 <= f < total_frames:
                    dense[f] = max(dense[f], int(l))
            labels[video_name] = dense.tolist()
            all_videos.append(video_name)
        else:
            # Dense format: direct array
            key = "labels" if "labels" in data else list(data.keys())[0]
            dense = data[key].astype(int)
            video_name = _find_matching_video(npz_path.stem, video_dir)
            if video_name is None:
                continue
            labels[video_name] = dense.tolist()
            all_videos.append(video_name)

    if not all_videos:
        raise ValueError("No matching video/label pairs found")

    # Build splits
    if splits is None:
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(all_videos))
        n_train = int(len(all_videos) * train_fraction)
        n_val = int(len(all_videos) * val_fraction)
        splits = {
            "train": [all_videos[i] for i in indices[:n_train]],
            "val": [all_videos[i] for i in indices[n_train : n_train + n_val]],
            "test": [all_videos[i] for i in indices[n_train + n_val :]],
            "inference": [],
        }

    feral_json = {
        "is_multilabel": is_multilabel,
        "splits": splits,
        "labels": labels,
        "class_names": class_names,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(feral_json, f, indent=2)

    return output_path


def _find_matching_video(stem: str, video_dir: Path) -> str | None:
    """Find a video file matching the NPZ stem."""
    for ext in (".mp4", ".avi", ".mkv"):
        candidate = video_dir / f"{stem}{ext}"
        if candidate.exists():
            return f"{stem}{ext}"
    # Try searching subdirectories
    for ext in (".mp4", ".avi", ".mkv"):
        matches = list(video_dir.rglob(f"{stem}{ext}"))
        if matches:
            return str(matches[0].relative_to(video_dir))
    return None


def _get_frame_count(video_path: Path) -> int | None:
    """Get frame count from a video file."""
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n if n > 0 else None
    except Exception:
        return None
