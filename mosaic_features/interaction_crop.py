"""
InteractionCropPipeline -- generate egocentric crop videos for interaction segments.

Consumes the output of PairInteractionFilter (which identifies interaction
segments) and generates per-individual egocentric crop videos for each segment.
This makes the pipeline tractable for long videos by only extracting the
frames that matter.

Output goes to ``media/interaction_crops/<group>__<sequence>/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, final

import cv2
import numpy as np
import pandas as pd
from pydantic import Field

from mosaic.core.pipeline._utils import Scope
from mosaic.core.pipeline.types import (
    COLUMNS as C,
)
from mosaic.core.pipeline.types import (
    DependencyLookup,
    Inputs,
    InputStream,
    Params,
    PoseConfig,
    Result,
    TrackInput,
)

from mosaic.behavior.feature_library.registry import register_feature
from mosaic.behavior.visualization_library.helpers import (
    compute_heading_angle,
    create_video_writer,
    infer_angle_degrees,
    safe_crop_with_padding,
)


@final
@register_feature
class InteractionCropPipeline:
    """
    Generate egocentric crop videos for detected interaction segments.

    Reads interaction segments from an upstream ``pair-interaction-filter``
    result and generates per-individual cropped videos for each segment.
    Optionally applies body masking, CLAHE, and grayscale conversion
    (delegating to the same algorithms as ``EgocentricCrop``).

    Inputs
    ------
    This feature takes *two* inputs:
      1. Tracks (standard trajectory data with pose keypoints)
      2. A ``pair-interaction-filter`` result providing interaction segments

    The pipeline iterates over the filter result's interaction segments
    (grouped by ``id_a``, ``id_b``, ``interaction_id``) and extracts
    egocentric crops from the source video for each individual in the pair.

    Output
    ------
    Videos are written to ``media/interaction_crops/<group>__<sequence>/``.
    Returns a metadata DataFrame with one row per generated clip.
    """

    name = "interaction-crop-pipeline"
    version = "0.1"
    parallelizable = True
    scope_dependent = False

    class Inputs(Inputs[TrackInput | Result]):
        pass

    class Params(Params):
        # Crop parameters (mirror EgocentricCrop)
        crop_size: tuple[int, int] = (192, 192)
        pose: PoseConfig = Field(default_factory=PoseConfig)
        center_mode: str | int = "default"
        center_offset_px: float = 0.0
        rotate_to_heading: bool = True
        heading_points: tuple[int, int] = (3, 6)
        margin_factor: float = 1.5
        angle_col: str | None = None
        interpolation: int = 1  # cv2.INTER_LINEAR
        background_color: int = 0
        # Post-processing
        body_mask: bool = False
        body_mask_length_px: int = 96
        body_mask_width_px: int = 64
        use_clahe: bool = False
        clahe_clip_limit: float = 2.0
        clahe_tile_grid_size: int = 25
        grayscale: bool = False
        # Which individuals in the pair to crop
        crop_both_individuals: bool = True
        # Output
        output_fps: float | None = None
        output_root: str | None = None

    def __init__(
        self,
        inputs: InteractionCropPipeline.Inputs = Inputs(
            ("tracks", Result(feature="pair-interaction-filter"))
        ),
        params: dict[str, object] | None = None,
    ):
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)
        self._ds = None
        self._scope: Scope = Scope()

    # --- Dataset hooks ---

    def bind_dataset(self, ds):
        self._ds = ds

    def set_scope(self, scope: Scope) -> None:
        self._scope = scope

    # --- State (stateless) ---

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        return True

    def fit(self, inputs: InputStream) -> None:
        pass

    def save_state(self, run_root: Path) -> None:
        pass

    # --- Apply ---

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process merged tracks + interaction-filter DataFrame.

        The pipeline merges both inputs on frame, so *df* contains
        track columns **and** filter columns (id_a, id_b,
        interaction_id, interaction_start, interaction_end).
        """
        if df.empty:
            return pd.DataFrame()

        from mosaic.media.video_io import MultiVideoReader

        p = self.params
        group = str(df[C.group_col].iloc[0]) if C.group_col in df.columns else ""
        sequence = str(df[C.seq_col].iloc[0]) if C.seq_col in df.columns else ""

        # Resolve video
        video_paths = self._ds.resolve_media_paths(group, sequence)

        # Group by interaction segment
        required = ["id_a", "id_b", "interaction_id", "interaction_start", "interaction_end"]
        for col in required:
            if col not in df.columns:
                raise ValueError(
                    f"Missing column '{col}' — ensure pair-interaction-filter "
                    f"output is provided as an input."
                )

        seg_groups = list(df.groupby(["id_a", "id_b", "interaction_id"]))

        # Sort segments by start frame for sequential video reading
        seg_groups.sort(key=lambda x: int(x[1]["interaction_start"].iloc[0]))

        # Open video reader once for the whole sequence
        reader = MultiVideoReader(video_paths)
        output_fps = p.output_fps or reader.fps

        clip_records = []
        try:
            for (id_a, id_b, seg_id), seg_df in seg_groups:
                start_frame = int(seg_df["interaction_start"].iloc[0])
                end_frame = int(seg_df["interaction_end"].iloc[0])

                # Determine which individuals to crop
                target_ids = [id_a]
                if p.crop_both_individuals:
                    target_ids.append(id_b)

                for target_id in target_ids:
                    record = self._crop_segment(
                        reader=reader,
                        output_fps=output_fps,
                        df_tracks=df,
                        target_id=target_id,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        id_a=id_a,
                        id_b=id_b,
                        seg_id=seg_id,
                        group=group,
                        sequence=sequence,
                    )
                    if record is not None:
                        clip_records.append(record)
        finally:
            reader.close()

        if not clip_records:
            return pd.DataFrame()
        return pd.DataFrame(clip_records)

    def _crop_segment(
        self,
        reader: Any,
        output_fps: float,
        df_tracks: pd.DataFrame,
        target_id: Any,
        start_frame: int,
        end_frame: int,
        id_a: Any,
        id_b: Any,
        seg_id: int,
        group: str,
        sequence: str,
    ) -> dict | None:
        """Crop a single segment for a single individual."""
        p = self.params

        # Filter tracks for this individual and frame range
        df_target = df_tracks[
            (df_tracks[C.id_col] == target_id)
            & (df_tracks[C.frame_col] >= start_frame)
            & (df_tracks[C.frame_col] <= end_frame)
        ].sort_values(C.frame_col)

        if df_target.empty:
            return None

        # Output path
        out_dir = self._get_output_dir(group, sequence)
        seg_dir = out_dir / f"bees_{id_a}_and_{id_b}_{start_frame}--{end_frame}"
        seg_dir.mkdir(parents=True, exist_ok=True)
        video_out = seg_dir / f"bee_{target_id}.mp4"

        crop_w, crop_h = p.crop_size

        # Angle inference
        angle_is_degrees = False
        if p.angle_col and p.angle_col in df_target.columns:
            angle_is_degrees = infer_angle_degrees(df_target[p.angle_col])

        # Pre-compute geometry for all frames (vectorized)
        angles, centers_x, centers_y = self._precompute_geometry(df_target, angle_is_degrees)
        frame_indices = df_target[C.frame_col].to_numpy(dtype=int)
        frame_set = set(frame_indices.tolist())
        frame_to_geom_idx = {int(f): i for i, f in enumerate(frame_indices)}

        if p.grayscale:
            codec = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(video_out), codec, float(output_fps), (crop_w, crop_h), isColor=False
            )
        else:
            writer = create_video_writer(video_out, output_fps, (crop_w, crop_h))

        n_written = 0
        try:
            reader.seek(start_frame)
            for frame_idx in range(start_frame, end_frame + 1):
                ret, frame = reader.read()
                if not ret:
                    break

                if frame_idx not in frame_set:
                    continue

                gi = frame_to_geom_idx[frame_idx]
                crop = self._extract_crop(
                    frame, (centers_x[gi], centers_y[gi]), angles[gi]
                )
                writer.write(crop)
                n_written += 1
        finally:
            writer.release()

        if n_written == 0:
            video_out.unlink(missing_ok=True)
            return None

        return {
            "group": group,
            "sequence": sequence,
            "id_a": id_a,
            "id_b": id_b,
            "target_id": target_id,
            "interaction_id": seg_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "n_frames": n_written,
            "video_path": str(video_out),
        }

    # --- Vectorized geometry ---

    def _precompute_geometry(
        self, df_target: pd.DataFrame, angle_is_degrees: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute angles and centers for all frames as numpy arrays."""
        p = self.params
        n = len(df_target)

        # --- Angles ---
        angles = np.zeros(n, dtype=np.float64)
        if p.angle_col and p.angle_col in df_target.columns:
            raw = df_target[p.angle_col].to_numpy(dtype=np.float64)
            valid = np.isfinite(raw)
            if angle_is_degrees:
                angles[valid] = np.radians(raw[valid])
            else:
                angles[valid] = raw[valid]
        else:
            neck_idx, tail_idx = p.heading_points
            nx = df_target[f"{p.pose.x_prefix}{neck_idx}"].to_numpy(dtype=np.float64)
            ny = df_target[f"{p.pose.y_prefix}{neck_idx}"].to_numpy(dtype=np.float64)
            tx = df_target[f"{p.pose.x_prefix}{tail_idx}"].to_numpy(dtype=np.float64)
            ty = df_target[f"{p.pose.y_prefix}{tail_idx}"].to_numpy(dtype=np.float64)
            valid = np.isfinite(nx) & np.isfinite(ny) & np.isfinite(tx) & np.isfinite(ty)
            angles[valid] = np.arctan2(ny[valid] - ty[valid], nx[valid] - tx[valid])

        # --- Centers ---
        mode = p.center_mode
        if mode == "default":
            xs = np.column_stack([
                df_target[f"{p.pose.x_prefix}{i}"].to_numpy(dtype=np.float64)
                for i in range(p.pose.pose_n)
            ])
            ys = np.column_stack([
                df_target[f"{p.pose.y_prefix}{i}"].to_numpy(dtype=np.float64)
                for i in range(p.pose.pose_n)
            ])
            cx = np.nanmean(xs, axis=1)
            cy = np.nanmean(ys, axis=1)
        elif mode == "pose0" or isinstance(mode, int):
            idx = 0 if mode == "pose0" else int(mode)
            cx = df_target[f"{p.pose.x_prefix}{idx}"].to_numpy(dtype=np.float64).copy()
            cy = df_target[f"{p.pose.y_prefix}{idx}"].to_numpy(dtype=np.float64).copy()
        else:
            raise ValueError(f"Unknown center_mode: {mode}")

        # Apply offset
        if p.center_offset_px != 0.0:
            finite_a = np.isfinite(angles)
            cx[finite_a] -= np.cos(angles[finite_a]) * p.center_offset_px
            cy[finite_a] -= np.sin(angles[finite_a]) * p.center_offset_px

        return angles, cx, cy

    # --- Per-frame crop (unchanged) ---

    def _extract_crop(
        self,
        frame: np.ndarray,
        center: tuple[float, float],
        angle: float,
    ) -> np.ndarray:
        p = self.params
        crop_w, crop_h = p.crop_size
        cx, cy = center
        n_channels = frame.shape[2] if frame.ndim == 3 else 1

        if not np.isfinite(cx) or not np.isfinite(cy):
            if p.grayscale or n_channels == 1:
                return np.full((crop_h, crop_w), p.background_color, dtype=np.uint8)
            return np.full((crop_h, crop_w, 3), p.background_color, dtype=np.uint8)

        if p.rotate_to_heading:
            angle_deg = np.degrees(angle)
            pre_crop_size = int(max(crop_w, crop_h) * p.margin_factor)

            pre_crop = safe_crop_with_padding(
                frame, (int(cx), int(cy)),
                (pre_crop_size, pre_crop_size),
                pad_value=p.background_color,
            )
            pre_center = (pre_crop_size / 2.0, pre_crop_size / 2.0)
            M = cv2.getRotationMatrix2D(pre_center, angle_deg, 1.0)
            bv = (p.background_color,) * (n_channels if n_channels > 1 else 1)
            rotated = cv2.warpAffine(
                pre_crop, M, (pre_crop_size, pre_crop_size),
                flags=p.interpolation,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=bv,
            )
            crop = safe_crop_with_padding(
                rotated, (pre_crop_size // 2, pre_crop_size // 2),
                (crop_w, crop_h),
                pad_value=p.background_color,
            )
        else:
            crop = safe_crop_with_padding(
                frame, (int(cx), int(cy)),
                (crop_w, crop_h),
                pad_value=p.background_color,
            )

        # Body mask
        if p.body_mask:
            mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
            cv2.ellipse(
                mask, (crop_w // 2, crop_h // 2),
                (p.body_mask_length_px // 2, p.body_mask_width_px // 2),
                0, 0, 360, 255, -1,
            )
            if crop.ndim == 3:
                crop = cv2.bitwise_and(crop, crop, mask=mask)
            else:
                crop = np.where(mask > 0, crop, 0).astype(crop.dtype)

        # Grayscale
        if p.grayscale and crop.ndim == 3:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # CLAHE
        if p.use_clahe:
            clahe = cv2.createCLAHE(
                clipLimit=p.clahe_clip_limit,
                tileGridSize=(p.clahe_tile_grid_size, p.clahe_tile_grid_size),
            )
            if crop.ndim == 2:
                crop = clahe.apply(crop)
            else:
                lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                crop = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return crop

    def _get_output_dir(self, group: str, sequence: str) -> Path:
        if self.params.output_root:
            return Path(self.params.output_root) / f"{group}__{sequence}"
        media_root = Path(self._ds.get_root("media"))
        return media_root / "interaction_crops" / f"{group}__{sequence}"
