from pathlib import Path
from typing import NamedTuple
import xarray as xr
import numpy as np
import copy

from vame.logging.logger import VameLogger
from vame.io.load_poses import read_pose_estimation_file
from vame.util.auxiliary import update_config

logger_config = VameLogger(__name__)
logger = logger_config.logger


class Feature(NamedTuple):
    name: str
    series: np.ndarray


def add_derived_features(
    config: dict,
    feature_data: dict[str, list[Feature]],     # video_name --> [(feature_name, feature_data)]
    n_features_without_deriv: int,
    save_logs: bool = True,
) -> None:
    if save_logs:
        log_path = Path(config["project_path"]) / "logs" / "preprocessing.log"
        logger_config.add_file_handler(str(log_path))

    project_path = config["project_path"]
    sessions = config["session_names"]
    n_features = config["num_features"]

    if len(feature_data) != len(sessions):
        raise ValueError(
            f"The length of feature_data ({len(feature_data)}) does not match the number of videos ({len(sessions)}). " \
            "The same derived features have to be passed for all videos."
        )
    # check that all sessions have the same number of features
    feature_lengths = {len(v) for v in feature_data.values()}
    if len(feature_lengths) > 1:
        raise ValueError(
            "The number of features is not the same for all videos in feature_data. " \
            "The same derived features have to be passed for all videos."
        )
    n_deriv_features = feature_lengths.pop()

    logger.info(f"Adding {n_deriv_features} derived feature(s).")

    for i, session in enumerate(sessions):
        logger.info(f"Session: {session}")
        # Read session data
        file_path = str(Path(project_path) / "data" / "processed" / f"{session}_processed.nc")
        _, _, ds = read_pose_estimation_file(file_path=file_path)

        deriv_features: list[Feature] = feature_data[session]
        for feature, series in deriv_features:
            n_frames = ds.dims["time"]
            if len(series) != n_frames:
                raise ValueError(f"The feature {feature} does not match the time dimension of the position data.")
            ds[feature] = (("time"), series)

        ds.to_netcdf(
            path=file_path,
            engine="netcdf4",
        )
    config_update = copy.deepcopy(config)
    if n_features != n_features_without_deriv:
        logger.info("Derived features were already added previously. Overwriting features..")
    config_update["num_features"] = n_features_without_deriv + n_deriv_features
    update_config(config=config, config_update=config_update)
