import os
import subprocess
import tempfile
import shutil
import imageio
import cv2
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
from enum import Enum
from bb_binary.parsing import parse_video_fname


def extract_frames_from_video(video_path, target_directory, n_frames, start_frame=0,
                              codec="hevc_cuvid", command="ffmpeg", scale=1.0, output_format="bmp"):
    """Extract frames from a video file using FFmpeg."""
    
    if codec is not None:
        codec = ['-c:v', codec]
    else:
        codec = []

    if scale != 1.0:
        scale = f",scale=iw*{scale}:ih*{scale}"
    else:
        scale = ""


    call_args = codec + [
        "-y", "-v", "24", "-hwaccel", "cuda",
        "-i", video_path, "-start_number", "0",
        "-vf", f"select='gte(n\\,{start_frame})'{scale},setpts=N/FRAME_RATE/TB",
        "-vframes", str(n_frames),
        f"{target_directory}/%04d.{output_format}"
    ]

    path_to_command = "/usr/local/bin/"
    command = path_to_command + command
    p = subprocess.Popen([command] + call_args, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    if p.wait() != 0:
        raise ValueError(f"FFmpeg Error: {stderr.decode()}")

##################################################################################################################
# Custom video manager is used for caching and extracting frames from videos as well as saving frames to videos. 
# It is partly adapted from a jupyter notebook by Jacob Davidson
##################################################################################################################
class CustomVideoManager:
    def __init__(self, video_root, cache_path, video_output_path, max_workers=8):
        if not video_root.endswith("/"):
            video_root += "/"
        self.video_root = video_root
        if not cache_path.endswith("/"):
            cache_path += "/"
        self.cache_path = cache_path
        if not video_output_path.endswith("/"):
            video_output_path += "/"
        self.video_output_path = video_output_path

        self.last_requests = []
        self.video_files = []

        self.command = "ffmpeg"
        self.codec = "hevc"
        self.output_format = "png"
        self.scale = 1.0
        self.logger = logging.getLogger(__name__)

        self.loader_thread_pool = PoolExecutor(max_workers=max_workers)

        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        if not os.path.exists(video_output_path):
            os.makedirs(video_output_path)

    def clear_video_cache(self, retain_last_n_requests=None):
        """Removes all cached image files."""
        retained_images = set()
        if retain_last_n_requests:
            retain_last_n_requests = min(retain_last_n_requests, len(self.last_requests))
            for i in range(retain_last_n_requests):
                retained_images |= {str(f_id) + "." + self.output_format for f_id in self.last_requests[-(i + 1)]}
            del self.last_requests[0:(len(self.last_requests) - retain_last_n_requests)]

        # Remove old images from the cache
        for filename in os.listdir(self.cache_path):
            if filename.endswith("." + self.output_format) and filename not in retained_images:
                os.remove(os.path.join(self.cache_path, filename))


    def extract_frames(self, video_name, start_frame, n_frames, frame_ids):
        """Extracts frames from a video using FFmpeg and caches them."""
        assert n_frames == len(frame_ids), f"Number of frames ({n_frames}) and length of frame IDs ({len(frame_ids)}) must match."

        with tempfile.TemporaryDirectory(dir=self.cache_path, prefix="vidtmp_") as dirpath:
            try:
                extract_frames_from_video(
                    video_name,
                    target_directory=dirpath,
                    start_frame=start_frame,
                    n_frames=n_frames,
                    command=self.command,
                    codec=self.codec,
                    output_format=self.output_format,
                    scale=self.scale
                )
            except Exception as e:
                self.logger.error(f"Error extracting frames from video {video_name}: {e}")
                raise e

            for (frame_id, filepath) in zip(frame_ids, sorted(os.listdir(dirpath))):
                full_filepath = os.path.join(dirpath, filepath)
                if os.path.getsize(full_filepath) > 0:
                    shutil.move(full_filepath, os.path.join(self.cache_path, f"{frame_id}.{self.output_format}"))
                else:
                    self.logger.warning("Zero-size file created by FFmpeg.")

    def get_frame_id_path(self, frame_id):
        """Returns the path to the cached image for a given frame ID."""
        return os.path.join(self.cache_path, f"{frame_id}.{self.output_format}")

    def is_frame_cached(self, frame_id):
        """Checks if a frame is already cached."""
        return os.path.isfile(self.get_frame_id_path(frame_id))

    def get_frame(self, frame_id):
        """Loads a cached frame image as a NumPy array."""
        if frame_id is None:
            return None
        return imageio.v3.imread(self.get_frame_id_path(frame_id), plugin="opencv", colorspace="GRAY")
        #return skimage.io.imread(self.get_frame_id_path(frame_id), as_gray=True, plugin="matplotlib")

    def cache_frames(self, frame_ids, video_name, frame_indices):
        """Ensures all specified frames in the dataframe are cached."""
        frame_ids_to_cache = [f_id for f_id in frame_ids if not self.is_frame_cached(f_id)]

        if not frame_ids_to_cache:
            self.logger.info(f"All frames for video {video_name} are already cached.")
            return
    
        frames_df = pd.DataFrame({
            'frame_id': frame_ids,
            'frame_index': frame_indices
        })
        frames_df = frames_df[frames_df['frame_id'].isin(frame_ids_to_cache)]
    
        # Extract frames video
        frame_indices = frames_df['frame_index'].tolist()
        frame_ids = frames_df['frame_id'].tolist()

        frame_ids = np.arange(min(frame_ids), max(frame_ids)+1)

        start_idx = min(frame_indices)
        end_idx = max(frame_indices)
        n_frames = end_idx - start_idx + 1

        self.logger.info(f"Caching {n_frames} frames from {video_name} ...")
        self.logger.debug(f"\tFrame-Index range: {start_idx} - {end_idx}, Frame-ID range: {min(frame_ids)} - {max(frame_ids)}")

        self.extract_frames(
            video_name,
            start_frame=start_idx,
            n_frames=n_frames,
            frame_ids=frame_ids
        )
        self.logger.info(f"Caching complete.")


    def get_frames(self, frame_ids, video_name, frame_indices):
        """Returns a list of frames as NumPy arrays for the given frame IDs in the dataframe."""
        self.cache_frames(frame_ids, video_name, frame_indices)
        self.last_requests.append(frame_ids)
    
        # Load frames
        images = [self.get_frame(f_id) for f_id in frame_ids]
        return images
    
    
    def write_to_video(self, images, filename, frame_rate):
        self.logger.info("Writing to video " + self.video_output_path + filename + " ...")
        if len(images) == 0:
            return

        out = cv2.VideoWriter(self.video_output_path + filename, cv2.VideoWriter_fourcc(*'avc1'), frame_rate, images[0].shape, isColor=False)
        for image in images:
            out.write(image)
        out.release()

        self.logger.info("Writing to video complete.")
    
    def get_all_video_files(self):
        video_files = []
        if os.path.exists(self.video_root):
            video_files.extend([
                os.path.join(self.video_root, f) for f in os.listdir(self.video_root) if f.endswith('.mp4')
            ])

        video_info_list = []
        for video_file in video_files:
            video_info_list.append({
                'video_filename': video_file
            })

        videos_df = pd.DataFrame(video_info_list)
        return videos_df
    
    def get_video_dimensions(self, video_name):
        width, height = imageio.v3.immeta(video_name)["size"]
        return (width, height)
    
    def display_frame_konstanz_data(self, frame_id, positions, orientations, colors=['red', 'green', 'blue', 'orange', 'purple']):
        frame = self.get_frame(frame_id)
        fig, ax = plt.subplots(figsize=(25, 10))
        ax.imshow(X=frame, cmap='gray')

        for idx, position in enumerate(positions):
            plt.scatter(position[0], position[1], s=20, c=colors[idx], marker='o',alpha=0.3)
        for idx, orient in enumerate(orientations):
            x, y = positions[idx]
            dx = 10 * np.cos(orient)
            dy = 10 * np.sin(orient)
            plt.arrow(x, y, dx, dy, color='yellow', head_width=5, head_length=5)
        plt.show()

    def get_all_video_files_for_cam(self, cam_id, df_cam):
        from datetime import timedelta

        # Get the range of dates to consider
        min_date = df_cam['timestamp'].dt.date.min()
        max_date = df_cam['timestamp'].dt.date.max()
        date_range = pd.date_range(start=min_date - pd.Timedelta(days=1),
                                end=max_date + pd.Timedelta(days=1))
        date_strs = date_range.strftime('%Y%m%d')

        video_files = []
        for date_str in date_strs:
            video_dir = os.path.join(self.video_root, f'cam-{cam_id}', date_str)
            if os.path.exists(video_dir):
                video_files.extend([
                    os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.mp4')
                ])

        # Parse video filenames to get start and end times
        video_info_list = []
        for video_file in video_files:
            try:
                _, start_time, end_time = parse_video_fname(os.path.basename(video_file))
                video_info_list.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'video_filename': video_file
                })
            except Exception as e:
                print(f"Failed to parse video filename {video_file}: {e}")

        videos_df = pd.DataFrame(video_info_list)
        videos_df.sort_values('start_time', inplace=True)
        return videos_df


    def delete_files_in_dir(self, cache_path):
        if not os.path.exists(cache_path):
            return
        # Clear all files in the directory
        for filename in os.listdir(cache_path):
            file_path = os.path.join(cache_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove file or symbolic link
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)  # Remove empty directory
            except Exception as e:
                self.logger.error(f"Failed to delete {file_path}: {e}")

