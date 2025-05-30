from pathlib import Path
from tqdm import tqdm
import urllib.request
import tarfile
from src.utils.pylogger import get_pylogger
import os
import shutil
import numpy as np # Import numpy

log = get_pylogger(__name__)

ROOT = Path(__file__).parent.parent.parent
ID2NAME = {i: str(i) for i in range(10)}

DATA_PATH = ROOT / "datasets"
MODELS_PATH = ROOT / "models"


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        return self.update(b * bsize - self.n)  # also sets self.n = b * bsize


def download_file(url, filepath):
    log.info(f"Downloading {url} to {filepath}.")
    with TqdmUpTo(
        unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=url.split("/")[-1]
    ) as t:  # all optional kwargs
        urllib.request.urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n
    log.info("Download finished.")


def unzip_tar_gz(file_path, dst_path, remove=False):
    log.info(f"Unzipping {file_path} to {dst_path}.")
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(dst_path)
    log.info(f"Unzipping finished.")
    if remove:
        os.remove(file_path)
        log.info(f"Removed {file_path}.")


def save_txt_to_file(txt, filename):
    with open(filename, "w") as file:
        file.write(txt)


def read_text_file(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]  # Optional: Remove leading/trailing whitespace

    return lines


def add_prefix_to_files(directory, prefix, ext=".png"):
    log.info(f"Adding {prefix} prefix to all {ext} files in {directory} directory.")

    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)) and filename.endswith(ext):
            new_filename = prefix + filename
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
    log.info("Prefix addition finished.")


def move_directory(source_dir, destination_dir):
    shutil.move(source_dir, destination_dir)
    log.info(f"Moved {source_dir} to {destination_dir}.")


def copy_directory(source_dir, destination_dir):
    shutil.copytree(source_dir, destination_dir)
    log.info(f"Copied {source_dir} to {destination_dir}.")


def remove_directory(dir_path):
    shutil.rmtree(dir_path)
    log.info(f"Removed {dir_path} directory")


def copy_all_files(source_dir, destination_dir, ext=".png"):
    filenames = os.listdir(source_dir)
    for filename in tqdm(filenames, desc="Copying files"):
        if filename.lower().endswith(ext):
            source = source_dir / filename
            destination = destination_dir / filename
            shutil.copy2(source, destination)

    log.info(f"Copied all {ext} files ({len(filenames)}) from {source_dir} to {destination_dir}.")


def copy_files(source_filepaths, dest_filepaths):
    for source_filepath, dest_filepath in tqdm(
        zip(source_filepaths, dest_filepaths), desc="Copying files"
    ):
        shutil.copy2(source_filepath, dest_filepath)
    log.info(f"Copied files ({len(source_filepaths)}).")


def group_and_merge_boxes(
    boxes_xywhn: np.ndarray, class_ids: np.ndarray, class_scores: np.ndarray, distance_threshold: float = 0.05
):
    """
    Groups and merges bounding boxes of individual digits into multi-digit bounding boxes.

    Args:
        boxes_xywhn: Normalized bounding boxes in (x_center, y_center, width, height) format.
                     Shape: (num_boxes, 4)
        class_ids: Class IDs of the detected digits. Shape: (num_boxes,)
        class_scores: Confidence scores of the detected digits. Shape: (num_boxes,)
        distance_threshold: Maximum horizontal distance (normalized) between two digits
                            to be considered part of the same multi-digit number.

    Returns:
        Tuple of (merged_boxes_xywhn, merged_class_ids, merged_class_scores)
    """
    if len(boxes_xywhn) == 0:
        return boxes_xywhn, class_ids, class_scores

    # Convert to x1y1x2y2 for easier merging
    x_center, y_center, width, height = boxes_xywhn.T
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    boxes_x1y1x2y2 = np.stack([x1, y1, x2, y2], axis=1)

    # Sort by x1 to process digits from left to right
    sorted_indices = np.argsort(boxes_x1y1x2y2[:, 0])
    sorted_boxes = boxes_x1y1x2y2[sorted_indices]
    sorted_class_ids = class_ids[sorted_indices]
    sorted_class_scores = class_scores[sorted_indices]

    merged_boxes = []
    merged_ids = []
    merged_scores = []

    current_group = []
    for i in range(len(sorted_boxes)):
        if not current_group:
            current_group.append(i)
        else:
            # Check horizontal distance between current box and the last box in the group
            last_box_idx = current_group[-1]
            # Distance is calculated from the right edge of the last box to the left edge of the current box
            dist = sorted_boxes[i, 0] - sorted_boxes[last_box_idx, 2]

            if dist < distance_threshold:
                current_group.append(i)
            else:
                # Merge current group
                group_boxes = sorted_boxes[current_group]
                group_scores = sorted_class_scores[current_group]
                group_ids = sorted_class_ids[current_group]

                min_x1 = np.min(group_boxes[:, 0])
                min_y1 = np.min(group_boxes[:, 1])
                max_x2 = np.max(group_boxes[:, 2])
                max_y2 = np.max(group_boxes[:, 3])

                # Convert back to xywhn
                merged_x_center = (min_x1 + max_x2) / 2
                merged_y_center = (min_y1 + max_y2) / 2
                merged_width = max_x2 - min_x1
                merged_height = max_y2 - min_y1

                merged_boxes.append([merged_x_center, merged_y_center, merged_width, merged_height])
                # Convert class IDs to digit strings and join them
                digit_strings = [ID2NAME[id] for id in group_ids]
                merged_ids.append("".join(digit_strings))
                merged_scores.append(np.mean(group_scores))  # Average score for the group

                current_group = [i]  # Start new group with current box

    # Merge the last group
    if current_group:
        group_boxes = sorted_boxes[current_group]
        group_scores = sorted_class_scores[current_group]
        group_ids = sorted_class_ids[current_group]

        min_x1 = np.min(group_boxes[:, 0])
        min_y1 = np.min(group_boxes[:, 1])
        max_x2 = np.max(group_boxes[:, 2])
        max_y2 = np.max(group_boxes[:, 3])

        merged_x_center = (min_x1 + max_x2) / 2
        merged_y_center = (min_y1 + max_y2) / 2
        merged_width = max_x2 - min_x1
        merged_height = max_y2 - min_y1

        merged_boxes.append([merged_x_center, merged_y_center, merged_width, merged_height])
        digit_strings = [ID2NAME[id] for id in group_ids]
        merged_ids.append("".join(digit_strings))
        merged_scores.append(np.mean(group_scores))

    return np.array(merged_boxes), np.array(merged_ids), np.array(merged_scores)
