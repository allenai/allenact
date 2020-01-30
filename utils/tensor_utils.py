import numbers
from collections import defaultdict
from typing import List, Dict, Optional, DefaultDict

import numpy as np
import torch


def batch_observations(
    observations: List[Dict], device: Optional[torch.device] = None
) -> Dict[str, torch.Tensor]:
    """Transpose a batch of observation dicts to a dict of batched
    observations.

    Args:
        observations:  list of dicts of observations.
        device: The torch.device to put the resulting tensors on.
            Will not move the tensors if None

    Returns:
        transposed dict of lists of observations.
    """
    batch: DefaultDict = defaultdict(list)

    for obs in observations:
        for sensor in obs:
            batch[sensor].append(to_tensor(obs[sensor]))

    for sensor in batch:
        batch[sensor] = torch.stack(batch[sensor], dim=0).to(device=device)

    return batch


def to_tensor(v):
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(
            v, dtype=torch.int64 if isinstance(v, numbers.Integral) else torch.float
        )


def tile_images(images: List[np.ndarray]) -> np.ndarray:
    """Tile multiple images into single image.

    # Parameters

    images : list of images where each image has dimension
        (height x width x channels)

    # Returns

    Tiled image (new_height x width x channels).
    """
    assert len(images) > 0, "empty list of images"
    np_images = np.asarray(images)
    n_images, height, width, n_channels = np_images.shape
    new_height = int(np.ceil(np.sqrt(n_images)))
    new_width = int(np.ceil(float(n_images) / new_height))
    # pad with empty images to complete the rectangle
    np_images = np.array(
        images + [images[0] * 0 for _ in range(n_images, new_height * new_width)]
    )
    # img_HWhwc
    out_image = np_images.reshape((new_height, new_width, height, width, n_channels))
    # img_HhWwc
    out_image = out_image.transpose(0, 2, 1, 3, 4)
    # img_Hh_Ww_c
    out_image = out_image.reshape((new_height * height, new_width * width, n_channels))
    return out_image
