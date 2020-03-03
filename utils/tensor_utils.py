"""Functions used to manipulate pytorch tensors and numpy arrays."""

import numbers
from collections import defaultdict
import typing
from typing import List, Dict, Optional, DefaultDict, Union, Any

import numpy as np
import torch
from tensorboardX import SummaryWriter as SummaryWriterBase, summary as tbxsummary
from tensorboardX.x2num import make_np as tbxmake_np
from tensorboardX.utils import _prepare_video as tbx_prepare_video
from tensorboardX.proto.summary_pb2 import Summary as TBXSummary


def batch_observations(
    observations: List[Dict], device: Optional[torch.device] = None
) -> Dict[str, Union[Dict, torch.Tensor]]:
    """Transpose a batch of observation dicts to a dict of batched
    observations.

    # Arguments

    observations :  List of dicts of observations.
    device : The torch.device to put the resulting tensors on.
        Will not move the tensors if None.

    # Returns

    Transposed dict of lists of observations.
    """
    # batch: DefaultDict = defaultdict(list)
    #
    # for obs in observations[1:]:
    #     for sensor in obs:
    #         batch[sensor].append(to_tensor(obs[sensor]))
    #
    # for sensor in batch:
    #     batch[sensor] = torch.stack(batch[sensor], dim=0).to(device=device)

    def dict_from_observation(observation: Dict[str, Any]) -> Dict[str, List[Any]]:
        batch: DefaultDict = defaultdict(list)

        for sensor in observation:
            if isinstance(observation[sensor], Dict):
                batch[sensor] = dict_from_observation(observation[sensor])
            else:
                batch[sensor].append(to_tensor(observation[sensor]))

        return batch

    def fill_dict_from_observations(batch: Dict[str, Union[Dict, List]], observation: Dict[str, Any]) -> None:
        for sensor in observation:
            if isinstance(observation[sensor], Dict):
                fill_dict_from_observations(batch[sensor], observation[sensor])
            else:
                batch[sensor].append(to_tensor(observation[sensor]))

    def dict_to_batch(batch: Dict[str, Union[Dict, List]], device: Optional[torch.device]=None) -> None:
        for sensor in batch:
            if isinstance(batch[sensor], Dict):
                dict_to_batch(batch[sensor], device)
            else:
                batch[sensor] = torch.stack(batch[sensor], dim=0).to(device=device)

    batch = dict_from_observation(observations[0])
    for obs in observations[1:]:
        fill_dict_from_observations(batch, obs)

    dict_to_batch(batch, device)

    return typing.cast(Dict[str, Union[Dict, torch.Tensor]], batch)


def to_tensor(v) -> torch.Tensor:
    """Return a torch.Tensor version of the input.

    # Parameters

    v : Input values that can be coerced into being a tensor.

    # Returns

    A tensor version of the input.
    """
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


class SummaryWriter(SummaryWriterBase):
    def _video(self, tag, vid):
        tag = tbxsummary._clean_tag(tag)
        return TBXSummary(value=[TBXSummary.Value(tag=tag, image=vid)])

    def add_vid(self, tag, vid, global_step=None, walltime=None):
        self._get_file_writer().add_summary(
            self._video(tag, vid), global_step, walltime
        )


def tensor_to_video(tensor, fps=4):
    tensor = tbxmake_np(tensor)
    tensor = tbx_prepare_video(tensor)
    # If user passes in uint8, then we don't need to rescale by 255
    if tensor.dtype != np.uint8:
        tensor = (tensor * 255.0).astype(np.uint8)

    return tbxsummary.make_video(tensor, fps)
