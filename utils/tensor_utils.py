"""Functions used to manipulate pytorch tensors and numpy arrays."""

import numbers
import os
import tempfile
import typing
from collections import defaultdict
from typing import List, Dict, Optional, DefaultDict, Union, Any

import PIL
import numpy as np
import torch
from PIL import Image
from moviepy import editor as mpy
from moviepy.editor import concatenate_videoclips
from tensorboardX import SummaryWriter as TBXSummaryWriter, summary as tbxsummary
from tensorboardX.proto.summary_pb2 import Summary as TBXSummary
from tensorboardX.utils import _prepare_video as tbx_prepare_video
from tensorboardX.x2num import make_np as tbxmake_np

from utils.system import get_logger


def to_device_recursively(input: Any, device: str, inplace: bool = True):
    """Recursively places tensors on the appropriate device."""
    if input is None:
        return input
    elif isinstance(input, torch.Tensor):
        return input.to(device)
    elif isinstance(input, tuple):
        return tuple(
            to_device_recursively(input=subinput, device=device, inplace=inplace)
            for subinput in input
        )
    elif isinstance(input, list):
        if inplace:
            for i in range(len(input)):
                input[i] = to_device_recursively(
                    input=input[i], device=device, inplace=inplace
                )
            return input
        else:
            return [
                to_device_recursively(input=subpart, device=device, inplace=inplace)
                for subpart in input
            ]
    elif isinstance(input, dict):
        if inplace:
            for key in input:
                input[key] = to_device_recursively(
                    input=input[key], device=device, inplace=inplace
                )
            return input
        else:
            return {
                k: to_device_recursively(input=input[k], device=device, inplace=inplace)
                for k in input
            }
    elif isinstance(input, set):
        if inplace:
            for element in list(input):
                input.remove(element)
                input.add(
                    to_device_recursively(element, device=device, inplace=inplace)
                )
        else:
            return set(
                to_device_recursively(k, device=device, inplace=inplace) for k in input
            )
    elif isinstance(input, np.ndarray) or np.isscalar(input) or isinstance(input, str):
        return input
    elif hasattr(input, "to"):
        # noinspection PyCallingNonCallable
        return input.to(device=device, inplace=inplace)
    else:
        raise NotImplementedError(
            "Sorry, value of type {} is not supported.".format(type(input))
        )


def detach_recursively(input: Any, inplace=True):
    """Recursively detaches tensors in some data structure from their
    computation graph."""
    if input is None:
        return input
    elif isinstance(input, torch.Tensor):
        return input.detach()
    elif isinstance(input, tuple):
        return tuple(
            detach_recursively(input=subinput, inplace=inplace) for subinput in input
        )
    elif isinstance(input, list):
        if inplace:
            for i in range(len(input)):
                input[i] = detach_recursively(input[i], inplace=inplace)
            return input
        else:
            return [
                detach_recursively(input=subinput, inplace=inplace)
                for subinput in input
            ]
    elif isinstance(input, dict):
        if inplace:
            for key in input:
                input[key] = detach_recursively(input[key], inplace=inplace)
            return input
        else:
            return {k: detach_recursively(input[k], inplace=inplace) for k in input}
    elif isinstance(input, set):
        if inplace:
            for element in list(input):
                input.remove(element)
                input.add(detach_recursively(element, inplace=inplace))
        else:
            return set(detach_recursively(k, inplace=inplace) for k in input)
    elif isinstance(input, np.ndarray) or np.isscalar(input) or isinstance(input, str):
        return input
    elif hasattr(input, "detach_recursively"):
        # noinspection PyCallingNonCallable
        return input.detach_recursively(inplace=inplace)
    else:
        raise NotImplementedError(
            "Sorry, hidden state of type {} is not supported.".format(type(input))
        )


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
    # for obs in observations:
    #     for sensor in obs:
    #         batch[sensor].append(to_tensor(obs[sensor]))
    #
    # for sensor in batch:
    #     batch[sensor] = torch.stack(batch[sensor], dim=0).to(device=device)

    def dict_from_observation(
        observation: Dict[str, Any]
    ) -> Dict[str, Union[Dict, List]]:
        batch: DefaultDict = defaultdict(list)

        for sensor in observation:
            if isinstance(observation[sensor], Dict):
                batch[sensor] = dict_from_observation(observation[sensor])
            else:
                batch[sensor].append(to_tensor(observation[sensor]))

        return batch

    def fill_dict_from_observations(
        batch: Dict[str, Union[Dict, List]], observation: Dict[str, Any]
    ) -> None:
        for sensor in observation:
            if isinstance(observation[sensor], Dict):
                fill_dict_from_observations(batch[sensor], observation[sensor])
            else:
                batch[sensor].append(to_tensor(observation[sensor]))

    def dict_to_batch(
        batch: Dict[str, Union[Dict, List]], device: Optional[torch.device] = None
    ) -> None:
        batch = typing.cast(Union[Dict, List, torch.Tensor], batch)
        for sensor in batch:
            if isinstance(batch[sensor], Dict):
                dict_to_batch(batch[sensor], device)
            else:
                batch[sensor] = torch.stack(batch[sensor], dim=0).to(device=device)

    if len(observations) == 0:
        return typing.cast(Dict[str, Union[Dict, torch.Tensor]], observations)

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


class SummaryWriter(TBXSummaryWriter):
    def _video(self, tag, vid):
        tag = tbxsummary._clean_tag(tag)
        return TBXSummary(value=[TBXSummary.Value(tag=tag, image=vid)])

    def add_vid(self, tag, vid, global_step=None, walltime=None):
        self._get_file_writer().add_summary(
            self._video(tag, vid), global_step, walltime
        )

    def add_image(
        self, tag, img_tensor, global_step=None, walltime=None, dataformats="CHW"
    ):
        self._get_file_writer().add_summary(
            image(tag, img_tensor, dataformats=dataformats), global_step, walltime
        )


def image(tag, tensor, rescale=1, dataformats="CHW"):
    """Outputs a `Summary` protocol buffer with images. The summary has up to
    `max_images` summary values containing images. The images are built from
    `tensor` which must be 3-D with shape `[height, width, channels]` and where
    `channels` can be:

    *  1: `tensor` is interpreted as Grayscale.
    *  3: `tensor` is interpreted as RGB.
    *  4: `tensor` is interpreted as RGBA.

    Args:
      tag: A name for the generated node. Will also serve as a series name in
        TensorBoard.
      tensor: A 3-D `uint8` or `float32` `Tensor` of shape `[height, width,
        channels]` where `channels` is 1, 3, or 4.
        'tensor' can either have values in [0, 1] (float32) or [0, 255] (uint8).
        The image() function will scale the image values to [0, 255] by applying
        a scale factor of either 1 (uint8) or 255 (float32).
    Returns:
      A scalar `Tensor` of type `string`. The serialized `Summary` protocol
      buffer.
    """
    tag = tbxsummary._clean_tag(tag)
    tensor = tbxmake_np(tensor)
    tensor = convert_to_HWC(tensor, dataformats)
    # Do not assume that user passes in values in [0, 255], use data type to detect
    if tensor.dtype != np.uint8:
        tensor = (tensor * 255.0).astype(np.uint8)

    image = tbxsummary.make_image(tensor, rescale=rescale)
    return TBXSummary(value=[TBXSummary.Value(tag=tag, image=image)])


def convert_to_HWC(tensor, input_format):  # tensor: numpy array
    import numpy as np

    assert len(set(input_format)) == len(
        input_format
    ), "You can not use the same dimension shordhand twice. \
        input_format: {}".format(
        input_format
    )
    assert len(tensor.shape) == len(
        input_format
    ), "size of input tensor and input format are different. \
        tensor shape: {}, input_format: {}".format(
        tensor.shape, input_format
    )
    input_format = input_format.upper()

    if len(input_format) == 4:
        index = [input_format.find(c) for c in "NCHW"]
        tensor_NCHW = tensor.transpose(index)
        tensor_CHW = make_grid(tensor_NCHW)
        return tensor_CHW.transpose(1, 2, 0)

    if len(input_format) == 3:
        index = [input_format.find(c) for c in "HWC"]
        tensor_HWC = tensor.transpose(index)
        if tensor_HWC.shape[2] == 1:
            tensor_HWC = np.concatenate([tensor_HWC, tensor_HWC, tensor_HWC], 2)
        return tensor_HWC

    if len(input_format) == 2:
        index = [input_format.find(c) for c in "HW"]
        tensor = tensor.transpose(index)
        tensor = np.stack([tensor, tensor, tensor], 2)
        return tensor


def make_grid(I, ncols=8):
    # I: N1HW or N3HW
    import numpy as np

    assert isinstance(I, np.ndarray), "plugin error, should pass numpy array here"
    if I.shape[1] == 1:
        I = np.concatenate([I, I, I], 1)
    assert I.ndim == 4 and I.shape[1] == 3 or I.shape[1] == 4
    nimg = I.shape[0]
    H = I.shape[2]
    W = I.shape[3]
    ncols = min(nimg, ncols)
    nrows = int(np.ceil(float(nimg) / ncols))
    canvas = np.zeros((I.shape[1], H * nrows, W * ncols), dtype=I.dtype)
    i = 0
    for y in range(nrows):
        for x in range(ncols):
            if i >= nimg:
                break
            canvas[:, y * H : (y + 1) * H, x * W : (x + 1) * W] = I[i]
            i = i + 1
    return canvas


def tensor_to_video(tensor, fps=4):
    tensor = tbxmake_np(tensor)
    tensor = tbx_prepare_video(tensor)
    # If user passes in uint8, then we don't need to rescale by 255
    if tensor.dtype != np.uint8:
        tensor = (tensor * 255.0).astype(np.uint8)

    return tbxsummary.make_video(tensor, fps)


def tensor_to_clip(tensor, fps=4):
    tensor = tbxmake_np(tensor)
    tensor = tbx_prepare_video(tensor)
    # If user passes in uint8, then we don't need to rescale by 255
    if tensor.dtype != np.uint8:
        tensor = (tensor * 255.0).astype(np.uint8)

    t, h, w, c = tensor.shape

    clip = mpy.ImageSequenceClip(list(tensor), fps=fps)

    return clip, (h, w, c)


def clips_to_video(clips, h, w, c):
    # encode sequence of images into gif string
    clip = concatenate_videoclips(clips)

    filename = tempfile.NamedTemporaryFile(suffix=".gif", delete=False).name

    # moviepy >= 1.0.0 use logger=None to suppress output.
    try:
        clip.write_gif(filename, verbose=False, logger=None)
    except TypeError:
        get_logger().warning(
            "Upgrade to moviepy >= 1.0.0 to suppress the progress bar."
        )
        clip.write_gif(filename, verbose=False)

    with open(filename, "rb") as f:
        tensor_string = f.read()

    try:
        os.remove(filename)
    except OSError:
        get_logger().warning("The temporary file used by moviepy cannot be deleted.")

    return TBXSummary.Image(
        height=h, width=w, colorspace=c, encoded_image_string=tensor_string
    )


def process_video(render, max_clip_len=500, max_video_len=-1):
    output = []
    hwc = None
    if len(render) > 0:
        if len(render) > max_video_len > 0:
            get_logger().warning(
                "Clipping video to first {} frames out of {} original frames".format(
                    max_video_len, len(render)
                )
            )
            render = render[:max_video_len]
        for clipstart in range(0, len(render), max_clip_len):
            clip = render[clipstart : clipstart + max_clip_len]
            try:
                current = np.stack(clip, axis=0)  # T, H, W, C
                current = current.transpose((0, 3, 1, 2))  # T, C, H, W
                current = np.expand_dims(current, axis=0)  # 1, T, C, H, W
                current, cur_hwc = tensor_to_clip(current, fps=4)

                if hwc is None:
                    hwc = cur_hwc
                else:
                    assert (
                        hwc == cur_hwc
                    ), "Inconsistent clip shape: previous {} current {}".format(
                        hwc, cur_hwc
                    )

                output.append(current)
            except MemoryError:
                get_logger().error(
                    "Skipping video due to memory error with clip of length {}".format(
                        len(clip)
                    )
                )
                return None
    else:
        get_logger().warning("Calling process_video with 0 frames")
        return None

    assert len(output) > 0, "No clips to concatenate"
    assert hwc is not None, "No tensor dims assigned"

    try:
        result = clips_to_video(output, *hwc)
    except MemoryError:
        get_logger().error("Skipping video due to memory error calling clips_to_video")
        result = None

    return result


class ScaleBothSides(object):
    """Rescales the input PIL.Image to the given 'width' and `height`.

    Attributes
        width: new width
        height: new height
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, width: int, height: int, interpolation=Image.BILINEAR):
        self.width = width
        self.height = height
        self.interpolation = interpolation

    def __call__(self, img: PIL.Image) -> PIL.Image:
        return img.resize((self.width, self.height), self.interpolation)
