# utils.tensor_utils [[source]](https://github.com/allenai/embodied-rl/tree/master/utils/tensor_utils.py)
Functions used to manipulate pytorch tensors and numpy arrays.
## batch_observations
```python
batch_observations(
    observations: List[Dict],
    device: Optional[torch.device] = None,
) -> Dict[str, Union[Dict, torch.Tensor]]
```
Transpose a batch of observation dicts to a dict of batched
observations.

__Arguments__


- __observations __:  List of dicts of observations.
- __device __: The torch.device to put the resulting tensors on.
    Will not move the tensors if None.

__Returns__


Transposed dict of lists of observations.

## detach_recursively
```python
detach_recursively(input: Any, inplace=True)
```
Recursively detaches tensors in some data structure from their
computation graph.
## image
```python
image(tag, tensor, rescale=1, dataformats='CHW')
```
Outputs a `Summary` protocol buffer with images. The summary has up to
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

## ScaleBothSides
```python
ScaleBothSides(self, width: int, height: int, interpolation=2)
```
Rescales the input PIL.Image to the given 'width' and `height`.

Attributes
    width: new width
    height: new height
    interpolation: Default: PIL.Image.BILINEAR

## tile_images
```python
tile_images(images: List[numpy.ndarray]) -> numpy.ndarray
```
Tile multiple images into single image.

__Parameters__


- __images __: list of images where each image has dimension
    (height x width x channels)

__Returns__


Tiled image (new_height x width x channels).

## to_device_recursively
```python
to_device_recursively(input: Any, device: str, inplace: bool = True)
```
Recursively places tensors on the appropriate device.
## to_tensor
```python
to_tensor(v) -> torch.Tensor
```
Return a torch.Tensor version of the input.

__Parameters__


- __v __: Input values that can be coerced into being a tensor.

__Returns__


A tensor version of the input.

