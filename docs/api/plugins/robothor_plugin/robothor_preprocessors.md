# plugins.robothor_plugin.robothor_preprocessors [[source]](https://github.com/allenai/allenact/tree/master/plugins/robothor_plugin/robothor_preprocessors.py)

## BatchedFasterRCNN
```python
BatchedFasterRCNN(self, thres=0.12, maxdets=3, res=7)
```

### COCO_INSTANCE_CATEGORY_NAMES
Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.
## FasterRCNNPreProcessorRoboThor
```python
FasterRCNNPreProcessorRoboThor(
    self,
    input_uuids: List[str],
    output_uuid: str,
    input_height: int,
    input_width: int,
    max_dets: int,
    detector_spatial_res: int,
    detector_thres: float,
    parallel: bool = False,
    device: Optional[torch.device] = None,
    device_ids: Optional[List[torch.device]] = None,
    kwargs: Any,
)
```
Preprocess RGB image using a ResNet model.
### COCO_INSTANCE_CATEGORY_NAMES
Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.
