from enum import Enum
from typing import Dict, Any, Optional

import attr


class TrackingInfoType(Enum):
    LOSS = "loss"
    TEACHER_FORCING = "teacher_forcing"
    UPDATE_INFO = "update_info"


@attr.define
class TrackingInfo:
    type: TrackingInfoType
    info: Dict[str, Any]
    n: int
    storage_uuid: Optional[str]
    stage_component_uuid: Optional[str]
