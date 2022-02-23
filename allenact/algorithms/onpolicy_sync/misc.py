from enum import Enum
from typing import Dict, Any, Optional

import attr


class TrackingInfoType(Enum):
    LOSS = "loss"
    TEACHER_FORCING = "teacher_forcing"
    UPDATE_INFO = "update_info"


@attr.s(kw_only=True)
class TrackingInfo:
    type: TrackingInfoType = attr.ib()
    info: Dict[str, Any] = attr.ib()
    n: int = attr.ib()
    storage_uuid: Optional[str] = attr.ib()
    stage_component_uuid: Optional[str] = attr.ib()
