import torch

from .bc_teacher_forcing import BCTeacherForcingBabyAIGoToLocalExperimentConfig
from utils.experiment_utils import PipelineStage, LinearDecay


class DistributedBCTeacherForcingBabyAIGoToLocalExperimentConfig(
    BCTeacherForcingBabyAIGoToLocalExperimentConfig
):
    """Distributed behavior clone with teacher forcing."""

    USE_EXPERT = True

    GPU_ID = 0 if torch.cuda.is_available() else None

    @classmethod
    def METRIC_ACCUMULATE_INTERVAL(cls):
        return 1

    @classmethod
    def tag(cls):
        return "BabyAIGoToLocalBCTeacherForcingDistributed"

    @classmethod
    def machine_params(
        cls, mode="train", gpu_id="default", n_train_processes="default", **kwargs
    ):
        res = super().machine_params(mode, gpu_id, n_train_processes, **kwargs)

        if res["nprocesses"] > 0 and torch.cuda.is_available():
            ngpu_to_use = min(torch.cuda.device_count(), 2)
            res["nprocesses"] = [res["nprocesses"] // ngpu_to_use] * ngpu_to_use
            res["gpu_ids"] = list(range(ngpu_to_use))

        return res
