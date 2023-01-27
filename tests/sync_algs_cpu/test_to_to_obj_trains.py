import io
import math
import os
import pathlib
from contextlib import redirect_stdout, redirect_stderr
from typing import Optional, List, Dict, Any

import torch

from allenact.algorithms.onpolicy_sync.losses.abstract_loss import (
    AbstractActorCriticLoss,
)
from allenact.algorithms.onpolicy_sync.policy import ObservationType
from allenact.algorithms.onpolicy_sync.runner import OnPolicyRunner
from allenact.algorithms.onpolicy_sync.storage import (
    StreamingStorageMixin,
    ExperienceStorage,
    RolloutBlockStorage,
)
from allenact.base_abstractions.experiment_config import MachineParams
from allenact.base_abstractions.misc import (
    Memory,
    GenericAbstractLoss,
    ModelType,
    LossOutput,
)
from allenact.utils.experiment_utils import PipelineStage, StageComponent
from allenact.utils.misc_utils import prepare_locals_for_super
from projects.babyai_baselines.experiments.go_to_obj.ppo import (
    PPOBabyAIGoToObjExperimentConfig,
)

SILLY_STORAGE_VALUES = [1.0, 2.0, 3.0, 4.0]
SILLY_STORAGE_REPEATS = [1, 2, 3, 4]


class FixedConstantLoss(AbstractActorCriticLoss):
    def __init__(self, name: str, value: float):
        super().__init__()
        self.name = name
        self.value = value

    def loss(  # type: ignore
        self, *args, **kwargs,
    ):
        return self.value, {self.name: self.value}


class SillyStorage(ExperienceStorage, StreamingStorageMixin):
    def __init__(self, values_to_return: List[float], repeats: List[int]):
        self.values_to_return = values_to_return
        self.repeats = repeats
        assert len(self.values_to_return) == len(self.repeats)
        self.index = 0

    def initialize(self, *, observations: ObservationType, **kwargs):
        pass

    def add(
        self,
        observations: ObservationType,
        memory: Optional[Memory],
        actions: torch.Tensor,
        action_log_probs: torch.Tensor,
        value_preds: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor,
    ):
        pass

    def to(self, device: torch.device):
        pass

    def set_partition(self, index: int, num_parts: int):
        pass

    @property
    def total_experiences(self) -> int:
        return 0

    @total_experiences.setter
    def total_experiences(self, value: int):
        pass

    def next_batch(self) -> Dict[str, Any]:
        if self.index >= len(self.values_to_return):
            raise EOFError

        to_return = {
            "value": torch.tensor(
                [self.values_to_return[self.index]] * self.repeats[self.index]
            ),
        }
        self.index += 1
        return to_return

    def reset_stream(self):
        self.index = 0

    def empty(self) -> bool:
        return len(self.values_to_return) == 0


class AverageBatchValueLoss(GenericAbstractLoss):
    def loss(
        self,
        *,
        model: ModelType,
        batch: ObservationType,
        batch_memory: Memory,
        stream_memory: Memory,
    ) -> LossOutput:
        v = batch["value"].mean()
        return LossOutput(
            value=v,
            info={"avg_batch_val": v},
            per_epoch_info={},
            batch_memory=batch_memory,
            stream_memory=stream_memory,
            bsize=batch["value"].shape[0],
        )


class PPOBabyAIGoToObjTestExperimentConfig(PPOBabyAIGoToObjExperimentConfig):
    NUM_CKPTS_TO_SAVE = 2

    @classmethod
    def tag(cls):
        return "BabyAIGoToObjPPO-TESTING"

    @classmethod
    def machine_params(cls, mode="train", **kwargs):
        mp = super().machine_params(mode=mode, **kwargs)
        if mode == "valid":
            mp = MachineParams(
                nprocesses=1,
                devices=mp.devices,
                sensor_preprocessor_graph=mp.sensor_preprocessor_graph,
                sampler_devices=mp.sampler_devices,
                visualizer=mp.visualizer,
                local_worker_ids=mp.local_worker_ids,
            )
        return mp

    @classmethod
    def training_pipeline(cls, **kwargs):
        total_train_steps = cls.TOTAL_RL_TRAIN_STEPS
        ppo_info = cls.rl_loss_default("ppo", steps=total_train_steps)

        tp = cls._training_pipeline(
            named_losses={
                "ppo_loss": ppo_info["loss"],
                "3_loss": FixedConstantLoss("3_loss", 3.0),
                "avg_value_loss": AverageBatchValueLoss(),
            },
            named_storages={
                "onpolicy": RolloutBlockStorage(),
                "silly_storage": SillyStorage(
                    values_to_return=SILLY_STORAGE_VALUES, repeats=SILLY_STORAGE_REPEATS
                ),
            },
            pipeline_stages=[
                PipelineStage(
                    loss_names=["ppo_loss", "3_loss"],
                    max_stage_steps=total_train_steps,
                    stage_components=[
                        StageComponent(
                            uuid="onpolicy",
                            storage_uuid="onpolicy",
                            loss_names=["ppo_loss", "3_loss"],
                        )
                    ],
                ),
            ],
            num_mini_batch=ppo_info["num_mini_batch"],
            update_repeats=ppo_info["update_repeats"],
            total_train_steps=total_train_steps,
            valid_pipeline_stage=PipelineStage(
                loss_names=["ppo_loss", "3_loss"],
                max_stage_steps=-1,
                update_repeats=1,
                num_mini_batch=1,
            ),
            test_pipeline_stage=PipelineStage(
                loss_names=["avg_value_loss"],
                stage_components=[
                    StageComponent(
                        uuid="debug",
                        storage_uuid="silly_storage",
                        loss_names=["avg_value_loss"],
                    ),
                ],
                max_stage_steps=-1,
                update_repeats=1,
                num_mini_batch=1,
            ),
        )

        tp.training_settings.save_interval = int(
            math.ceil(cls.TOTAL_RL_TRAIN_STEPS / cls.NUM_CKPTS_TO_SAVE)
        )
        return tp

    def valid_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        # Also run validation
        return self.test_task_sampler_args(**prepare_locals_for_super(locals()))


class TestGoToObjTrains:
    def test_ppo_trains(self, tmpdir):
        cfg = PPOBabyAIGoToObjTestExperimentConfig()

        d = tmpdir / "test_ppo_trains"
        if isinstance(d, pathlib.Path):
            d.mkdir(parents=True, exist_ok=True)
        else:
            d.mkdir()
        output_dir = str(d)

        train_runner = OnPolicyRunner(
            config=cfg,
            output_dir=output_dir,
            loaded_config_src_files=None,
            seed=1,
            mode="train",
            deterministic_cudnn=True,
        )

        f = io.StringIO()
        with redirect_stdout(f), redirect_stderr(f):
            start_time_str = train_runner.start_train(
                max_sampler_processes_per_worker=1
            )
        s = f.getvalue()

        def extract_final_metrics_from_log(s: str, mode: str):
            lines = s.splitlines()
            lines = [l for l in lines if mode.upper() in l]
            metrics_and_losses_list = (
                lines[-1].split(")")[-1].split("[")[0].strip().split(" ")
            )

            def try_float(f):
                try:
                    return float(f)
                except:
                    return f

            metrics_and_losses_dict = {
                k: try_float(v)
                for k, v in zip(
                    metrics_and_losses_list[::2], metrics_and_losses_list[1::2]
                )
            }
            return metrics_and_losses_dict

        train_metrics = extract_final_metrics_from_log(s, "train")
        assert train_metrics["global_batch_size"] == 256

        valid_metrics = extract_final_metrics_from_log(s, "valid")
        assert valid_metrics["3_loss/3_loss"] == 3, "Incorrect validation loss"
        assert (
            valid_metrics["new_tasks_completed"] == cfg.NUM_TEST_TASKS
        ), "Incorrect number of tasks evaluated in validation"

        test_runner = OnPolicyRunner(
            config=cfg,
            output_dir=output_dir,
            loaded_config_src_files=None,
            seed=1,
            mode="test",
            deterministic_cudnn=True,
        )

        test_results = test_runner.start_test(
            checkpoint_path_dir_or_pattern=os.path.join(
                output_dir, "checkpoints", "**", start_time_str, "*.pt"
            ),
            max_sampler_processes_per_worker=1,
        )

        assert (
            len(test_results) == 2
        ), f"Too many or too few test results ({test_results})"

        tr = test_results[-1]
        assert (
            tr["training_steps"]
            == round(
                math.ceil(
                    cfg.TOTAL_RL_TRAIN_STEPS
                    / (cfg.ROLLOUT_STEPS * cfg.NUM_TRAIN_SAMPLERS)
                )
            )
            * cfg.ROLLOUT_STEPS
            * cfg.NUM_TRAIN_SAMPLERS
        ), "Incorrect number of training steps"
        assert len(tr["tasks"]) == cfg.NUM_TEST_TASKS, "Incorrect number of test tasks"
        assert tr["test-metrics/success"] == sum(
            task["success"] for task in tr["tasks"]
        ) / len(tr["tasks"]), "Success counts don't seem to match"
        assert (
            tr["test-metrics/success"] > 0.95
        ), f"PPO did not seem to converge for the go_to_obj task (success {tr['success']})."
        assert tr["test-debug-losses/avg_value_loss/avg_batch_val"] == sum(
            ssv * ssr for ssv, ssr in zip(SILLY_STORAGE_VALUES, SILLY_STORAGE_REPEATS)
        ) / sum(SILLY_STORAGE_REPEATS)
        assert tr["test-debug-losses/avg_value_loss/avg_batch_val"] == sum(
            ssv * ssr for ssv, ssr in zip(SILLY_STORAGE_VALUES, SILLY_STORAGE_REPEATS)
        ) / sum(SILLY_STORAGE_REPEATS)
        assert tr["test-debug-misc/worker_batch_size"] == sum(
            SILLY_STORAGE_VALUES
        ) / len(SILLY_STORAGE_VALUES)


if __name__ == "__main__":
    TestGoToObjTrains().test_ppo_trains(
        pathlib.Path("experiment_output/testing"), using_pytest=True
    )  # type:ignore
