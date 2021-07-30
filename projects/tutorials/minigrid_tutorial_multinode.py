from allenact.base_abstractions.experiment_config import MachineParams
from projects.tutorials.minigrid_tutorial import MiniGridTutorialExperimentConfig


class MiniGridTutorialMultiNodeExperimentConfig(MiniGridTutorialExperimentConfig):
    @classmethod
    def tag(cls) -> str:
        return "MiniGridTutorialMultiNode"

    @classmethod
    def machine_params(cls, mode="train", **kwargs):
        nmachines = 2
        machine_procs = 2

        mp = MachineParams(
            nprocesses=[128 // (nmachines * machine_procs)] * nmachines * machine_procs
            if mode == "train"
            else 16,
            devices=["cpu"] * nmachines * machine_procs if mode == "train" else [],
        )

        if "machine_id" in kwargs:
            assert mode == "train"
            machine_id = kwargs["machine_id"]
            assert 0 <= machine_id < nmachines
            mp.set_local_worker_ids(
                list(
                    range(machine_procs * machine_id, machine_procs * (machine_id + 1))
                )
            )

        return mp
