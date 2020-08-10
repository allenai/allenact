from typing import Dict, Any, List, Optional

from torch import nn

from core.base_abstractions.experiment_config import ExperimentConfig
from utils.experiment_utils import TrainingPipeline
from core.base_abstractions.task import TaskSampler


class MyConfig(ExperimentConfig):
    MY_VAR: int = 3

    def tag(cls) -> str:
        return ""

    @classmethod
    def training_pipeline(cls, **kwargs) -> TrainingPipeline:
        return None

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return None

    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        return None

    def my_var_is(cls, val):
        assert cls.MY_VAR == val


class MySpecConfig(MyConfig):
    @classmethod
    def machine_params(cls, mode="train", **kwargs) -> Dict[str, Any]:
        return {}

    MY_VAR = 6


scfg = MySpecConfig()


class TestFrozenAttribs(object):
    def test_frozen_inheritance(self, tmpdir):
        from abc import abstractmethod
        from core.base_abstractions.experiment_config import FrozenClassVariables

        class SomeBase(metaclass=FrozenClassVariables):
            yar = 3

            @abstractmethod
            def use(cls):
                raise NotImplementedError()

        class SomeDerived(SomeBase):
            yar = 33

            def use(cls):
                return cls.yar

        failed = False
        try:
            SomeDerived.yar = 6  # Error
        except Exception as _:
            failed = True
        assert failed

        inst = SomeDerived()
        inst2 = SomeDerived()
        inst.yar = 12  # No error
        assert inst.use() == 12
        assert inst2.use() == 33

    @staticmethod
    def my_func(config, val):
        config.my_var_is(val)

    def test_frozen_experiment_config(self, tmpdir):
        import torch.multiprocessing as mp

        val = 5

        failed = False
        try:
            MyConfig()
        except:
            failed = True
        assert failed

        scfg.MY_VAR = val
        scfg.my_var_is(val)

        failed = False
        try:
            MyConfig.MY_VAR = val
        except RuntimeError:
            failed = True
        assert failed

        failed = False
        try:
            MySpecConfig.MY_VAR = val
        except RuntimeError:
            failed = True
        assert failed

        for fork_method in ["forkserver", "fork"]:
            ctxt = mp.get_context(fork_method)
            p = ctxt.Process(target=self.my_func, kwargs=dict(config=scfg, val=val))
            p.start()
            p.join()


if __name__ == "__main__":
    TestFrozenAttribs().test_frozen_inheritance()
    TestFrozenAttribs().test_frozen_experiment_config()
