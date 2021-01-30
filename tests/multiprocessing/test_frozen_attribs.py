from typing import Dict, Any

import torch
import torch.multiprocessing as mp

from allenact.base_abstractions.experiment_config import ExperimentConfig
from allenact.base_abstractions.task import TaskSampler
from allenact.utils.experiment_utils import TrainingPipeline


# noinspection PyAbstractClass,PyTypeChecker
class MyConfig(ExperimentConfig):
    MY_VAR: int = 3

    @classmethod
    def tag(cls) -> str:
        return ""

    @classmethod
    def training_pipeline(cls, **kwargs) -> TrainingPipeline:
        return None

    @classmethod
    def create_model(cls, **kwargs) -> torch.nn.Module:
        return None

    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        return None

    def my_var_is(self, val):
        assert self.MY_VAR == val


# noinspection PyAbstractClass
class MySpecConfig(MyConfig):
    MY_VAR = 6

    @classmethod
    def machine_params(cls, mode="train", **kwargs) -> Dict[str, Any]:
        return {}

    @classmethod
    def tag(cls) -> str:
        return "SpecTag"


scfg = MySpecConfig()


class TestFrozenAttribs(object):
    def test_frozen_inheritance(self):
        from abc import abstractmethod
        from allenact.base_abstractions.experiment_config import FrozenClassVariables

        class SomeBase(metaclass=FrozenClassVariables):
            yar = 3

            @abstractmethod
            def use(self):
                raise NotImplementedError()

        class SomeDerived(SomeBase):
            yar = 33

            def use(self):
                return self.yar

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

    def test_frozen_experiment_config(self):
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
    TestFrozenAttribs().test_frozen_inheritance()  # type:ignore
    TestFrozenAttribs().test_frozen_experiment_config()  # type:ignore
