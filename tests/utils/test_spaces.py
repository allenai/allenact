from typing import Tuple
from collections import OrderedDict

from gym import spaces as gsp
import numpy as np
import torch

from utils import spaces_utils as asp


class TestSpaces(object):
    space = gsp.Dict(
        {
            "first": gsp.Tuple(
                [gsp.Box(-10, 10, (3, 4)), gsp.MultiDiscrete([2, 3, 4]),]
            ),
            "second": gsp.Tuple(
                [gsp.Dict({"third": gsp.Discrete(11)}), gsp.MultiBinary(8),]
            ),
        }
    )

    @staticmethod
    def same(a, b, bidx=None):
        if isinstance(a, OrderedDict):
            for key in a:
                if not TestSpaces.same(a[key], b[key], bidx):
                    return False
            return True
        elif isinstance(a, Tuple):
            for it in range(len(a)):
                if not TestSpaces.same(a[it], b[it], bidx):
                    return False
            return True
        else:
            if bidx is None:
                return np.array_equal(a, b)
            else:
                return np.array_equal(a, b[bidx])

    def test_conversion(self):
        gsample = self.space.sample()

        asample = asp.torch_point(self.space, gsample)

        back = asp.numpy_point(self.space, asample)

        assert self.same(back, gsample)

    def test_flatten(self):
        assert asp.flatdim(self.space) == 24
        assert gsp.flatdim(self.space) == 34

        asample = asp.torch_point(self.space, self.space.sample())
        flattened = asp.flatten(self.space, asample)
        unflattened = asp.unflatten(self.space, flattened)
        assert self.same(asample, unflattened)

        flattened_space = asp.flatten_space(self.space)
        assert flattened_space.shape == (24,)
        assert flattened_space.high.max() == 11.0
        assert flattened_space.low.min() == -10.0

        gym_flattened_space = gsp.flatten_space(self.space)
        assert gym_flattened_space.shape == (34,)
        assert gym_flattened_space.high.max() == 10.0
        assert gym_flattened_space.low.min() == -10.0

    def test_batched(self):
        samples = [self.space.sample() for _ in range(10)]
        fsamples = [
            asp.flatten(self.space, asp.torch_point(self.space, sample))
            for sample in samples
        ]
        stacked = torch.stack(fsamples, dim=0)
        unflattened = asp.unflatten(self.space, stacked)
        for bidx, refsample in enumerate(samples):
            assert self.same(asp.torch_point(self.space, refsample), unflattened, bidx)


if __name__ == "__main__":
    TestSpaces().test_conversion()  # type:ignore
    TestSpaces().test_flatten()  # type:ignore
    TestSpaces().test_batched()  # type:ignore
