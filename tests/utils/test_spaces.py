from typing import Tuple
from collections import OrderedDict
import warnings

import numpy as np
import torch
from gym import spaces as gyms

from utils import spaces_utils as su


class TestSpaces(object):
    space = gyms.Dict(
        {
            "first": gyms.Tuple(
                [
                    gyms.Box(-10, 10, (3, 4)),
                    gyms.MultiDiscrete([2, 3, 4]),
                    gyms.Box(-1, 1, ()),
                ]
            ),
            "second": gyms.Tuple(
                [gyms.Dict({"third": gyms.Discrete(11)}), gyms.MultiBinary(8),]
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
            # np.array_equal also works for torch tensors and scalars
            if bidx is None:
                return np.array_equal(a, b)
            else:
                return np.array_equal(a, b[bidx])

    def test_conversion(self):
        gsample = self.space.sample()

        asample = su.torch_point(self.space, gsample)

        back = su.numpy_point(self.space, asample)

        assert self.same(back, gsample)

    def test_flatten(self):
        # We flatten Discrete to 1 value
        assert su.flatdim(self.space) == 25
        # gym flattens Discrete to one-hot
        assert gyms.flatdim(self.space) == 35

        asample = su.torch_point(self.space, self.space.sample())
        flattened = su.flatten(self.space, asample)
        unflattened = su.unflatten(self.space, flattened)
        assert self.same(asample, unflattened)

        # suppress `UserWarning: WARN: Box bound precision lowered by casting to float32`
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            flattened_space = su.flatten_space(self.space)
            assert flattened_space.shape == (25,)
            # The maximum comes from Discrete(11)
            assert flattened_space.high.max() == 11.0
            assert flattened_space.low.min() == -10.0

            gym_flattened_space = gyms.flatten_space(self.space)
            assert gym_flattened_space.shape == (35,)
            # The maximum comes from Box(-10, 10, (3, 4))
            assert gym_flattened_space.high.max() == 10.0
            assert gym_flattened_space.low.min() == -10.0

    def test_batched(self):
        samples = [self.space.sample() for _ in range(10)]
        flattened = [
            su.flatten(self.space, su.torch_point(self.space, sample))
            for sample in samples
        ]
        stacked = torch.stack(flattened, dim=0)
        unflattened = su.unflatten(self.space, stacked)
        for bidx, refsample in enumerate(samples):
            # Compare each torch-ified sample to the corresponding unflattened from the stack
            assert self.same(su.torch_point(self.space, refsample), unflattened, bidx)

        assert self.same(su.flatten(self.space, unflattened), stacked)

    def test_log_prob_space(self):
        lp_space = gyms.Dict(
            {
                "first": gyms.Tuple(
                    [
                        gyms.Box(-np.inf, np.inf, (3, 4)),
                        gyms.Box(-np.inf, 0, (3,)),
                        gyms.Box(-np.inf, np.inf, ()),
                    ]
                ),
                "second": gyms.Tuple(
                    [
                        gyms.Dict({"third": gyms.Box(-np.inf, 0, ())}),
                        gyms.Box(-np.inf, 0, (8,)),
                    ]
                ),
            }
        )

        lp_auto = su.log_prob_space(self.space)

        lp_sample = su.torch_point(lp_space, lp_space.sample())
        auto_sample = su.torch_point(lp_auto, lp_auto.sample())

        for sample in [lp_sample, auto_sample]:
            lp_flatten = su.flatten(lp_space, sample)
            auto_flatten = su.flatten(lp_auto, sample)

            assert np.array_equal(lp_flatten, auto_flatten)

            assert self.same(
                su.unflatten(lp_auto, lp_flatten), su.unflatten(lp_space, auto_flatten),
            )


if __name__ == "__main__":
    TestSpaces().test_conversion()  # type:ignore
    TestSpaces().test_flatten()  # type:ignore
    TestSpaces().test_batched()  # type:ignore
    TestSpaces().test_log_prob_space()  # type:ignore
