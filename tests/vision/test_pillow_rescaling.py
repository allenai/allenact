import hashlib
import os

import imageio
import numpy as np
from torchvision.transforms import transforms

from allenact.utils.tensor_utils import ScaleBothSides
from constants import ABS_PATH_OF_TOP_LEVEL_DIR

to_pil = transforms.ToPILImage()  # Same as used by the vision sensors


class TestPillowRescaling(object):
    def _load_thor_img(self) -> np.ndarray:
        img_path = os.path.join(
            ABS_PATH_OF_TOP_LEVEL_DIR, "docs/img/iTHOR_framework.jpg"
        )
        img = imageio.imread(img_path)
        return img

    def _get_img_hash(self, img: np.ndarray) -> str:
        img_hash = hashlib.sha1(np.ascontiguousarray(img))
        return img_hash.hexdigest()

    def _random_rgb_image(self, width: int, height: int, seed: int) -> np.ndarray:
        s = np.random.get_state()
        np.random.seed(seed)
        img = np.random.randint(
            low=0, high=256, size=(width, height, 3), dtype=np.uint8
        )
        np.random.set_state(s)
        return img

    def _random_depthmap(
        self, width: int, height: int, max_depth: float, seed: int
    ) -> np.ndarray:
        s = np.random.get_state()
        np.random.seed(seed)
        img = max_depth * np.random.rand(width, height, 1)
        np.random.set_state(s)
        return np.float32(img)

    def test_scaler_rgb_thor(self):
        thor_img_arr = np.uint8(self._load_thor_img())

        assert (
            self._get_img_hash(thor_img_arr)
            == "80ff8a342b4f74966796eee91babde31409d0457"
        )

        img = to_pil(thor_img_arr)

        scaler = ScaleBothSides(width=75, height=75)
        scaled_img = np.array(scaler(img))
        assert (
            self._get_img_hash(scaled_img) == "2c47057aa188240cb21b2edc39e0f269c1085bac"
        )

        scaler = ScaleBothSides(width=500, height=600)
        scaled_img = np.array(scaler(img))
        assert (
            self._get_img_hash(scaled_img) == "faf0be2b9ec9bfd23a1b7b465c86ad961d03c259"
        )

    def test_scaler_rgb_random(self):
        arr = self._random_rgb_image(width=100, height=100, seed=1)

        assert self._get_img_hash(arr) == "d01bd8ba151ab790fde9a8cc29aa8a3c63147334"

        img = to_pil(arr)

        scaler = ScaleBothSides(width=60, height=60)
        scaled_img = np.array(scaler(img))
        assert (
            self._get_img_hash(scaled_img) == "22473537e50d5e39abeeec4f92dbfde51c754010"
        )

        scaler = ScaleBothSides(width=1000, height=800)
        scaled_img = np.array(scaler(img))
        assert (
            self._get_img_hash(scaled_img) == "5e5b955981e4ee3b5e22287536040d001a31fbd3"
        )

    def test_scaler_depth_thor(self):
        thor_depth_arr = 5 * np.float32(self._load_thor_img()).sum(-1)
        thor_depth_arr /= thor_depth_arr.max()

        assert (
            self._get_img_hash(thor_depth_arr)
            == "d3c1474400ba57ed78f52cf4ba6a4c2a1d90516c"
        )

        img = to_pil(thor_depth_arr)

        scaler = ScaleBothSides(width=75, height=75)
        scaled_img = np.array(scaler(img))
        assert (
            self._get_img_hash(scaled_img) == "6a879beb6bed49021e438c1e3af7a62c428a44d8"
        )

        scaler = ScaleBothSides(width=500, height=600)
        scaled_img = np.array(scaler(img))
        assert (
            self._get_img_hash(scaled_img) == "79f11fb741ae638afca40125e4c501f54b22cc01"
        )

    def test_scaler_depth_random(self):
        depth_arr = self._random_depthmap(width=96, height=103, max_depth=5.0, seed=1)

        assert (
            self._get_img_hash(depth_arr) == "cbd8ca127951ffafb6848536d9d731970a5397e9"
        )

        img = to_pil(depth_arr)

        scaler = ScaleBothSides(width=60, height=60)
        scaled_img = np.array(scaler(img))
        assert (
            self._get_img_hash(scaled_img) == "5bed173f2d783fb2badcde9b43904ef85a1a5820"
        )

        scaler = ScaleBothSides(width=1000, height=800)
        scaled_img = np.array(scaler(img))
        assert (
            self._get_img_hash(scaled_img) == "9dceb7f77d767888f24a84c00913c0cf4ccd9d49"
        )


if __name__ == "__main__":
    TestPillowRescaling().test_scaler_rgb_thor()
    TestPillowRescaling().test_scaler_rgb_random()
    TestPillowRescaling().test_scaler_depth_thor()
    TestPillowRescaling().test_scaler_depth_random()
