import torch.nn as nn
import torch
from gym.spaces.dict import Dict as SpaceDict


class RobothorTargetImTensorProcessor(nn.Module):
    def __init__(
        self,
        observation_spaces: SpaceDict,
        goal_sensor_uuid: str,
        output_dims: int = 1568,
        class_emb_dims: int = 32,
    ) -> None:
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid
        self.embed_classes = nn.Embedding(
            observation_spaces.spaces[self.goal_sensor_uuid].n, class_emb_dims
        )

        self.im_compressor = nn.Sequential(
            nn.Conv2d(512, 128, 1), nn.ReLU(), nn.Conv2d(128, 32, 1)
        )

        assert output_dims % (7 * 7) == 0, "output dims must be a multiple of 7 x 7"

        self.target_viz_projector = nn.Sequential(
            nn.Conv2d(32 * 2, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, output_dims // (7 * 7), 1),
        )

    @property
    def is_blind(self):
        return False

    def forward(self, observations):
        im, target = observations["rgb_resnet"], observations[self.goal_sensor_uuid]

        target_emb = self.embed_classes(target).view(-1, 32, 1, 1)

        im = self.im_compressor(im)  # project features to 32-d

        x = self.target_viz_projector(
            torch.cat(
                (im, target_emb.expand(-1, -1, im.shape[-2], im.shape[-1])), dim=-3
            )
        )  #  adds projected target

        x = x.view(x.size(0), -1)  # flatten

        return x
