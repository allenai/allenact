import abc
from typing import List, Any, Dict
from typing import Sequence
from typing import Union

import gym
import networkx as nx
import torch
from gym.spaces import Dict as SpaceDict

from allenact.utils.experiment_utils import Builder


class Preprocessor(abc.ABC):
    """Represents a preprocessor that transforms data from a sensor or another
    preprocessor to the input of agents or other preprocessors. The user of
    this class needs to implement the process method and the user is also
    required to set the below attributes:

    # Attributes:
        input_uuids : List of input universally unique ids.
        uuid : Universally unique id.
        observation_space : ``gym.Space`` object corresponding to processed observation spaces.
    """

    input_uuids: List[str]
    uuid: str
    observation_space: gym.Space

    def __init__(
        self,
        input_uuids: List[str],
        output_uuid: str,
        observation_space: gym.Space,
        **kwargs: Any
    ) -> None:
        self.uuid = output_uuid
        self.input_uuids = input_uuids
        self.observation_space = observation_space

    @abc.abstractmethod
    def process(self, obs: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        """Returns processed observations from sensors or other preprocessors.

        # Parameters

        obs : Dict with available observations and processed observations.

        # Returns

        Processed observation.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def to(self, device: torch.device) -> "Preprocessor":
        raise NotImplementedError()


class SensorPreprocessorGraph:
    """Represents a graph of preprocessors, with each preprocessor being
    identified through a universally unique id.

    Allows for the construction of observations that are a function of
    sensor readings. For instance, perhaps rather than giving your agent
    a raw RGB image, you'd rather first pass that image through a pre-trained
    convolutional network and only give your agent the resulting features
    (see e.g. the `ResNetPreprocessor` class).

    # Attributes

    preprocessors : List containing preprocessors with required input uuids, output uuid of each
        sensor must be unique.
    observation_spaces: The observation spaces of the values returned when calling `get_observations`.
        By default (see the `additionally_exposed_uuids` parameter to to change this default) the observations
        returned by the `SensorPreprocessorGraph` **include only the sink nodes** of the graph (i.e.
        those that are not used by any other preprocessor).
        Thus if one of the input preprocessors takes as input the `'YOUR_SENSOR_UUID'` sensor, then
        `'YOUR_SENSOR_UUID'` will not be returned when calling `get_observations`.
    device: The `torch.device` upon which the preprocessors are run.
    """

    preprocessors: Dict[str, Preprocessor]
    observation_spaces: SpaceDict
    device: torch.device

    def __init__(
        self,
        source_observation_spaces: SpaceDict,
        preprocessors: Sequence[Union[Preprocessor, Builder[Preprocessor]]],
        additional_output_uuids: Sequence[str] = tuple(),
    ) -> None:
        """Initializer.

        # Parameters

        source_observation_spaces : The observation spaces of all sensors before preprocessing.
            This generally should be the output of `SensorSuite.observation_spaces`.
        preprocessors : The preprocessors that will be included in the graph.
        additional_output_uuids: As described in the documentation for this class, the observations
            returned when calling `get_observations` only include, by default, those observations
            that are not processed by any preprocessor. If you'd like to include observations that
            would otherwise not be included, the uuids of these sensors should be included as
            a sequence of strings here.
        """
        self.device: torch.device = torch.device("cpu")

        obs_spaces: Dict[str, gym.Space] = {
            k: source_observation_spaces[k] for k in source_observation_spaces
        }

        self.preprocessors: Dict[str, Preprocessor] = {}
        for preprocessor in preprocessors:
            if isinstance(preprocessor, Builder):
                preprocessor = preprocessor()

            assert (
                preprocessor.uuid not in self.preprocessors
            ), "'{}' is duplicated preprocessor uuid".format(preprocessor.uuid)

            self.preprocessors[preprocessor.uuid] = preprocessor
            obs_spaces[preprocessor.uuid] = preprocessor.observation_space

        g = nx.DiGraph()
        for k in obs_spaces:
            g.add_node(k)
        for k in self.preprocessors:
            for j in self.preprocessors[k].input_uuids:
                g.add_edge(j, k)

        assert nx.is_directed_acyclic_graph(
            g
        ), "preprocessors do not form a direct acyclic graph"

        self.observation_spaces = SpaceDict(
            spaces={
                uuid: obs_spaces[uuid]
                for uuid in obs_spaces
                if uuid in additional_output_uuids or g.out_degree(uuid) == 0
            }
        )

        # ensure dependencies are precomputed
        self.compute_order = [n for n in nx.dfs_postorder_nodes(g)]

    def get(self, uuid: str) -> Preprocessor:
        """Return preprocessor with the given `uuid`.

        # Parameters

        uuid : The unique id of the preprocessor.

        # Returns

        The preprocessor with unique id `uuid`.
        """
        return self.preprocessors[uuid]

    def to(self, device: torch.device) -> "SensorPreprocessorGraph":
        for k, v in self.preprocessors.items():
            self.preprocessors[k] = v.to(device)
        self.device = device
        return self

    def get_observations(
        self, obs: Dict[str, Any], *args: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        """Get processed observations.

        # Returns

        Collect observations processed from all sensors and return them packaged inside a Dict.
        """

        for uuid in self.compute_order:
            if uuid not in obs:
                obs[uuid] = self.preprocessors[uuid].process(obs)

        return {uuid: obs[uuid] for uuid in self.observation_spaces}


class PreprocessorGraph(SensorPreprocessorGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise DeprecationWarning(
            "`PreprocessorGraph` has been deprecated, use `SensorPreprocessorGraph` instead."
        )


class ObservationSet:
    def __init__(self, *args, **kwargs) -> None:
        raise DeprecationWarning(
            "`ObservationSet` has been deprecated. Use `SensorPreprocessorGraph` instead."
        )


class ResNetEmbedder:
    def __init__(self, *args, **kwargs) -> None:
        raise DeprecationWarning(
            "`allenact.base_abstractions.preprocessor.ResNetEmbedder` has been moved"
            " to `allenact.embodied_ai.preprocessors.resnet.ResNetEmbedder`."
        )


class ResNetPreprocessor:
    def __init__(self, *args, **kwargs) -> None:
        raise DeprecationWarning(
            "`allenact.base_abstractions.preprocessor.ResNetPreprocessor` has been moved"
            " to `allenact.embodied_ai.preprocessors.resnet.ResNetPreprocessor`."
        )
