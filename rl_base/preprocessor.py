import abc
from collections import OrderedDict
from typing import Dict, Any, List, Union

import gym
import networkx as nx
import torch
from gym.spaces import Dict as SpaceDict

from rl_base.sensor import Sensor, SensorSuite
from utils.experiment_utils import Builder
from utils.misc_utils import prepare_locals_for_super


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


class PreprocessorGraph:
    """Represents a graph of preprocessors, with each preprocessor being
    identified through a universally unique id.

    # Attributes

    preprocessors : List containing preprocessors with required input uuids, output uuid of each
        sensor must be unique.
    """

    preprocessors: Dict[str, Preprocessor]
    observation_spaces: SpaceDict

    def __init__(
        self, preprocessors: List[Union[Preprocessor, Builder[Preprocessor]]],
    ) -> None:
        """Initializer.

        # Parameters

        preprocessors : The preprocessors that will be included in the graph.
        """
        self.preprocessors: Dict[str, Preprocessor] = OrderedDict()
        spaces: OrderedDict[str, gym.Space] = OrderedDict()
        for preprocessor in preprocessors:
            if isinstance(preprocessor, Builder):
                preprocessor = preprocessor()

            assert (
                preprocessor.uuid not in self.preprocessors
            ), "'{}' is duplicated preprocessor uuid".format(preprocessor.uuid)
            self.preprocessors[preprocessor.uuid] = preprocessor
            spaces[preprocessor.uuid] = preprocessor.observation_space
        self.observation_spaces = SpaceDict(spaces=spaces)

        g = nx.DiGraph()
        for k in self.preprocessors:
            g.add_node(k)
        for k in self.preprocessors:
            for j in self.preprocessors[k].input_uuids:
                if j not in g:
                    g.add_node(j)
                g.add_edge(k, j)
        assert nx.is_directed_acyclic_graph(
            g
        ), "preprocessors do not form a direct acyclic graph"

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

    def to(self, device: torch.device) -> "PreprocessorGraph":
        for k, v in self.preprocessors.items():
            self.preprocessors[k] = v.to(device)
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

        return obs


class ObservationSet:
    """Represents a list of source_ids, corresponding to sensors and
    preprocessors, with each source being identified through a unique id.

    # Attributes

    source_ids : List containing sensor and preprocessor ids to be consumed by agents. Each source uuid must be unique.
    graph : Computation graph for all preprocessors.
    observation_spaces : Observation spaces of all output sources.
    device : Device where the PreprocessorGraph is executed.
    """

    source_ids: List[str]
    graph: PreprocessorGraph
    observation_spaces: SpaceDict
    device: torch.device = torch.device("cpu")

    def __init__(
        self,
        source_ids: List[str],
        all_preprocessors: List[Union[Preprocessor, Builder[Preprocessor]]],
        all_sensors: List[Sensor],
    ) -> None:
        """Initializer.

        # Parameters

        source_ids : The sensors and preprocessors that will be included in the set.
        all_preprocessors : The entire list of preprocessors to be executed.
        all_sensors : The entire list of sensors.
        """

        self.graph = PreprocessorGraph(all_preprocessors)

        self.source_ids = source_ids
        assert len(set(self.source_ids)) == len(
            self.source_ids
        ), "No duplicated uuids allowed in source_ids"

        sensor_spaces = SensorSuite(all_sensors).observation_spaces
        preprocessor_spaces = self.graph.observation_spaces
        spaces: OrderedDict[str, gym.Space] = OrderedDict()
        for uuid in self.source_ids:
            assert (
                uuid in sensor_spaces.spaces or uuid in preprocessor_spaces.spaces
            ), "uuid {} missing from sensor suite and preprocessor graph".format(uuid)
            if uuid in sensor_spaces.spaces:
                spaces[uuid] = sensor_spaces[uuid]
            else:
                spaces[uuid] = preprocessor_spaces[uuid]
        self.observation_spaces = SpaceDict(spaces=spaces)

    def get(self, uuid: str) -> Preprocessor:
        """Return preprocessor with the given `uuid`.

        # Parameters

        uuid : The unique id of the preprocessor.

        # Returns

        The preprocessor with unique id `uuid`.
        """
        return self.graph.get(uuid)

    def to(self, device: torch.device) -> "ObservationSet":
        self.graph = self.graph.to(device)
        self.device = device
        return self

    def get_observations(
        self, obs: Dict[str, Any], *args: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        """Get all observations within a dictionary.

        # Returns

        Collect observations from all sources and return them packaged inside a Dict.
        """
        obs = self.graph.get_observations(obs)
        return OrderedDict([(k, obs[k]) for k in self.source_ids])
