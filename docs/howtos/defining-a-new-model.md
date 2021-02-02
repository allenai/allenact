# Defining a new model

All actor-critic models must implement the interface described by the
[ActorCriticModel class](/api/allenact/algorithms/onpolicy_sync/policy/#actorcriticmodel). This interface includes two methods that need to be 
implemented:

* `recurrent_memory_specification`, returning a description of the model's recurrent memory; and 
* `forward`, returning an [ActorCriticOutput](/api/allenact/base_abstractions/misc/#actorcriticoutput) given the current observation,
hidden state and previous actions.

For convenience, we provide a [recurrent network module](/api/allenact/embodiedai/models/basic_models/#rnnstateencoder) and
[a simple CNN module](/api/allenact/embodiedai/models/basic_models/#simplecnn) from the Habitat baseline navigation
models, that will be used in this example.

### Actor-critic model interface

As an example, let's build an object navigation agent.

```python
class ObjectNavBaselineActorCritic(ActorCriticModel[CategoricalDistr]):
    """Baseline recurrent actor critic model for object-navigation.

    # Attributes
    action_space : The space of actions available to the agent. Currently only discrete
        actions are allowed (so this space will always be of type `gym.spaces.Discrete`).
    observation_space : The observation space expected by the agent. This observation space
        should include (optionally) 'rgb' images and 'depth' images and is required to
        have a component corresponding to the goal `goal_sensor_uuid`.
    goal_sensor_uuid : The uuid of the sensor of the goal object. See `GoalObjectTypeThorSensor`
        as an example of such a sensor.
    hidden_size : The hidden size of the GRU RNN.
    object_type_embedding_dim: The dimensionality of the embedding corresponding to the goal
        object type.
    """

    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        goal_sensor_uuid: str,
        rgb_uuid: Optional[str],
        depth_uuid: Optional[str],
        hidden_size=512,
        object_type_embedding_dim=8,
        trainable_masked_hidden_state: bool = False,
        num_rnn_layers=1,
        rnn_type="GRU",
    ):
        """Initializer.

        See class documentation for parameter definitions.
        """
        super().__init__(action_space=action_space, observation_space=observation_space)

        self.goal_sensor_uuid = goal_sensor_uuid
        self._n_object_types = self.observation_space.spaces[self.goal_sensor_uuid].n
        self._hidden_size = hidden_size
        self.object_type_embedding_size = object_type_embedding_dim

        self.visual_encoder = SimpleCNN(
            observation_space=self.observation_space,
            output_size=self._hidden_size,
            rgb_uuid=rgb_uuid,
            depth_uuid=depth_uuid,
        )

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self._hidden_size) + object_type_embedding_dim,
            self._hidden_size,
            trainable_masked_hidden_state=trainable_masked_hidden_state,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )

        self.actor = LinearActorHead(self._hidden_size, action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)

        self.object_type_embedding = nn.Embedding(
            num_embeddings=self._n_object_types,
            embedding_dim=object_type_embedding_dim,
        )

        self.train()

    @property
    def recurrent_hidden_state_size(self) -> int:
        """The recurrent hidden state size of the model."""
        return self._hidden_size

    @property
    def is_blind(self) -> bool:
        """True if the model is blind (e.g. neither 'depth' or 'rgb' is an
        input observation type)."""
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self) -> int:
        """Number of recurrent hidden layers."""
        return self.state_encoder.num_recurrent_layers

    def _recurrent_memory_specification(self):
        return dict(
            rnn=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        )

    def get_object_type_encoding(
        self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        # noinspection PyTypeChecker
        return self.object_type_embedding(  # type:ignore
            observations[self.goal_sensor_uuid].to(torch.int64)
        )

    def forward(  # type:ignore
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        """Processes input batched observations to produce new actor and critic
        values. Processes input batched observations (along with prior hidden
        states, previous actions, and masks denoting which recurrent hidden
        states should be masked) and returns an `ActorCriticOutput` object
        containing the model's policy (distribution over actions) and
        evaluation of the current state (value).

        # Parameters
        observations : Batched input observations.
        memory : `Memory` containing the hidden states from initial timepoints.
        prev_actions : Tensor of previous actions taken.
        masks : Masks applied to hidden states. See `RNNStateEncoder`.

        # Returns
        Tuple of the `ActorCriticOutput` and recurrent hidden state.
        """
        target_encoding = self.get_object_type_encoding(
            cast(Dict[str, torch.FloatTensor], observations)
        )
        x = [target_encoding]

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            x = [perception_embed] + x

        x_cat = torch.cat(x, dim=-1)  # type: ignore
        x_out, rnn_hidden_states = self.state_encoder(
            x_cat, memory.tensor("rnn"), masks
        )

        return (
            ActorCriticOutput(
                distributions=self.actor(x_out), values=self.critic(x_out), extras={}
            ),
            memory.set_tensor("rnn", rnn_hidden_states),
        )
```
