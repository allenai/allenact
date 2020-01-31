# Defining a new model

All actor-critic models must implement the interface described by the
[ActorCriticModel class](/api/onpolicy_sync/policy#actorcriticmodel). This interface includes two methods that need to be 
implemented:

* `recurrent_hidden_state_size`, returning the size of the model's hidden state; and 
* `forward`, returning an [ActorCriticOutput](/api/rl_base/common#actorcriticoutput) given the current observation,
hidden state and previous actions.

For convenience, we have already defined a [recurrent module](/api/models/basic_models#RNNStateEncoder) and
[a simple CNN module](/api/basic_models#SimpleCNN) that will be used in this example.

As an example, let's build an object navigation agent.

```python
class ObjectNavActorCritic(ActorCriticModel[CategoricalDistr]):
        def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        goal_sensor_uuid: str,
        hidden_size=512,
        object_type_embedding_dim=8,
    ):
        super().__init__(
            action_space=action_space,
            observation_space=observation_space
        )

        self.goal_sensor_uuid = goal_sensor_uuid
        self._n_object_types =
            self.observation_space.spaces[self.goal_sensor_uuid].n
        self._hidden_size = hidden_size
        self.object_type_embedding_size = object_type_embedding_dim

        self.visual_encoder = SimpleCNN(
            self.observation_space,
            hidden_size
        )

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size)
            + object_type_embedding_dim,
            self.recurrent_hidden_state_size,
        )

        self.actor = LinearActorHead(
            self.recurrent_hidden_state_size,
            action_space.n
        )
        self.critic = LinearCriticHead(
            self.recurrent_hidden_state_size
        )

        self.object_type_embedding = nn.Embedding(
            num_embeddings=self._n_object_types,
            embedding_dim=object_type_embedding_dim,
        )

        self.train()

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        target_encoding = self.object_type_embedding(
            observations[self.goal_sensor_uuid].to(torch.int64)
        )
        x = [target_encoding]

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            x = [perception_embed] + x

        x = torch.cat(x, dim=1)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        return (
            ActorCriticOutput(
                distributions=self.actor(x), values=self.critic(x), extras={}
            ),
            rnn_hidden_states,
        )
     ...
```

## Engine requirements

Apart from the interface expected by all actor-critic models, we also need to provide a utility function to allow
the engine to properly initalize the rollouts storage, `num_recurrent_layers`:

```python
class ObjectNavActorCritic(ActorCriticModel[CategoricalDistr]):
    ...
    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers
```
