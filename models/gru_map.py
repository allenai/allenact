import torch
import torch.nn as nn

from typing import Tuple, Union, Sequence


class RNNMap(nn.Module):

    def __init__(
            self,
            embedding_size=8,
            input_size=512,
            rnn_type='GRU',
            num_rnn_layers=1,
            map_width = 32,
            map_height = 32,
    ):
        super().__init__()
        self.rnn = getattr(nn, rnn_type)(
            input_size=input_size, hidden_size=embedding_size, num_layers=num_rnn_layers
        )
        self.map_width = map_width
        self.map_height = map_height

    def layer_init(self):
        """Initialize the RNN parameters in the model."""
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    @property
    def num_recurrent_layers(self) -> int:
        """The number of recurrent layers in the network."""
        return self._num_recurrent_layers * (2 if "LSTM" in self._rnn_type else 1)

    def _pack_hidden(
        self, hidden_states: Union[torch.FloatTensor, Sequence[torch.FloatTensor]]
    ) -> torch.FloatTensor:
        """Stacks hiddens states in an LSTM together (if using a GRU rather
        than an LSTM this is just the identitiy).
        # Parameters
        hidden_states : The hidden states to (possibly) stack.
        """
        if "LSTM" in self._rnn_type:
            hidden_states = cast(
                torch.FloatTensor,
                torch.cat([hidden_states[0], hidden_states[1]], dim=0),
            )

        return cast(torch.FloatTensor, hidden_states)

    def _unpack_hidden(
        self, hidden_states: torch.FloatTensor
    ) -> Union[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor]]:
        """Partial inverse of `_pack_hidden` (exact if there are 2 hidden
        layers)."""
        if "LSTM" in self._rnn_type:
            new_hidden_states = (
                hidden_states[0 : self._num_recurrent_layers],
                hidden_states[self._num_recurrent_layers :],
            )
            return cast(Tuple[torch.FloatTensor, torch.FloatTensor], new_hidden_states)
        return hidden_states

    def fold_map(self, hidden_state: torch.FloatTensor) -> torch.FloatTensor:
        """Folds GRU Map into single dimensional tensor to have the same shape as regular GRU memory"""
        return hidden_state.view(hidden_state.size(0), hidden_state.size(1), -1)

    def unfold_map(self, hidden_state: torch.FloatTensor) -> torch.FloatTensor:
        """Unfolds single dimensional tensor into the shape of the GRU Map"""
        return hidden_state.view(hidden_state.size(0), hidden_state.size(1), self.map_width, self.map_height, -1)

    def single_forward(
            self,
            x: torch.Tensor,
            memory_map: torch.FloatTensor,
            position: Tuple[int, int]
    ) -> (torch.FloatTens or, torch.FloatTensor):

        memory = memory_map[position]
        x_out, memory_out = self.rnn(x, memory)
        delta_memory = memory_out - memory

        new_memory_map = memory_map.clone()
        new_memory_map[position] += delta_memory
        return x_out, new_memory_map

    def forward(
        self,
        x: torch.FloatTensor,
        hidden_states: torch.FloatTensor,
        masks: torch.FloatTensor,
    ) -> Tuple[
        torch.FloatTensor, Union[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]
    ]:
        """Calls `seq_forward` or `single_forward` depending on the input size.
        See the above methods for more information.
        """
        if x.size(0) == hidden_states.size(1):
            return self.single_forward(x, hidden_states, masks)
        else:
            return self.seq_forward(x, hidden_states, masks)
