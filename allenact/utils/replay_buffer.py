import torch
from tensordict import TensorDict
from collections import defaultdict


class StorageAdapter:
    """Adapter to convert your RolloutBlockStorage data into a format compatible with torchrl."""

    def __init__(self, rollout_storage, device):
        self.rollout_storage = rollout_storage
        self.device = device

    def to_tensordict(self, batch_size):
        """Convert storage data to a fixed-length TensorDict."""
        # Sample or pad the episode to match window_size
        episode_length = self.rollout_storage.step
        start_idx, end_idx = 0, episode_length

        # Gather episode data from rollout storage
        observations = [
            self.rollout_storage.pick_observation_step(i) for i in range(start_idx, end_idx)
        ]
        prev_actions = [
            self.rollout_storage.pick_prev_actions_step(i) for i in range(start_idx, end_idx)
        ]

        # Convert to TensorDict
        data = {
            "observations": batchify_tensordicts(observations, self.device, swap_dim_0=0, swap_dim_1=1),
            "actions": self.rollout_storage.actions.to(self.device).movedim(0, 1),
            "prev_actions": torch.cat(prev_actions).to(self.device).movedim(0, 1),
            "masks": self.rollout_storage.masks[:-1].to(self.device).movedim(0, 1),
            "rewards": self.rollout_storage.rewards.to(self.device).movedim(0, 1),
            "returns": self.rollout_storage.returns[:-1].to(self.device).movedim(0, 1),
            "values": self.rollout_storage.value_preds[:-1].to(self.device).movedim(0, 1),
            "old_action_log_probs": self.rollout_storage.action_log_probs.to(self.device).movedim(0, 1),
            "adv_targ": self.rollout_storage._advantages.to(self.device).movedim(0, 1),
            "norm_adv_targ": self.rollout_storage._normalized_advantages.to(self.device).movedim(0, 1),
        }
        return TensorDict(data, batch_size=batch_size)


def batchify_tensordicts(tensordict_list, device=None, unsqueeze_dim=None, unsqueeze_first=False, cat_dim=0,
                         swap_dim_0=None, swap_dim_1=None):
    """
    Convert a list of TensorDicts into a batched format, recursively handling nested structures,
    and applying transformations such as moving to a device, unsqueezing, concatenating, and swapping dimensions.

    Args:
        tensordict_list (list of TensorDict): List of TensorDicts where each TensorDict represents an episode.
        device (torch.device or None): Desired device to move the tensors to. If None, no device change is done.
        unsqueeze_dim (int or None): Dimension along which to unsqueeze tensors. If None, no unsqueezing is done.
        unsqueeze_first (bool): Whether to unsqueeze tensors before concatenation.
        cat_dim (int): Dimension along which to concatenate tensors. Default is 0.
        swap_dim_0 (int or None): The first dimension to swap.
        swap_dim_1 (int or None): The second dimension to swap.

    Returns:
        dict: A dictionary with keys 'observations', 'actions', 'rewards', and 'masks',
              where each value is a tensor or a dictionary of tensors containing batch data.
    """
    # Initialize dictionary to collect batched data
    batched_data = defaultdict(list)

    # Process each TensorDict in the list
    for td in tensordict_list:
        for key, value in td.items():
            if isinstance(value, (dict, TensorDict)):
                # Append nested structures to be processed later in a single recursive call
                batched_data[key].append(value)
            else:
                # Process individual tensor
                if device is not None:
                    value = value.to(device)
                # Apply unsqueeze before collecting data if `unsqueeze_first` is True
                if unsqueeze_dim is not None and unsqueeze_first:
                    value = value.unsqueeze(unsqueeze_dim)
                batched_data[key].append(value)

    # Prepare the final batched data dictionary
    final_batched_data = {}

    # Process each key in the batched data
    for key, value_list in batched_data.items():
        if isinstance(value_list[0], (dict, TensorDict)):
            # Recursively handle nested structures in a single recursive call
            final_batched_data[key] = batchify_tensordicts(
                value_list, device, unsqueeze_dim, unsqueeze_first, cat_dim, swap_dim_0, swap_dim_1
            )
        else:
            # Concatenate along the specified dimension (cat_dim)
            concatenated_tensor = torch.cat(value_list, dim=cat_dim)

            # Apply unsqueeze after concatenation if needed and `unsqueeze_first` is False
            if unsqueeze_dim is not None and not unsqueeze_first:
                concatenated_tensor = concatenated_tensor.unsqueeze(unsqueeze_dim)

            # Swap dimensions if needed
            if swap_dim_0 is not None and swap_dim_1 is not None:
                concatenated_tensor = concatenated_tensor.movedim(swap_dim_0, swap_dim_1)

            final_batched_data[key] = concatenated_tensor

    return final_batched_data


def swap_and_squeeze_batched_data(batched_data, dim0=0, dim1=1, squeeze_dim=None):
    """
    Swap dimensions dim0 and dim1 for all tensors inside the batched data and squeeze a specified dimension.

    Args:
        batched_data (dict): The batched data dictionary containing tensors or nested dictionaries of tensors.
        dim0 (int): The first dimension to swap.
        dim1 (int): The second dimension to swap.
        squeeze_dim (int or None): The dimension to squeeze. If None, no squeezing is done.

    Returns:
        dict: A new dictionary with all tensors having swapped dimensions and the specified dimension squeezed.
    """
    swapped_data = {}

    for key, value in batched_data.items():
        if isinstance(value, dict):
            # If the value is a nested dictionary (like observations), handle each sub-key separately
            swapped_data[key] = {}
            for sub_key, sub_value in value.items():
                # Swap dimensions
                tensor = sub_value.transpose(dim0, dim1).contiguous()
                # Squeeze if needed
                if squeeze_dim is not None and tensor.size(squeeze_dim) == 1:
                    tensor = tensor.squeeze(squeeze_dim)
                swapped_data[key][sub_key] = tensor
        else:
            # For direct tensors, swap the dimensions
            tensor = value.transpose(dim0, dim1).contiguous()
            # Squeeze if needed
            if squeeze_dim is not None and tensor.size(squeeze_dim) == 1:
                tensor = tensor.squeeze(squeeze_dim)
            swapped_data[key] = tensor

    return swapped_data
