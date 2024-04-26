import torch
from diff_apf_pytorch.functional import parametric_eq
from typing import Dict, List


def denormalize(norm_val, max_val, min_val):
    return (norm_val * (max_val - min_val)) + min_val


def normalize(val, min_val, max_val):
    return (val - min_val) / (max_val - min_val)


class Processor:
    def __init__(self):
        pass

    def process_normalized(self, x: torch.Tensor, param_tensor: torch.Tensor):
        """Run the processor using normalized parameters on (0,1) in a tensor.

        This function assumes that the parameters in the tensor are in the same
        order as the parameter defined in the processor.

        Args:
            x (torch.Tensor): Input audio tensor (batch, channels, samples)
            param_tensor (torch.Tensor): Tensor of parameters on (0,1) (batch, num_params)

        Returns:
            torch.Tensor: Output audio tensor
        """
        # extract parameters from tensor
        param_dict = self.extract_param_dict(param_tensor)

        # denormalize parameters to full range
        denorm_param_dict = self.denormalize_param_dict(param_dict)

        # now process audio with denormalized parameters
        y = self.process_fn(
            x,
            self.sample_rate,
            **denorm_param_dict,
        )

        return y

    def process(self, x: torch.Tensor, *args):
        return self.process_fn(x, *args)

    def extract_param_dict(self, param_tensor: torch.Tensor):
        # check the number of parameters in tensor matches the number of processor parameters
        if param_tensor.shape[1] != len(self.param_ranges):
            raise ValueError(
                f"Parameter tensor has {param_tensor.shape[1]} parameters, but processor has {len(self.param_ranges)} parameters."
            )

        # extract parameters from tensor
        param_dict = {}
        for param_idx, param_name in enumerate(self.param_ranges.keys()):
            param_dict[param_name] = param_tensor[:, param_idx]

        return param_dict

    def denormalize_param_dict(self, param_dict: dict):
        """Given parameters on (0,1) restore them to the ranges expected by the processor.

        Args:
            param_dict (dict): Dictionary of parameter tensors on (0,1).

        Returns:
            dict: Dictionary of parameter tensors on their full range.

        """
        denorm_param_dict = {}
        for param_name, param_tensor in param_dict.items():
            # check for out of range parameters
            if param_tensor.min() < 0 or param_tensor.max() > 1:
                raise ValueError(f"Parameter {param_name} of is out of range.")
            param_val_denorm = denormalize(
                param_tensor,
                self.param_ranges[param_name][1],
                self.param_ranges[param_name][0],
            )
            denorm_param_dict[param_name] = param_val_denorm
        return denorm_param_dict


class ParametricEQ(Processor):
    def __init__(
        self,
        sample_rate: int,
        min_cutoff_freq: float = 20,
        max_cutoff_freq: float = 20_000,
        min_q_factor: float = 0.1,
        max_q_factor: float = 10.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.process_fn = parametric_eq
        self.param_ranges = {
            "band0_cutoff_freq": (min_cutoff_freq, max_cutoff_freq),
            "band0_q_factor": (min_q_factor, max_q_factor),
            "band1_cutoff_freq": (min_cutoff_freq, max_cutoff_freq),
            "band1_q_factor": (min_q_factor, max_q_factor),
            "band2_cutoff_freq": (min_cutoff_freq, max_cutoff_freq),
            "band2_q_factor": (min_q_factor, max_q_factor),
            "band3_cutoff_freq": (min_cutoff_freq, max_cutoff_freq),
            "band3_q_factor": (min_q_factor, max_q_factor),
            "band4_cutoff_freq": (min_cutoff_freq, max_cutoff_freq),
            "band4_q_factor": (min_q_factor, max_q_factor),
            "band5_cutoff_freq": (min_cutoff_freq, max_cutoff_freq),
            "band5_q_factor": (min_q_factor, max_q_factor),
        }
        self.num_params = len(self.param_ranges)