import gin
import numpy as np
from alf.tensor_specs import TensorSpec
@gin.configurable
def compute_discount_from_horizon(T, num_episodes=1):
    r"""For an MDP with an infinite horizon and a time limit :math:`T` for each
    episode, compute the discount factor as :math:`1-\frac{1}{T}`. This function
    is mainly used by gin files.

    Args:
        T (int): the length of each episode
        num_episodes (int): when computing the discount, treat this many epsiodes
            as if they are just one episode

    Returns:
        float: the discount factor :math:`\gamma` of the MDP
    """
    return 1 - 1 / float(T * num_episodes)

def numel(spec):
	"""Returns the number of elements in TensorSpec"""
	return int(np.prod(spec.shape))

def flatten(spec):
    """Create a new TensorSpec so that the shape is flattened."""
    return TensorSpec(shape=(numel(spec), ), dtype=spec.dtype)
