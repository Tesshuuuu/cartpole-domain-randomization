import jax
import jax.numpy as jnp
from jax import jit
import flax.linen as nn
import functools
from typing import List, Callable

class MLPController(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) neural network controller for the pendulum system.

    Attributes:
        features (List[int]): List of integers defining the size of each layer.
    """
    features: List[int]

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args: 
            x (jnp.ndarray): Input observation vector of shape (3,)
        Returns:
            jnp.ndarray: Control action of shape (1,)
        """
        # Hidden layers with ReLU activation
        for feature in self.features[:-1]:
            x = nn.Dense(feature)(x)
            x = nn.relu(x)
        
        # Output layer (linear activation)
        x = nn.Dense(self.features[-1])(x)

        return x
    
def create_controller(mlp_controller: MLPController) -> Callable:
    """
    Wraps the MLPController for easier use.
    """
    @functools.partial(jit, static_argnums = (0, ))
    def controller_fn(mlp_fn: Callable, nn_params: dict, obs: jnp.ndarray) -> jnp.ndarray:
        """
        Computes control action given neural network parameters and observation.
        """
        return mlp_fn(nn_params, obs)
    
    return functools.partial(controller_fn, mlp_controller)

def create_example_controller(obs_dim: int = 3, 
                              action_dim: int = 1,
                              hidden_layers: List[int] = [64, 32],
                              seed: int = 0) -> tuple:
    """
    Creates an example MLPController with specified architecture.
    Returns:
        controller: MLPController instance
        nn_params: Initialized parameters
        controller_fn: Compiled controller function
    """
    # Create layer architecture (input -> hidden -> output)
    features = hidden_layers + [action_dim]

    # Initialize controller
    controller = MLPController(features=features)

    # Initialize parameters with dummy input by calling init function
    key = jax.random.PRNGKey(seed) 
    dummy_obs = jnp.zeros(obs_dim) # dummy input
    variables = controller.init(key, dummy_obs)
    nn_params = variables['params'] # initial parameter

    controller_fn = create_controller(controller)

    return controller, nn_params, controller_fn