import flax.linen as nn
import functools
from typing import List, Callable
import jax
import jax.numpy as jnp
from jax import jit

class MLPController(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) neural network controller for the cart-pole system.
    
    Attributes:
        features (List[int]): List of integers defining the size of each layer.
    
    Input:
        x: System state vector [x, x_dot, theta, theta_dot]
    
    Output:
        Control action (force applied to cart)
    """
    features: List[int]

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass of the MLP controller.

        Args:
            x (jnp.ndarray): Input state vector of shape (4,)

        Returns:
            jnp.ndarray: Control action of shape (1,)
        """
        # Hidden layers with ReLU activation
        for feat in self.features[:-1]:
            x = nn.Dense(
                feat,
                kernel_init=nn.initializers.glorot_uniform(),
                bias_init=nn.initializers.zeros
            )(x)
            x = nn.relu(x)
        
        # Output layer (linear activation)
        x = nn.Dense(
            self.features[-1],
            kernel_init=nn.initializers.glorot_uniform(),
            bias_init=nn.initializers.zeros
        )(x)
        
        return x

def create_controller(mlp_controller: MLPController) -> Callable:
    """
    Creates a controller function that wraps the MLPController for easier use.
    
    Args:
        mlp_controller (MLPController): Instance of the MLPController class

    Returns:
        Callable: A function that takes parameters and state as input and returns control action
    
    Example usage:
        # Initialize controller
        controller = MLPController(features=[64, 32, 1])
        variables = controller.init(jax.random.PRNGKey(0), jnp.zeros(4))
        params = variables['params']

        # Create controller function
        controller_fn = create_controller(controller)
        
        # Get control action
        state = jnp.array([0.0, 0.0, np.pi, 0.0])
        action = controller_fn(params, state)
    """
    @functools.partial(jit, static_argnums=(0,))
    def controller_fn(mlp_apply_fn: Callable, params: dict, state: jnp.ndarray) -> jnp.ndarray:
        """
        Computes control action for given parameters and state.

        Args:
            mlp_apply_fn: The apply function from the MLPController
            params: Neural network parameters
            state: Current system state

        Returns:
            jnp.ndarray: Control action
        """
        return mlp_apply_fn({'params': params}, state)
    
    return functools.partial(controller_fn, mlp_controller.apply)


def create_example_controller(state_dim: int = 4, 
                            hidden_layers: List[int] = [64, 32],
                            action_dim: int = 1,
                            seed: int = 0) -> tuple:
    """
    Creates an example MLPController with specified architecture.

    Args:
        state_dim (int): Dimension of state input (default: 4 for cart-pole)
        hidden_layers (List[int]): List of hidden layer sizes
        action_dim (int): Dimension of action output (default: 1)
        seed (int): Random seed for initialization

    Returns:
        tuple: (controller, params, controller_fn)
            - controller: MLPController instance
            - params: Initialized parameters
            - controller_fn: Compiled controller function
    """
    # Create layer architecture (input -> hidden -> output)
    features = hidden_layers + [action_dim]
    
    # Initialize controller
    controller = MLPController(features=features)
    
    # Initialize parameters with dummy input
    key = jax.random.PRNGKey(seed)
    dummy_state = jnp.zeros(state_dim)
    variables = controller.init(key, dummy_state)
    params = variables['params']
    
    # Create controller function
    controller_fn = create_controller(controller)
    
    return controller, params, controller_fn