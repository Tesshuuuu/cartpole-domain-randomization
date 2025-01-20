from functools import partial
from typing import Tuple, Dict, Any
import jax
import jax.numpy as jnp
import jax.random as random
from jax import jit, value_and_grad, lax
import optax
from noiseless_dyn import noiseless_dyn
from mlp_controller import create_example_controller
import os
import pickle
from datetime import datetime


class CartPoleTrainer:
    """
    Trainer class for the cart-pole system using an MLP controller.
    
    This class handles the training of a neural network controller for the
    cart-pole balancing task using gradient-based optimization.
    """
    
    def __init__(self, 
                 dynamics_params: jnp.ndarray,
                 state_dim: int = 4,
                 action_dim: int = 1,
                 hidden_layers: list = [64, 32],
                 noise_std: float = 0.01,
                 seed: int = 0):
        """
        Initialize the trainer.

        Args:
            dynamics_params: Physical parameters of the cart-pole system
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_layers: Architecture of the MLP controller
            noise_std: Standard deviation of the process noise
            seed: Random seed for initialization
        """
        self.dynamics_params = dynamics_params
        self.noise_std = noise_std
        
        # Initialize controller
        self.controller, self.params, self.controller_fn = create_example_controller(
            state_dim=state_dim,
            hidden_layers=hidden_layers,
            action_dim=action_dim,
            seed=seed
        )

    @partial(jit, static_argnums=(0,))
    def dynamics(self, x: jnp.ndarray, u: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
        """Compute next state using dynamics with noise."""
        return noiseless_dyn(x, u, self.dynamics_params) + self.noise_std * w

    @partial(jit, static_argnums=(0,))
    def closed_loop_dynamics(self, 
                           state: jnp.ndarray, 
                           noise: jnp.ndarray, 
                           params: Dict) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Simulate one step of the closed-loop system."""
        # Get control action from policy
        action = self.controller_fn(params, state)
        # Update state using dynamics
        next_state = self.dynamics(state, action, noise)
        return next_state, action

    @partial(jit, static_argnums=(0,))
    def simulate_trajectory(self, 
                          initial_state: jnp.ndarray,
                          noises: jnp.ndarray,
                          params: Dict) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Simulate a complete trajectory.
        
        Args:
            initial_state: Starting state
            noises: Noise sequence for the trajectory
            params: Controller parameters

        Returns:
            Tuple of (states, actions) for the trajectory
        """
        def step_fn(state, noise):
            next_state, action = self.closed_loop_dynamics(state, noise, params)
            return next_state, (state, action)

        _, (states, actions) = jax.lax.scan(step_fn, initial_state, noises)
        return states, actions

    @partial(jit, static_argnums=(0,))
    def compute_cost(self, 
                    states: jnp.ndarray, 
                    actions: jnp.ndarray, 
                    cost_matrices: Tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        """
        Compute the LQR cost for a trajectory.
        
        Args:
            states: Trajectory states
            actions: Control actions
            cost_matrices: Tuple of (Q, R) matrices for state and action costs

        Returns:
            Total trajectory cost
        """
        Q, R = cost_matrices
        state_costs = jnp.einsum('ti,ij,tj->t', states, Q, states)
        action_costs = jnp.einsum('ti,ij,tj->t', actions, R, actions)
        return jnp.mean(state_costs + action_costs)

    @partial(jit, static_argnums=(0,))
    def loss_fn(self, 
                params: Dict, 
                initial_state: jnp.ndarray,
                noises: jnp.ndarray,
                cost_matrices: Tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        """Compute loss for a given set of parameters."""
        states, actions = self.simulate_trajectory(initial_state, noises, params)
        return self.compute_cost(states, actions, cost_matrices)
    
    def save_controller(self, params, save_dir='saved_controllers', suffix=''):
        """
        Save the trained controller parameters and configuration.
        
        Args:
            params: Trained parameters to save
            save_dir: Directory to save the controller
            suffix: Optional suffix to add to filename (e.g., 'nominal', 'robust')
        """
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Create a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save configuration and parameters
        save_data = {
            'params': params,
            'dynamics_params': self.dynamics_params,
            'noise_std': self.noise_std,
            'architecture': self.controller.features,  # Save network architecture
        }
        
        # Create filename with optional suffix
        filename = f'controller_{suffix}_{timestamp}.pkl' if suffix else f'controller_{timestamp}.pkl'
        filepath = os.path.join(save_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Controller saved to: {filepath}")
        return filepath

    @classmethod
    def load_controller(cls, filepath):
        """
        Load a saved controller.
        
        Args:
            filepath: Path to the saved controller file
            
        Returns:
            trainer: CartPoleTrainer instance
            params: Loaded parameters
        """
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        # Create trainer with saved configuration
        trainer = cls(
            dynamics_params=save_data['dynamics_params'],
            hidden_layers=save_data['architecture'][:-1],  # Remove output layer from hidden layers
            noise_std=save_data['noise_std']
        )
        
        return trainer, save_data['params']


    def train(self, 
             cost_matrices: Tuple[jnp.ndarray, jnp.ndarray],
             num_iterations: int = 1000,
             T: int = 100,
             initial_learning_rate: float = 0.01,
             reg_strength: float = 0.01,
             seed: int = 0) -> Tuple[Dict, jnp.ndarray]:
        """
        Train the controller.
        
        Args:
            cost_matrices: Tuple of (Q, R) cost matrices
            num_iterations: Number of training iterations
            T: Length of each trajectory
            initial_learning_rate: Initial learning rate
            reg_strength: L2 regularization strength
            seed: Random seed

        Returns:
            Tuple of (optimized parameters, training losses)
        """
        # Learning rate schedule
        schedule_fn = lambda step: initial_learning_rate / jnp.sqrt(step + 1)
        optimizer = optax.adam(learning_rate=schedule_fn)
        opt_state = optimizer.init(self.params)

        # Initial state
        initial_state = jnp.zeros(4)  # Can be modified for different initial conditions
        
        # Prepare for training
        key = random.PRNGKey(seed)
        losses = []

        # Training loop
        for i in range(num_iterations):
            # Generate noise sequence
            key, subkey = random.split(key)
            noises = random.normal(subkey, shape=(T, 4))

            # Compute loss and gradients with regularization
            loss, grads = value_and_grad(lambda p: self.loss_fn(
                p, initial_state, noises, cost_matrices) + 
                reg_strength * self.compute_l2_regularization(p))(self.params)
            
            # Update parameters
            updates, opt_state = optimizer.update(grads, opt_state, self.params)
            self.params = optax.apply_updates(self.params, updates)
            
            losses.append(loss)
            
            # Print progress
            if (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{num_iterations}, Loss: {loss:.4f}")

        return self.params, jnp.array(losses)
    
    def train_DR(self, 
         cost_matrices: Tuple[jnp.ndarray, jnp.ndarray],
         num_iterations: int = 1000,
         T: int = 100,
         initial_learning_rate: float = 0.01,
         scale_ellipsoid: float = 0.1,
         reg_strength: float = 0.01,
         seed: int = 0) -> Tuple[Dict, jnp.ndarray]:
        """
        Train the controller using Distributionally Robust optimization.
        
        Args:
            cost_matrices: Tuple of (Q, R) cost matrices
            num_iterations: Number of training iterations
            T: Length of each trajectory
            initial_learning_rate: Initial learning rate
            scale_ellipsoid: Scale factor for parameter uncertainty
            reg_strength: L2 regularization strength
            seed: Random seed

        Returns:
            Tuple of (optimized parameters, training losses)
        """
        # Learning rate schedule
        schedule_fn = lambda step: initial_learning_rate / jnp.sqrt(step + 1)
        optimizer = optax.adam(learning_rate=schedule_fn)
        opt_state = optimizer.init(self.params)

        # Initial state
        initial_state = jnp.zeros(4)
        
        # Define parameter uncertainty bounds
        delta_phi = scale_ellipsoid * jnp.abs(self.dynamics_params)
        
        # Prepare for training
        key = random.PRNGKey(seed)
        losses = []

        # Training loop
        for i in range(num_iterations):
            key, noise_key, param_key = random.split(key, 3)
            noises = random.normal(noise_key, shape=(T, 4))
            
            # Sample perturbed dynamics parameters
            dynamics_params_sample = self.dynamics_params + random.uniform(
                param_key,
                shape=self.dynamics_params.shape,
                minval=-delta_phi,
                maxval=delta_phi
            )
            
            # Compute loss and gradients with sampled parameters and regularization
            loss, grads = value_and_grad(lambda p: self.loss_fn_DR(
                p, initial_state, noises, dynamics_params_sample, cost_matrices) + 
                reg_strength * self.compute_l2_regularization(p))(self.params)
            
            # Update parameters
            updates, opt_state = optimizer.update(grads, opt_state, self.params)
            self.params = optax.apply_updates(self.params, updates)
            
            losses.append(loss)
            
            if (i + 1) % 100 == 0:
                print(f"DR Iteration {i+1}/{num_iterations}, Loss: {float(loss):.4f}")

        return self.params, jnp.array(losses)

    @partial(jit, static_argnums=(0,))
    def compute_l2_regularization(self, params: Dict) -> jnp.ndarray:
        """Compute L2 regularization term for network parameters."""
        reg_term = 0.0
        for layer_params in params.values():
            for param in layer_params.values():
                reg_term += jnp.sum(param ** 2)
        return reg_term

    @partial(jit, static_argnums=(0,))
    def loss_fn_DR(self,
                 params: Dict,
                 initial_state: jnp.ndarray,
                 noises: jnp.ndarray,
                 dynamics_params_sample: jnp.ndarray,
                 cost_matrices: Tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        """Compute loss for DR optimization using sampled dynamics parameters."""
        # Compute trajectory and cost with sampled parameters
        states, actions = self.simulate_trajectory_with_params(
            initial_state, 
            noises, 
            params, 
            dynamics_params_sample
        )
        loss = self.compute_cost(states, actions, cost_matrices)
        return loss

    def simulate_trajectory_with_params(self,
                                    initial_state: jnp.ndarray,
                                    noises: jnp.ndarray,
                                    params: Dict,
                                    dynamics_params: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Simulate trajectory with specified dynamics parameters."""
        def step(state, noise):
            action = self.controller_fn(params, state)
            next_state = noiseless_dyn(state, action, dynamics_params) + self.noise_std * noise
            return next_state, (state, action)
        
        final_state, (states, actions) = lax.scan(step, initial_state, noises)
        return states, actions


def create_default_cost_matrices(state_dim: int = 4, action_dim: int = 1) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Create default cost matrices for the cart-pole system."""
    Q = jnp.diag(jnp.array([10.0, 5.0, 10.0, 1.0]))  # State cost
    R = 0.1 * jnp.eye(action_dim)  # Action cost
    return Q, R

