from functools import partial
from typing import Tuple, Dict, Any
import jax
import jax.numpy as jnp
import jax.random as random
from jax import jit, value_and_grad, lax
import optax
import os
import pickle
from datetime import datetime
import matplotlib.pyplot as plt

from noiseless_dyn import noiseless_dyn
from mlp_controller import create_example_controller
from simulate import simulate_controller


class PendulumTrainer:
    """
    Trainer class for the pendulum system using an MLP controller.
    """

    def __init__(self, 
                 phi: jnp.ndarray,
                 FI: jnp.ndarray = None,
                 obs_dim: int = 3,
                 action_dim: int = 1,
                 hidden_layers: list[int] = [64, 32],
                 noise_std: float = 0.01,
                 seed: int = 0):
        
        self.phi = phi
        self.FI = FI if FI is not None else jnp.eye(len(phi))
        self.noise_std = noise_std

        self.controller, self.nn_params, self.controller_fn = create_example_controller(
            obs_dim=obs_dim, 
            action_dim=action_dim,
            hidden_layers=hidden_layers,
            seed=seed
        )

    @partial(jit, static_argnums=(0,))
    def closed_loop_dynamics(self, state: jnp.ndarray, noise: jnp.ndarray, nn_params: Dict) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute the closed loop dynamics."""
        # Use apply directly instead of controller_fn
        # compute observation given state
        obs = jnp.array([jnp.cos(state[0]), jnp.sin(state[0]), state[1]])
        action = self.controller.apply({'params': nn_params}, obs)
        next_state = noiseless_dyn(state, action, self.phi) + self.noise_std * noise
        return next_state, action

    @partial(jit, static_argnums=(0,))
    def loss_fn(self, 
                initial_state: jnp.ndarray,
                noises: jnp.ndarray,
                cost_matrices: Tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        """
        Given initial_state, random actions and noises, simulate the trajectory and compute the loss.
        """
        def step_fn(state, noise):
            next_state, action = self.closed_loop_dynamics(state, noise, self.nn_params)
            return next_state, (state, action)

        # Combine actions and noises into a single structure    
        _, (states_traj, actions_traj) = lax.scan(step_fn, initial_state, noises)

        # compute the cost
        Q, R = cost_matrices
        angle_error = jnp.mod(states_traj[:, 0] + jnp.pi, 2*jnp.pi) - jnp.pi
        velocity_error = states_traj[:, 1]

        cost = Q[0,0] * angle_error**2 + Q[1,1] * velocity_error**2 
        + jnp.einsum('ti,ij,tj->t', actions_traj, R, actions_traj)

        return jnp.mean(cost)
    
    def train(self, 
              cost_matrices: Tuple[jnp.ndarray, jnp.ndarray],
              num_iterations: int = 1000,
              T: int = 100,
              initial_learning_rate: float = 0.01,
              reg_strength: float = 1e-3,
              seed: int = 0) -> Tuple[Dict, jnp.ndarray]:
        
        schedule_fn = lambda step: initial_learning_rate / jnp.sqrt(step + 1)
        optimizer = optax.adam(learning_rate=schedule_fn)
        opt_state = optimizer.init(self.nn_params)

        key = random.PRNGKey(seed)
        losses = []
        
        state_dim = cost_matrices[0].shape[0]
        for i in range(num_iterations):

            key, noise_key = random.split(key, 2)
            noises = random.normal(noise_key, shape=(T, state_dim))
       
            # Fix: Generate random initial state properly
            key, subkey = random.split(key)
            initial_state = jnp.array([random.uniform(subkey, minval=-jnp.pi/2, maxval=jnp.pi/2), random.uniform(subkey, minval=-1, maxval=1)])

            # calculate the regularization term
            reg_loss = self.compute_l2_regularization(self.nn_params) * reg_strength

            loss_for_params = lambda params: self.loss_fn(initial_state, noises, cost_matrices) + reg_loss
            loss, grads = value_and_grad(loss_for_params)(self.nn_params)

            updates, opt_state = optimizer.update(grads, opt_state, self.nn_params)
            self.nn_params = optax.apply_updates(self.nn_params, updates)

            losses.append(loss)

            if (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{num_iterations}, Loss: {loss:.4f}")



        return self.nn_params, jnp.array(losses)

    @partial(jit, static_argnums=(0,))
    def compute_l2_regularization(self, params: Dict) -> jnp.ndarray:
        """Compute L2 regularization term for network parameters."""
        reg_term = 0.0
        for layer_params in params.values():
            for param in layer_params.values():
                reg_term += jnp.sum(param ** 2)
        return reg_term

    def save_controller(self, params, save_dir='saved_controllers', suffix=''):
        """Save the trained controller parameters and configuration."""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        save_data = {
            'params': params,
            'phi': self.phi,
            'FI': self.FI,
            'noise_std': self.noise_std,
            'architecture': self.controller.features,
        }
        
        filename = f'pendulum_controller_{suffix}_{timestamp}.pkl' if suffix else f'pendulum_controller_{timestamp}.pkl'
        filepath = os.path.join(save_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Controller saved to: {filepath}")
        return filepath

    @classmethod
    def load_controller(cls, filepath):
        """Load a saved controller."""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        trainer = cls(
            phi=save_data['phi'],
            FI=save_data.get('FI', None),
            hidden_layers=save_data['architecture'][:-1],
            noise_std=save_data['noise_std']
        )
        
        return trainer, save_data['params']
    
def run_experiment(
        phi, 
        num_iterations=2000, 
        episode_length=200, 
        hidden_layers = [64, 32], 
        noise_std=0.01,
        reg_strength=1e-3
    ):
    """
    Run the experiment to train the controller and simulate the pendulum system.
    """
    trainer = PendulumTrainer(phi, hidden_layers=hidden_layers, noise_std=noise_std)

    # Define the cost matrices
    Q = jnp.zeros((2, 2))
    Q = Q.at[0, 0].set(4.0)
    Q = Q.at[1, 1].set(0.1)
    R = 0.001 * jnp.eye(1)
    cost_matrices = (Q, R)

    print("=== Training controller ===")
    trained_params, losses = trainer.train(
        cost_matrices=cost_matrices,
        num_iterations=num_iterations,
        T=episode_length,
        initial_learning_rate=0.01, 
        reg_strength=reg_strength
    )

    # Plot the loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.grid(True)
    plt.savefig('training_losses.png')
    plt.close()

    # Simulate the controller
    print("=== Simulating controller ===")
    simulate_controller(lambda params, obs: trainer.controller.apply({'params': params}, obs), trained_params)

    pass

if __name__ == "__main__":
    run_experiment(
        jnp.array([1.0, 1.0, 9.81]), 
        num_iterations=20000, 
        episode_length=200, 
        hidden_layers = [64, 64, 32], 
        noise_std=0.0, 
        reg_strength=1e-5
    )