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
import numpy as np

from noiseless_dyn import noiseless_dyn
from mlp_controller import create_example_controller
from simulate import simulate_controller
from simulate_old import simulate_controller as simulate_controller_old


class PendulumTrainer:
    """
    Trainer class for the pendulum system using an MLP controller.
    """

    def __init__(self, 
                 phi: jnp.ndarray,
                 true_phi: jnp.ndarray,
                 FI: jnp.ndarray = None,
                 obs_dim: int = 3,
                 action_dim: int = 1,
                 hidden_layers: list[int] = [64, 32],
                 noise_std: float = 0.01,
                 seed: int = 0):
        
        self.phi = phi
        self.true_phi = true_phi
        self.FI = FI if FI is not None else jnp.eye(len(phi))
        self.noise_std = noise_std

        self.controller, self.nn_params, self.controller_fn = create_example_controller(
            obs_dim=obs_dim, 
            action_dim=action_dim,
            hidden_layers=hidden_layers,
            seed=seed
        )
    
    @partial(jit, static_argnums=(0,))
    def single_closed_loop_dynamics(self, state, noise, nn_params, phi, noise_std):
        obs = jnp.array([jnp.cos(state[0]), jnp.sin(state[0]), state[1]])
        action = self.controller.apply({'params': nn_params}, obs)
        next_state = noiseless_dyn(state, action, phi) + noise_std * noise
        return next_state, action

    @partial(jit, static_argnums=(0,))
    def closed_loop_dynamics(self, state, noise, nn_params, phi, noise_std):
        next_states, actions = self.single_closed_loop_dynamics(state, noise, nn_params, phi, noise_std)
        return next_states, actions

    @partial(jit, static_argnums=(0,))
    def closed_loop_dynamics_batch(self, states: jnp.ndarray, noises: jnp.ndarray, nn_params: Dict, phi: jnp.ndarray, noise_std: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute the closed loop dynamics."""
        next_states, actions = jax.vmap(
            lambda state, noise: self.single_closed_loop_dynamics(state, noise, nn_params, phi, noise_std)
        )(states, noises)

        return next_states, actions

    @partial(jit, static_argnums=(0,))
    def simulate_trajectory(self, initial_states, noises, phi, noise_std):
        def step_fn(carry, noise):
            states = carry
            next_states, actions = self.closed_loop_dynamics_batch(states, noise, self.nn_params, phi, noise_std)
            return next_states, (next_states, actions)  # Return next_states as carry

        _, (states_traj, actions_traj) = lax.scan(step_fn, initial_states, noises)
        return states_traj, actions_traj
    
    @partial(jit, static_argnums=(0,))
    def loss_fn(self, 
                states_traj: jnp.ndarray,
                actions_traj: jnp.ndarray,
                cost_matrices: Tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        """
        Given the trajectory, compute the loss.
        """
        Q, R = cost_matrices

        # Current angle wrapping might cause discontinuities
        # angle_error = (states_traj[:, :, 0] + jnp.pi) % (2 * jnp.pi) - jnp.pi
        
        # Consider using a continuous representation:
        angle_error = jnp.arctan2(jnp.sin(states_traj[:, :, 0]), jnp.cos(states_traj[:, :, 0]))
        
        velocity_error = states_traj[:, :, 1]
        # action_cost = jnp.sum(actions_traj * (R @ actions_traj[..., None])[..., 0], axis=-1)
        action_cost = actions_traj[:, :, 0]**2

        cost = Q[0,0] * angle_error**2 + Q[1,1] * velocity_error**2 + action_cost * R[0,0]

        return jnp.mean(cost)
    
    def train(self, 
              cost_matrices: Tuple[jnp.ndarray, jnp.ndarray],
              num_iterations: int = 1000,
              T: int = 100,
              initial_learning_rate: float = 0.01,
              reg_strength: float = 1e-3,
              batch_size: int = 32,
              seed: int = 0) -> Tuple[Dict, jnp.ndarray]:
        
        # schedule_fn = optax.constant_schedule(initial_learning_rate)
        schedule_fn = optax.exponential_decay(init_value=initial_learning_rate, transition_steps=num_iterations, decay_rate=0.9)
        optimizer = optax.adam(learning_rate=schedule_fn)
        opt_state = optimizer.init(self.nn_params)

        key = random.PRNGKey(seed)
        losses = []
        
        state_dim = cost_matrices[0].shape[0]

        # Define a function that returns just the loss value
        def loss_with_reg(params, states_traj, actions_traj, cost_matrices):
            return (self.loss_fn(states_traj, actions_traj, cost_matrices) + self.compute_l2_regularization(params) * reg_strength)

        # Create the value and gradient function
        loss_grad_fn = jit(value_and_grad(loss_with_reg))
        
        for i in range(num_iterations):
            key, noise_key = random.split(key, 2)
            noises = random.normal(noise_key, shape=(T, batch_size, state_dim))
       
            key, subkey1, subkey2 = random.split(key, 3)
            initial_states = jnp.stack([
                random.uniform(subkey1, shape=(batch_size, ), minval=-jnp.pi/2, maxval=jnp.pi/2),
                random.uniform(subkey2, shape=(batch_size, ), minval=-1, maxval=1)
            ], axis=1)

            # Simulate trajectory
            states_traj, actions_traj = self.simulate_trajectory(initial_states, noises, self.phi, self.noise_std)
            
            # Get loss and gradients
            loss, grads = loss_grad_fn(self.nn_params, states_traj, actions_traj, cost_matrices)

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
    
    def simulate_trajectory_eval(self, T, initial_conditions):
        
        all_states = []
        all_actions = []
        state_dim = initial_conditions[0].shape[0]
        
        # Simulate for each initial condition
        for initial_state in initial_conditions:
            noises = jnp.zeros((T, state_dim))

            def step_fn(carry, noise):
                states = carry
                next_states, actions = self.closed_loop_dynamics(states, noise, self.nn_params, self.true_phi, 0.0)
                return next_states, (states, actions)  # Return next_states as carry

            _, (states_traj, actions_traj) = lax.scan(step_fn, initial_state, noises)

            all_states.append(np.array(states_traj))
            all_actions.append(np.array(actions_traj))

        return all_states, all_actions
    
def plot_losses(losses):
    # Plot the loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    # plot the mean average loss
    window_size = 10
    mean_losses = jnp.convolve(losses, jnp.ones(window_size)/window_size, mode='valid')
    plt.plot(mean_losses, 'r--', alpha=0.8, label='Short moving average')
    window_size = 100
    mean_losses = jnp.convolve(losses, jnp.ones(window_size)/window_size, mode='valid')
    plt.plot(mean_losses, 'g--', alpha=0.8, label='Medium moving average')
    window_size = 1000
    mean_losses = jnp.convolve(losses, jnp.ones(window_size)/window_size, mode='valid')
    plt.plot(mean_losses, 'b--', alpha=0.8, label='Long moving average')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.grid(True)
    plt.savefig('training_losses.png')
    plt.close()

def run_experiment(
        phi = jnp.array([1.0, 1.0, 9.81]), 
        true_phi = jnp.array([1.0, 1.0, 9.81]),
        num_iterations=1000, 
        episode_length=100, 
        hidden_layers = [64, 64, 64, 32], 
        noise_std=0.0,
        reg_strength=1e-4, 
        initial_learning_rate=0.00001, 
        batch_size=32 ,
        cost_matrices=(jnp.array([[128.0, 0.0], [0.0, 0.1]]), jnp.array([[0.01]]))
    ):
    """
    Run the experiment to train the controller and simulate the pendulum system.
    """
    trainer = PendulumTrainer(phi, true_phi, hidden_layers=hidden_layers, noise_std=noise_std)

    print("=== Training controller ===")
    trained_params, losses = trainer.train(
        cost_matrices=cost_matrices,
        num_iterations=num_iterations,
        T=episode_length,
        initial_learning_rate=initial_learning_rate, 
        reg_strength=reg_strength,
        batch_size=batch_size
    )
    plot_losses(losses)

    # Simulate the controller
    print("=== Simulating controller ===")
    # simulate_controller(lambda params, obs: trainer.controller.apply({'params': params}, obs), trained_params)
    # Initialize state arrays for different initial conditions
    initial_conditions = [
        jnp.array([jnp.pi, 0.0]),     # Bottom position
        jnp.array([0.0, 0.0]),        # Top position
        jnp.array([jnp.pi/2, 0.0]),   # Horizontal position
    ]   
    # all_states, all_actions = trainer.simulate_trajectory_eval(episode_length, initial_conditions)
    # simulate_controller(all_states, all_actions, initial_conditions)
    simulate_controller_old(lambda params, obs: trainer.controller.apply({'params': params}, obs), trained_params, duration=60, dt=0.05)
    pass

if __name__ == "__main__":
    run_experiment()
