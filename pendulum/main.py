import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import jax.numpy as jnp
import os
import pickle
from datetime import datetime
from typing import Tuple

from noiseless_dyn import noiseless_dyn
from pendulum_trainer import PendulumTrainer
from simulate import simulate_controller

def run_experiment(phi, num_iterations=2000):
    """
    Run the experiment to train the controller and simulate the pendulum system.
    """
    trainer = PendulumTrainer(phi, hidden_layers=[64, 32], noise_std=0.01)

    # Define the cost matrices
    Q = jnp.zeros((2, 2))
    Q = Q.at[0, 0].set(1.0)
    Q = Q.at[1, 1].set(0.1)
    R = 0.001 * jnp.eye(1)
    cost_matrices = (Q, R)

    print("=== Training controller ===")
    trained_params, losses = trainer.train(
        cost_matrices=cost_matrices,
        num_iterations=num_iterations,
        T=100,
        initial_learning_rate=0.01
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
    run_experiment(jnp.array([1.0, 1.0, 9.81]), num_iterations=5000)
