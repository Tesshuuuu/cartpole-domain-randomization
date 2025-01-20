import jax.numpy as jnp
import matplotlib.pyplot as plt
from cartpole_trainer import CartPoleTrainer, create_default_cost_matrices
import time
import jax.random as random

def test_train_DR():
    """
    Test the Distributionally Robust training functionality by:
    1. Training both nominal and DR controllers
    2. Comparing their losses
    3. Testing their performance with perturbed parameters
    """
    # Define nominal dynamics parameters (mass_cart, mass_pole, length, gravity, friction_cart, friction_pole)
    nominal_params = jnp.array([1.0, 0.1, 0.5, 9.81, 0.1, 0.1])
    
    # Create cost matrices
    Q, R = create_default_cost_matrices()
    
    print("1. Training nominal controller...")
    trainer_nominal = CartPoleTrainer(
        dynamics_params=nominal_params,
        hidden_layers=[64, 32],
        noise_std=0.01,
        seed=42
    )
    
    start_time = time.time()
    params_nominal, losses_nominal = trainer_nominal.train(
        cost_matrices=(Q, R),
        num_iterations=1000,
        T=100,
        initial_learning_rate=0.01,
        seed=42
    )
    nominal_time = time.time() - start_time
    
    print("\n2. Training robust controller...")
    trainer_robust = CartPoleTrainer(
        dynamics_params=nominal_params,
        hidden_layers=[64, 32],
        noise_std=0.01,
        seed=42
    )
    
    start_time = time.time()
    params_robust, losses_robust = trainer_robust.train_DR(
        cost_matrices=(Q, R),
        num_iterations=1000,
        T=100,
        initial_learning_rate=0.01,
        scale_ellipsoid=0.1,  # 10% parameter uncertainty
        seed=42
    )
    robust_time = time.time() - start_time
    
    # Plot training losses
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses_nominal, label='Nominal')
    plt.plot(losses_robust, label='Robust')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)
    
    # Test robustness with parameter variations
    print("\n3. Testing robustness...")
    n_tests = 50
    perturbation_scale = 0.2  # 20% parameter variation
    
    nominal_costs = []
    robust_costs = []
    
    for i in range(n_tests):
        # Generate perturbed parameters
        key = random.PRNGKey(i)
        perturbed_params = nominal_params * (1 + perturbation_scale * (2 * random.uniform(key, shape=(6,)) - 1))
        
        # Test nominal controller
        trainer_nominal.dynamics_params = perturbed_params
        states_nominal, actions_nominal = trainer_nominal.simulate_trajectory(
            initial_state=jnp.array([0.1, 0.0, 0.1, 0.0]),
            noises=jnp.zeros((100, 4)),
            params=params_nominal
        )
        cost_nominal = trainer_nominal.compute_cost(states_nominal, actions_nominal, (Q, R))
        nominal_costs.append(float(cost_nominal))
        
        # Test robust controller
        trainer_robust.dynamics_params = perturbed_params
        states_robust, actions_robust = trainer_robust.simulate_trajectory(
            initial_state=jnp.array([0.1, 0.0, 0.1, 0.0]),
            noises=jnp.zeros((100, 4)),
            params=params_robust
        )
        cost_robust = trainer_robust.compute_cost(states_robust, actions_robust, (Q, R))
        robust_costs.append(float(cost_robust))
    
    # Plot robustness comparison
    plt.subplot(1, 2, 2)
    plt.boxplot([nominal_costs, robust_costs], labels=['Nominal', 'Robust'])
    plt.ylabel('Cost with Parameter Variations')
    plt.title('Robustness Comparison')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('dr_training_comparison.png')
    plt.close()
    
    # Print summary statistics
    print("\nSummary:")
    print(f"Training time - Nominal: {nominal_time:.2f}s, Robust: {robust_time:.2f}s")
    print(f"Final loss - Nominal: {float(losses_nominal[-1]):.4f}, Robust: {float(losses_robust[-1]):.4f}")
    print("\nRobustness test results:")
    print(f"Nominal controller - Mean cost: {jnp.mean(jnp.array(nominal_costs)):.4f}, Std: {jnp.std(jnp.array(nominal_costs)):.4f}")
    print(f"Robust controller - Mean cost: {jnp.mean(jnp.array(robust_costs)):.4f}, Std: {jnp.std(jnp.array(robust_costs)):.4f}")
    
    # Save controllers
    trainer_nominal.save_controller(params_nominal, suffix='nominal')
    trainer_robust.save_controller(params_robust, suffix='robust')

if __name__ == "__main__":
    test_train_DR()
