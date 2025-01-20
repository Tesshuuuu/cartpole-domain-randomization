import jax
import jax.numpy as jnp
from jax import jit
from noiseless_dyn import noiseless_dyn
from functools import partial
from scipy import optimize


@jit
def dynamics(state, input, noise, dynamics_params):
    """Simulates one step of the noisy dynamics using noiseless_dyn
    Args:
        state: Current system state
        input: Control input
        noise: State noise vector
        dynamics_params: Parameters of the dynamics model
    Returns:
        next_state: Next state after applying dynamics and noise
    """
    return noiseless_dyn(state, input, dynamics_params) + noise

@jit
def experiment_dynamics(state, noise, dynamics_params):
    """Simulates one step of the noisy dynamics
    Args:
        state: Current system state
        noise: Combined noise vector [state_noise (4), input_noise (du)]
        dynamics_params: Parameters of the dynamics model
    Returns:
        next_state: The next state after applying dynamics and noise
        input: The noisy input used
    """
    input = noise[4:]  # Extract input noise
    noise = noise[:4]  # Extract state noise
    next_state = dynamics(state, input, noise, dynamics_params)
    return next_state, input

@jit
def simulate_experiment(initial_state, noises, dynamics_params):
    """Simulates a trajectory using noisy dynamics
    Args:
        initial_state: Starting state
        noises: Sequence of noise vectors for each timestep
        dynamics_params: Parameters of the dynamics model
    Returns:
        states: Array of states including initial and final states
        inputs: Array of inputs used during simulation
    """
    def step_fn(carry, noise):
        state = carry
        next_state, input = experiment_dynamics(state, noise, dynamics_params)
        return next_state, (state, input)
    
    # Use scan for efficient loop operation
    last_state, (states, inputs) = jax.lax.scan(step_fn, initial_state, noises)
    return jnp.vstack([states, last_state[jnp.newaxis]]), inputs

def collect_traj(key, dynamics_params, x0s, du, T):
    """Collects multiple trajectories in parallel
    Args:
        key: JAX PRNGKey for random number generation
        dynamics_params: Parameters of the dynamics model
        x0s: Initial states for each trajectory
        du: Input dimension
        T: Number of timesteps
    Returns:
        Tuple of (states, inputs) for all trajectories
    """
    dx = x0s.shape[1]
    # Generate noise for all trajectories
    noise = jax.random.normal(key, (len(x0s), T, dx + du))
    # Vectorized simulation across multiple initial conditions
    return jax.vmap(simulate_experiment, in_axes=(0, 0, None), out_axes=(0,0))(x0s, noise, dynamics_params)

def est_phi(key, data, n_inits, learning_rate=1e-3, max_iterations=2000, init_noise=0.01):
    """Estimates dynamics parameters from trajectory data
    Args:
        key: JAX PRNGKey
        data: Tuple of (states, inputs) from collected trajectories
        n_inits: Number of random initializations
        learning_rate: Learning rate for optimization
        max_iterations: Maximum optimization iterations
    Returns:
        best_params: Estimated dynamics parameters
    """
    # Vectorized noiseless dynamics for batch processing
    noiseless_dyn_vec = jax.vmap(jax.vmap(noiseless_dyn, in_axes=(0,0,None)), in_axes=(0,0,None))
    
    def loss_fn(phi, data, training=False):
        xs, us = data
        # Compute prediction error
        pred_error = jnp.mean((xs[:,1:] - noiseless_dyn_vec(xs[:,:-1], us, phi))**2)
        
        if training:
            # Adjust regularization strength
            reg_lambda = 0.001
            reg_term = reg_lambda * jnp.sum(phi**2)
            return pred_error + reg_term
        return pred_error
    
    # Initialize tracking variables
    best_solution = None
    best_loss = jnp.inf
    
    for i in range(n_inits):
        try:
            key, subkey = jax.random.split(key)
            # Use better initialization scaling
            # init_phi = jax.random.uniform(subkey, (6,), 
            #                             minval=0.1, 
            #                             maxval=2.0)
            init_phi = jnp.array([1.0, 0.1, 0.5, 0.1, 0.1, 9.81]) + jax.random.normal(subkey, (6,)) * init_noise

            # Use scipy.optimize instead of jax.optimize
            result = optimize.minimize(
                fun=lambda x: loss_fn(x, data),  # Modified to work with scipy's interface
                x0=init_phi,
                method='L-BFGS-B',
                options={
                    'maxiter': max_iterations,
                    'ftol': 1e-8,
                    'gtol': 1e-8
                }
            )
            
            if result.success and result.fun < best_loss:
                best_loss = result.fun
                best_solution = result.x
                
        except Exception as e:
            print(f"Optimization {i+1}/{n_inits} failed: {str(e)}")
            continue
    
    if best_solution is None:
        raise ValueError("All optimization attempts failed")
        
    return best_solution

def validate_dynamics(phi, validation_data, noise_std=0.0):
    """Validates the estimated dynamics parameters
    Args:
        phi: Estimated parameters
        validation_data: Held-out trajectory data
        noise_std: Standard deviation of validation noise
    Returns:
        validation_error: Mean squared error on validation data
    """
    xs, us = validation_data
    noiseless_dyn_vec = jax.vmap(jax.vmap(noiseless_dyn, in_axes=(0,0,None)), 
                                in_axes=(0,0,None))
    
    # Add noise to validation if specified
    if noise_std > 0:
        key = jax.random.PRNGKey(0)
        noise = jax.random.normal(key, xs[:,:-1].shape) * noise_std
        predictions = noiseless_dyn_vec(xs[:,:-1] + noise, us, phi)
    else:
        predictions = noiseless_dyn_vec(xs[:,:-1], us, phi)
    
    return jnp.mean((xs[:,1:] - predictions)**2)

def main():
    """Main function to demonstrate dynamics parameter estimation"""
    # Ask user for init_noise value
    while True:
        try:
            init_noise = float(input("Please enter the initialization noise value (default=0.01): ") or "0.01")
            if init_noise < 0:
                print("Please enter a non-negative value.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")
    
    # Set random seed for reproducibility
    key = jax.random.PRNGKey(0)
    
    # True dynamics parameters (mass_cart, mass_pole, length, friction_cart, friction_pole, gravity)
    true_params = jnp.array([1.0, 0.1, 0.5, 0.1, 0.1, 9.81])
    
    # Generate training data
    n_trajectories = 100  # Number of trajectories
    T = 200  # Time steps per trajectory
    du = 1   # Input dimension (force)
    
    # Generate random initial states
    key, subkey = jax.random.split(key)
    x0s = jax.random.uniform(
        subkey, 
        shape=(n_trajectories, 4),
        minval=jnp.array([-2.0, -1.0, -0.2, -1.0]),  # Wider ranges
        maxval=jnp.array([2.0, 1.0, 0.2, 1.0])
    )
    
    # Collect trajectories
    training_data = collect_traj(key, true_params, x0s, du, T)
    
    # Generate validation data
    key, subkey = jax.random.split(key)
    x0s_val = jax.random.uniform(
        subkey, 
        shape=(n_trajectories // 5, 4),  # Smaller validation set
        minval=jnp.array([-1.0, -0.5, -0.1, -0.5]),
        maxval=jnp.array([1.0, 0.5, 0.1, 0.5])
    )
    validation_data = collect_traj(key, true_params, x0s_val, du, T)
    
    # Estimate parameters using user input
    print("Estimating dynamics parameters...")
    estimated_params = est_phi(key, training_data, n_inits=20, init_noise=init_noise)
    
    # Validate results
    training_error = validate_dynamics(estimated_params, training_data)
    validation_error = validate_dynamics(estimated_params, validation_data)
    
    # Print results
    print("\nResults:")
    print("True parameters:", true_params)
    print("Estimated parameters:", estimated_params)
    print("\nParameter-wise relative error:")
    rel_error = jnp.abs((estimated_params - true_params) / true_params) * 100
    param_names = ['mass_cart', 'mass_pole', 'length', 'friction_cart', 'friction_pole', 'gravity']
    for name, error in zip(param_names, rel_error):
        print(f"{name}: {error:.2f}%")
    print(f"\nTraining MSE: {training_error:.6f}")
    print(f"Validation MSE: {validation_error:.6f}")

    # Test with different noise levels
    noise_levels = [0.0, 0.01, 0.05]
    for noise_std in noise_levels:
        val_error = validate_dynamics(estimated_params, validation_data, noise_std)
        print(f"Validation MSE (noise={noise_std}): {val_error:.6f}")
    
    return estimated_params, true_params

if __name__ == "__main__":
    main()