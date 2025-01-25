import jax
import jax.numpy as jnp

def noiseless_dyn(state, action, phi):
    """
    Compute the continuous-time dynamics of the pendulum system.
    Inputs:
        state (array): The state vector [theta, theta_dot].
        action (array): The control vector [u], where u is the torque applied to the pendulum.
        phi (array): The parameters of the system [mass of pendulum, length of pendulum, gravity constant]
    Returns:
        array: the next state vector [theta, theta_dot]
    """
    
    # hyperparameters
    max_speed = 8
    max_torque = 2.0
    dt = 0.05

    # unpack state, action and parameters
    th, thdot = state
    u = action[0]
    m, l, g = phi

    # clip the action
    u = jnp.clip(u, -max_torque, max_torque)

    # compute the next state
    newthdot = thdot - (3 * g / (2 * l)) * jnp.sin(th) * dt + 3.0 / (m * l**2) * u * dt
    newthdot = jnp.clip(newthdot, -max_speed, max_speed)
    newth = th + newthdot * dt

    # return the next state
    return jnp.array([newth, newthdot], dtype=jnp.float32)