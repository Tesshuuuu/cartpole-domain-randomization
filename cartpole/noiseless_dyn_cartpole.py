import jax
import jax.numpy as jnp
from jax import jit

@jit
def noiseless_dyn_cartpole(in_state, u, phi):
    """
    Compute the continuous-time dynamics of the cart-pole system.
    Inputs:
        state (array): The state vector [x, x_dot, theta, theta_dot].
        control (array): The control vector [F], where F is the force applied to the cart.
        phi (array): The parameters of the system [mass of cart, mass of pole, length of pole, 
            damping coefficient of cart, damping coefficient of pole, gravity constant]
    Returns:
        array: the next state vector [x, x_dot, theta, theta_dot]
    """ 
    # Unpack the state vector
    x, x_dot, theta, theta_dot = in_state
    # Unpack the control vector
    F = u[0]
    # Unpack the parameters
    m_c, m_p, l, b_x, b_theta, g = phi
    
    # Intermediate calculations
    sin_theta = jnp.sin(theta)
    cos_theta = jnp.cos(theta)
    a1 = jnp.array([[m_c + m_p, m_p*l*cos_theta], [m_p*cos_theta, m_p*l]])
    b1 = jnp.array([[m_p*l*theta_dot**2*sin_theta+F], [m_p*g*sin_theta]])
    c1 = jnp.linalg.inv(a1)@b1
    x_ddot = c1[0,0]
    theta_ddot = c1[1,0]
    dt = 0.05
    new_state = in_state + dt*jnp.array([x_dot, x_ddot - b_x*x_dot, theta_dot, theta_ddot - b_theta*theta_dot])
    new_state = new_state.at[2].set(jnp.mod(new_state[2] + jnp.pi, 2*jnp.pi) - jnp.pi)
    return new_state