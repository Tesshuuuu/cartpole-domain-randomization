import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from noiseless_dyn import noiseless_dyn
import jax.numpy as jnp
from mlp_controller import create_example_controller

def simulate_controlled_cartpole():
    """
    Simulates and visualizes a cart-pole system with MLP controller.
    """
    # System parameters
    m_c = 1.0      # Cart mass (kg)
    m_p = 0.1      # Pole mass (kg)
    l = 0.5        # Pole length (m)
    b_x = 0.1      # Cart damping coefficient (N⋅s/m)
    b_theta = 0.1  # Pole damping coefficient (N⋅m⋅s/rad)
    g = 9.81       # Gravitational acceleration (m/s²)
    phi = jnp.array([m_c, m_p, l, b_x, b_theta, g])

    # Create and initialize controller
    controller, params, controller_fn = create_example_controller(
        state_dim=4,
        hidden_layers=[64, 32],  # Two hidden layers
        action_dim=1,
        seed=0
    )

    # Initial state configuration
    x0 = 0.0              # Initial cart position (m)
    x_dot0 = 0.0          # Initial cart velocity (m/s)
    theta0 = np.pi + 0.1  # Initial pole angle (rad) - slightly offset from upright
    theta_dot0 = 0.0      # Initial pole angular velocity (rad/s)
    state = jnp.array([x0, x_dot0, theta0, theta_dot0])

    # Simulation settings
    t_max = 5.0    # Total simulation time (seconds)
    dt = 0.05      # Time step (seconds)
    n_steps = int(t_max/dt)
    
    # Arrays to store history
    states = np.zeros((n_steps, 4))
    actions = np.zeros((n_steps, 1))
    states[0] = state

    # Run simulation
    for i in range(1, n_steps):
        # Get control action from MLP controller
        u = controller_fn(params, state)
        actions[i] = u
        
        # Update state using dynamics
        state = noiseless_dyn(state, u, phi)
        states[i] = state

    # Set up the animation
    fig = plt.figure(figsize=(15, 6))
    
    # Create subplot for cart-pole animation
    ax1 = fig.add_subplot(121, autoscale_on=False, xlim=(-2, 2), ylim=(-1, 1))
    ax1.grid(True)
    ax1.set_xlabel('Position (m)')
    ax1.set_ylabel('Height (m)')
    ax1.set_title('Cart-Pole System')
    ax1.set_aspect('equal')

    # Create subplot for control action plot
    ax2 = fig.add_subplot(122)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Control Force (N)')
    ax2.set_title('Control Action')
    ax2.grid(True)
    time_array = np.linspace(0, t_max, n_steps)
    control_line, = ax2.plot([], [], 'b-', label='Control Force')
    ax2.set_xlim(0, t_max)
    ax2.set_ylim(np.min(actions) - 1, np.max(actions) + 1)
    ax2.legend()

    # Drawing elements
    cart_width = 0.3
    cart_height = 0.2
    cart = plt.Rectangle((-cart_width/2, -cart_height/2), cart_width, cart_height, 
                        fill=True, color='blue', label='Cart')
    pole, = ax1.plot([], [], 'k-', linewidth=2, label='Pole')
    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)
    
    def init():
        """Initialize animation"""
        ax1.add_patch(cart)
        ax1.legend()
        control_line.set_data([], [])
        return cart, pole, time_text, control_line

    def animate(i):
        """Animation update function"""
        # Update cart-pole visualization
        x = states[i, 0]      # Cart position
        theta = states[i, 2]  # Pole angle
        
        # Update cart position
        cart.set_x(x - cart_width/2)
        
        # Update pole position
        pole_x = [x, x + l * np.sin(theta)]
        pole_y = [0, -l * np.cos(theta)]
        pole.set_data(pole_x, pole_y)
        
        # Update time display
        time_text.set_text(f'Time: {i*dt:.1f} s')

        # Update control action plot
        control_line.set_data(time_array[:i], actions[:i])
        
        return cart, pole, time_text, control_line

    # Create and display animation
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                 frames=n_steps, interval=dt*1000, blit=True)
    plt.tight_layout()
    plt.show()

    # Print final state
    print("\nFinal State:")
    print(f"Cart Position: {states[-1, 0]:.3f} m")
    print(f"Cart Velocity: {states[-1, 1]:.3f} m/s")
    print(f"Pole Angle: {states[-1, 2]:.3f} rad")
    print(f"Pole Angular Velocity: {states[-1, 3]:.3f} rad/s")

if __name__ == "__main__":
    simulate_controlled_cartpole()