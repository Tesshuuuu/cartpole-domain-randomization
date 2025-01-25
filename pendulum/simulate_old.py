import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import jax.numpy as jnp
import jax

from noiseless_dyn import noiseless_dyn
def simulate_controller(
    controller_fn, 
    params, 
    duration, 
    dt, 
    save_animation=False):

    """
    Simulate the pendulum system with the trained controller.
    
    Args:
        controller_fn: Trained controller function
        params: Trained controller parameters
        duration: Simulation duration in seconds
        dt: Time step in seconds
        save_animation: Whether to save the animation as MP4
    """
    # System parameters [m, l, g]
    dynamics_params = jnp.array([1.0, 1.0, 9.81])
    
    # Simulation settings
    n_steps = int(duration/dt)
    
    # Initialize state arrays for different initial conditions
    initial_conditions = [
        jnp.array([jnp.pi, 0.0]),     # Bottom position
        jnp.array([0.0, 0.0]),        # Top position
        jnp.array([jnp.pi/2, 0.0]),   # Horizontal position
    ]
    
    all_states = []
    all_actions = []
    
    # Simulate for each initial condition
    for initial_state in initial_conditions:
        states = np.zeros((n_steps, 2))
        actions = np.zeros((n_steps, 1))
        
        state = initial_state
        states[0] = state
        
        # Simulation loop
        def simulate_step(state, action):
            obs = jnp.array([jnp.cos(state[0]), jnp.sin(state[0]), state[1]])
            action = controller_fn(params, obs)
            next_state = noiseless_dyn(state, action, dynamics_params)
            return next_state, (next_state, action)

        _, (simulated_states, simulated_actions) = jax.lax.scan(simulate_step, initial_state, jnp.zeros(n_steps))
        states = np.array(simulated_states)
        actions = np.array(simulated_actions)

        # for i in range(1, n_steps):
        #     obs = jnp.array([jnp.cos(state[0]), jnp.sin(state[0]), state[1]])
        #     action = controller_fn(params, obs)
        #     actions[i-1] = action
        #     state = noiseless_dyn(state, action, dynamics_params)
        #     states[i] = state
            
        all_states.append(states)
        all_actions.append(actions)


    # Create visualization
    fig = plt.figure(figsize=(15, 10))

    n_steps = all_states[0].shape[0]
    duration = n_steps * dt
    dynamics_params = jnp.array([1.0, 1.0, 9.81])

    # Pendulum animation subplot
    ax1 = fig.add_subplot(221, autoscale_on=False)
    ax1.set_xlim(-2.0, 2.0)
    ax1.set_ylim(-2.0, 2.0)
    ax1.grid(True)
    ax1.set_aspect('equal')
    ax1.set_title('Pendulum System')
    
    # State plots
    ax2 = fig.add_subplot(222)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('States')
    ax2.set_title('System States')
    ax2.grid(True)
    ax2.set_ylim(-8.0, 8.0)
    ax2.set_xlim(0, duration)
    
    # Control action plot
    ax3 = fig.add_subplot(223)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Control Torque (N⋅m)')
    ax3.set_title('Control Actions')
    ax3.grid(True)
    ax3.set_ylim(-2.0, 2.0)
    ax3.set_xlim(0, duration)
    # Time array for plotting
    time_array = np.linspace(0, duration, n_steps)
    
    # Create drawing elements
    lines = []
    bobs = []
    state_lines = []
    action_lines = []
    colors = ['b', 'r', 'g']
    labels = ['Bottom start', 'Top start', 'Horizontal start']
    
    for i, color in enumerate(colors):
        # Pendulum visualization
        line, = ax1.plot([], [], f'{color}-', linewidth=2, label=labels[i])
        bob = ax1.plot([], [], f'{color}o', markersize=10)[0]
        lines.append(line)
        bobs.append(bob)
        
        # State plots
        state_lines.extend([
            ax2.plot([], [], f'{color}-', label=f'{labels[i]}: θ')[0],
            ax2.plot([], [], f'{color}--', label=f'{labels[i]}: θ̇')[0]
        ])
        
        # Control action plot
        action_lines.append(ax3.plot([], [], f'{color}-', label=labels[i])[0])
    
    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)
    
    def init():
        """Initialize animation"""
        ax1.legend()
        ax2.legend()
        ax3.legend()
        return lines + bobs + state_lines + action_lines + [time_text]
    
    def animate(frame):
        """Animation update function"""
        for i in range(len(initial_conditions)):
            theta = all_states[i][frame, 0]
            
            # Calculate pendulum position
            x = dynamics_params[1] * jnp.cos(theta)  # vertical position
            y = -dynamics_params[1] * jnp.sin(theta)  # horizontal position
            
            # Update pendulum visualization
            lines[i].set_data([0, y], [0, x])  # Swap x and y for correct orientation
            bobs[i].set_data([y], [x])
            
            # Update state plots
            state_lines[i*2].set_data(time_array[:frame], all_states[i][:frame, 0])
            state_lines[i*2+1].set_data(time_array[:frame], all_states[i][:frame, 1])
            
            # Update control action plot
            action_lines[i].set_data(time_array[:frame], all_actions[i][:frame, 0])
        
        time_text.set_text(f'Time: {frame*dt:.1f} s')
        return lines + bobs + state_lines + action_lines + [time_text]
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                 frames=n_steps, interval=dt*1000, blit=True)
    
    if save_animation:
        anim.save('pendulum_animation.mp4', writer='ffmpeg')
    
    plt.tight_layout()
    plt.show()
