import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import jax.numpy as jnp
from cartpole_trainer import CartPoleTrainer, create_default_cost_matrices
from noiseless_dyn import noiseless_dyn
import os
import pickle
from datetime import datetime

def simulate_trained_controller(controller_fn, params, duration=20.0, dt=0.02, save_animation=False):
    """
    Simulate the cart-pole system with the trained controller.
    
    Args:
        controller_fn: Trained controller function
        params: Trained controller parameters
        duration: Simulation duration in seconds
        dt: Time step in seconds
        save_animation: Whether to save the animation as MP4
    """
    # System parameters [m_c, m_p, l, b_x, b_theta, g]
    dynamics_params = jnp.array([1.0, 0.1, 0.5, 0.1, 0.1, 9.81])
    
    # Simulation settings
    n_steps = int(duration/dt)
    
    # Initialize state arrays for different initial conditions
    initial_conditions = [
        jnp.array([0.0, 0.0, np.pi + 0.1, 0.0]),  # Near upright
        jnp.array([1.0, 0.0, np.pi + 0.1, 0.0]),  # Start at x = 1.0
        jnp.array([-0.5, 0.0, np.pi - 0.1, 0.0]), # Start at x = -0.5
    ]
    
    all_states = []
    all_actions = []
    
    # Simulate for each initial condition
    for initial_state in initial_conditions:
        states = np.zeros((n_steps, 4))
        actions = np.zeros((n_steps, 1))
        
        state = initial_state
        states[0] = state
        
        # Simulation loop
        for i in range(1, n_steps):
            # Get control action from trained controller
            action = controller_fn(params, state)
            actions[i-1] = action
            
            # Update state
            state = noiseless_dyn(state, action, dynamics_params)
            states[i] = state
            
        all_states.append(states)
        all_actions.append(actions)

    # Visualization
    fig = plt.figure(figsize=(15, 10))
    
    # Cart-pole animation
    ax1 = fig.add_subplot(221, autoscale_on=False, xlim=(-2, 2), ylim=(-1, 1))
    ax1.grid(True)
    ax1.set_xlabel('Position (m)')
    ax1.set_ylabel('Height (m)')
    ax1.set_title('Cart-Pole System')
    
    # State plots
    ax2 = fig.add_subplot(222)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('States')
    ax2.set_title('System States')
    ax2.grid(True)
    
    # Control action plot
    ax3 = fig.add_subplot(212)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Control Force (N)')
    ax3.set_title('Control Actions')
    ax3.grid(True)
    
    # Time array for plotting
    time_array = np.linspace(0, duration, n_steps)
    
    # Drawing elements
    cart_width = 0.3
    cart_height = 0.2
    pole_length = 0.5
    
    # Create drawing elements for each initial condition
    carts = []
    poles = []
    state_lines = []
    action_lines = []
    colors = ['b', 'r', 'g']
    labels = ['Near upright', 'Start at x=1.0', 'Start at x=-0.5']
    
    for i, color in enumerate(colors):
        # Cart and pole
        cart = plt.Rectangle((-cart_width/2, -cart_height/2), cart_width, cart_height,
                           fill=True, color=color, alpha=0.3, label=labels[i])
        pole, = ax1.plot([], [], f'{color}-', linewidth=2)
        carts.append(cart)
        poles.append(pole)
        
        # State plots
        state_lines.extend([
            ax2.plot([], [], f'{color}-', label=f'{labels[i]}: pos')[0],
            ax2.plot([], [], f'{color}--', label=f'{labels[i]}: angle')[0]
        ])
        
        # Control action plot
        action_lines.append(ax3.plot([], [], f'{color}-', label=labels[i])[0])
    
    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)
    
    def init():
        """Initialize animation"""
        for cart in carts:
            ax1.add_patch(cart)
        ax1.legend()
        ax2.legend()
        ax3.legend()
        return carts + poles + state_lines + action_lines + [time_text]
    
    def animate(frame):
        """Animation update function"""
        for i in range(len(initial_conditions)):
            # Update cart and pole
            x = all_states[i][frame, 0]
            theta = all_states[i][frame, 2]
            
            carts[i].set_x(x - cart_width/2)
            
            pole_x = [x, x + pole_length * np.sin(theta)]
            pole_y = [0, -pole_length * np.cos(theta)]
            poles[i].set_data(pole_x, pole_y)
            
            # Update state plots
            state_lines[i*2].set_data(time_array[:frame], all_states[i][:frame, 0])
            state_lines[i*2+1].set_data(time_array[:frame], all_states[i][:frame, 2])
            
            # Update control action plot
            action_lines[i].set_data(time_array[:frame], all_actions[i][:frame, 0])
        
        time_text.set_text(f'Time: {frame*dt:.1f} s')
        return carts + poles + state_lines + action_lines + [time_text]
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                 frames=n_steps, interval=dt*1000, blit=True)
        
    plt.tight_layout()
    plt.show()



def load_and_simulate(controller_path, duration=10.0, dt=0.02, save_animation=False):
    """
    Load a saved controller and simulate it.
    
    Args:
        controller_path: Path to the saved controller file
        duration: Simulation duration in seconds
        dt: Time step in seconds
        save_animation: Whether to save the animation
    """
    print(f"Loading controller from: {controller_path}")
    trainer, loaded_params = CartPoleTrainer.load_controller(controller_path)
    
    print("Simulating loaded controller...")
    simulate_trained_controller(
        trainer.controller_fn, 
        loaded_params,
        duration=duration,
        dt=dt,
        save_animation=save_animation
    )

def list_saved_controllers(save_dir='saved_controllers'):
    """List all saved controllers and let user choose one."""
    if not os.path.exists(save_dir):
        print("No saved controllers found.")
        return None
    
    controllers = sorted([f for f in os.listdir(save_dir) if f.endswith('.pkl')])
    
    if not controllers:
        print("No controllers found in directory.")
        return None
    
    print("\nAvailable Controllers:")
    for i, controller in enumerate(controllers):
        print(f"{i+1}. {controller}")
    
    while True:
        try:
            choice = int(input("\nEnter controller number to load (0 to cancel): "))
            if choice == 0:
                return None
            if 1 <= choice <= len(controllers):
                return os.path.join(save_dir, controllers[choice-1])
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def main():
    """Main function to train and simulate the controller"""
    # Train the controller
    dynamics_params = jnp.array([1.0, 0.1, 0.5, 0.1, 0.1, 9.81])
    trainer = CartPoleTrainer(
        dynamics_params,
        hidden_layers=[64, 64, 32],
        noise_std=0.01
    )
    cost_matrices = create_default_cost_matrices()
    
    print("Training controller...")
    trained_params, losses = trainer.train(
        cost_matrices=cost_matrices,
        num_iterations=30000,
        T=300,
        initial_learning_rate=0.005
    )
    
    # Save the trained controller
    save_path = trainer.save_controller(trained_params)
    print(f"Controller saved to: {save_path}")
    
    # Plot training losses
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.grid(True)
    plt.savefig('training_losses.png')
    plt.close()
    
    print("Training completed. Simulating controller...")
    # Simulate the trained controller
    simulate_trained_controller(trainer.controller_fn, trained_params, save_animation=True)

if __name__ == "__main__":
    # Check if we want to train a new controller or load an existing one
    choice = input("Do you want to (t)rain a new controller or (l)oad an existing one? [t/l]: ").lower()
    
    if choice == 't':
        main()
    elif choice == 'l':
        controller_path = list_saved_controllers()
        if controller_path:
            load_and_simulate(controller_path, save_animation=True)
    else:
        print("Invalid choice. Exiting.")