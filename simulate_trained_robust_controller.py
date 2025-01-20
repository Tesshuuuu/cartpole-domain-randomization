import numpy as np
import jax.numpy as jnp
import jax
from cartpole_trainer import CartPoleTrainer, create_default_cost_matrices
from simulate_trained_controller import simulate_trained_controller
from estimate_dyn import main as estimate_dyn_main
import matplotlib.pyplot as plt
import os

def train_robust_controller():
    """
    Main function implementing the pipeline:
    1. (Optional) Estimate dynamics parameters if not using saved controllers
    2. Load existing controller and/or train robust controller
    3. Simulate controller with true parameters
    """
    # Ask user whether to use saved controller or train new one
    use_saved = input("Do you want to use a saved controller? (y/n): ").lower().strip() == 'y'
    
    if use_saved:
        # List available controllers
        saved_dir = "saved_controllers"
        if not os.path.exists(saved_dir):
            print(f"Error: {saved_dir} directory not found!")
            return
            
        saved_files = [f for f in os.listdir(saved_dir) if f.endswith('.pkl') and 'robust' in f]
        
        if not saved_files:
            print(f"No saved robust controllers found in {saved_dir}!")
            return
            
        print("\nAvailable robust controllers:")
        for i, file in enumerate(saved_files):
            print(f"{i+1}. {file}")
        
        # Get user choice
        while True:
            try:
                choice = int(input("\nEnter the number of the controller to load: "))
                if 1 <= choice <= len(saved_files):
                    break
                print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
        
        filepath = os.path.join(saved_dir, saved_files[choice-1])
        
        # Load controller and its estimated parameters
        trainer, trained_params = CartPoleTrainer.load_controller(filepath)
        
        print(f"Loaded robust controller from: {filepath}")
        print(f"Using estimated parameters from saved controller: {trainer.dynamics_params}")
        
        # Ask if user wants to train the loaded controller further
        continue_training = input("\nDo you want to continue training this controller? (y/n): ").lower().strip() == 'y'
        
        if continue_training:
            # Get training parameters from user
            try:
                num_iterations = int(input("Enter number of additional training iterations (default: 30000): ") or "30000")
                learning_rate = float(input("Enter initial learning rate (default: 0.005): ") or "0.005")
                scale_ellipsoid = float(input("Enter scale factor for parameter uncertainty (default: 0.1): ") or "0.1")
            except ValueError:
                print("Invalid input. Using default values.")
                num_iterations = 30000
                learning_rate = 0.005
                scale_ellipsoid = 0.1
            
            print("\nContinuing training with loaded controller...")
            cost_matrices = create_default_cost_matrices()
            trained_params, losses = trainer.train_DR(
                cost_matrices=cost_matrices,
                num_iterations=num_iterations,
                T=300,
                initial_learning_rate=learning_rate,
                scale_ellipsoid=scale_ellipsoid
            )
            
            # Plot training losses
            plt.figure(figsize=(8, 5))
            plt.plot(losses)
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('Additional Robust Controller Training Progress')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('robust_additional_training_loss.png')
            plt.close()
            
            # Save the further trained controller
            save_path = trainer.save_controller(trained_params, suffix='robust_continued')
            print(f"\nFurther trained controller saved to: {save_path}")
        
    else:
        # 1. Estimate dynamics parameters
        print("\nStep 1: Estimating dynamics parameters...")
        estimated_params, true_params = estimate_dyn_main()
        
        # 2. Train robust controller using estimated parameters
        print("\nStep 2: Training robust controller with estimated parameters...")
        trainer = CartPoleTrainer(
            estimated_params,
            hidden_layers=[64, 64, 32],
            noise_std=0.01
        )
        
        cost_matrices = create_default_cost_matrices()
        trained_params, losses = trainer.train_DR(
            cost_matrices=cost_matrices,
            num_iterations=30000,
            T=300,
            initial_learning_rate=0.005,
            scale_ellipsoid=0.1
        )
        
        # Plot training losses
        plt.figure(figsize=(8, 5))
        plt.plot(losses)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Robust Controller Training Progress')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('robust_training_loss.png')
        plt.close()
        
        # Save the controller
        save_path = trainer.save_controller(trained_params, suffix='robust')
        print(f"\nRobust controller saved to: {save_path}")

    # 3. Simulate with true parameters
    print("\nStep 3: Simulating controller with true parameters...")
    
    # Simulate robust controller
    print("\nSimulating robust controller...")
    simulate_trained_controller(
        trainer.controller_fn, 
        trained_params,
        duration=20.0,
        dt=0.02,
        save_animation=False
    )

if __name__ == "__main__":
    train_robust_controller()
