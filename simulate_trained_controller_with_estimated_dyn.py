import numpy as np
import jax.numpy as jnp
import jax
from cartpole_trainer import CartPoleTrainer, create_default_cost_matrices
from simulate_trained_controller import simulate_trained_controller, compare_controllers
from estimate_dyn import main as estimate_dyn_main
from noiseless_dyn import noiseless_dyn
import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation

def train_with_estimated_dynamics():
    """
    Main function implementing the pipeline:
    1. (Optional) Estimate dynamics parameters if not using saved controllers
    2. Train or load both nominal and robust controllers
    3. Simulate and compare both controllers with true parameters
    """
    # Ask user whether to use saved controllers or train new ones
    use_saved = input("Do you want to use both saved controllers? (y/n): ").lower().strip() == 'y'
    
    if use_saved:
        # List available controllers
        saved_dir = "saved_controllers"
        if not os.path.exists(saved_dir):
            print(f"Error: {saved_dir} directory not found!")
            return
            
        saved_files = [f for f in os.listdir(saved_dir) if f.endswith('.pkl')]
        
        if not saved_files:
            print(f"No saved controllers found in {saved_dir}!")
            return
            
        print("\nAvailable nominal controllers:")
        for i, file in enumerate(saved_files):
            if 'nominal' in file:
                print(f"{i+1}. {file}")
                
        print("\nAvailable robust controllers:")
        for i, file in enumerate(saved_files):
            if 'robust' in file:
                print(f"{i+1}. {file}")
        
        # Get user choices
        while True:
            try:
                choice_nominal = int(input("\nEnter the number of the nominal controller to load: "))
                choice_robust = int(input("Enter the number of the robust controller to load: "))
                if 1 <= choice_nominal <= len(saved_files) and 1 <= choice_robust <= len(saved_files):
                    break
                print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter valid numbers.")
        
        filepath_nominal = os.path.join(saved_dir, saved_files[choice_nominal-1])
        filepath_robust = os.path.join(saved_dir, saved_files[choice_robust-1])
        
        # Load controllers and their estimated parameters
        trainer_nominal, trained_params_nominal = CartPoleTrainer.load_controller(filepath_nominal)
        trainer_robust, trained_params_robust = CartPoleTrainer.load_controller(filepath_robust)
        
        # Store the estimated parameters from the loaded controllers
        estimated_params = trainer_nominal.dynamics_params  # or trainer_robust.dynamics_params
        
        print(f"Loaded nominal controller from: {filepath_nominal}")
        print(f"Loaded robust controller from: {filepath_robust}")
        print(f"Using estimated parameters from saved controller: {estimated_params}")
        
    else:
        load_nominal = input("Do you want to load a nominal controller? (y/n): ").lower().strip() == 'y'

        if load_nominal:
            # List available controllers
            saved_dir = "saved_controllers"
            if not os.path.exists(saved_dir):
                print(f"Error: {saved_dir} directory not found!")
                return
            
            saved_files = [f for f in os.listdir(saved_dir) if f.endswith('.pkl')]
            if not saved_files:
                print(f"No saved controllers found in {saved_dir}!")
                return
                
            print("\nAvailable nominal controllers:")
            for i, file in enumerate(saved_files):
                if 'nominal' in file:
                    print(f"{i+1}. {file}")
            # Get user choices
            while True:
                try:
                    choice_nominal = int(input("\nEnter the number of the nominal controller to load: "))
                    if 1 <= choice_nominal <= len(saved_files):
                        break
                    print("Invalid choice. Please try again.")
                except ValueError:
                    print("Please enter valid numbers.")
            
            filepath_nominal = os.path.join(saved_dir, saved_files[choice_nominal-1])
            
            # Load controllers and their estimated parameters
            trainer_nominal, trained_params_nominal = CartPoleTrainer.load_controller(filepath_nominal)
            
            # Store the estimated parameters from the loaded controllers
            estimated_params = trainer_nominal.dynamics_params  # or trainer_robust.dynamics_params
            
            print(f"Loaded nominal controller from: {filepath_nominal}")
            print(f"Using estimated parameters from saved controller: {estimated_params}") 

            continue_training = input("Do you want to continue training the robust controller? (y/n): ").lower().strip() == 'y'
            if continue_training:
                # load the robust controller
                print("\n Available robust controllers:")
                for i, file in enumerate(saved_files):
                    if 'robust' in file:
                        print(f"{i+1}. {file}")
                while True:
                    try:
                        choice_robust = int(input("\nEnter the number of the robust controller to load: "))
                        if 1 <= choice_robust <= len(saved_files):
                            break
                        print("Invalid choice. Please try again.")
                    except ValueError:
                        print("Please enter valid numbers.")
                
                filepath_robust = os.path.join(saved_dir, saved_files[choice_robust-1])
                trainer_robust, trained_params_robust = CartPoleTrainer.load_controller(filepath_robust)

                print("Continuing training...")
                # Get training parameters from user
                try:
                    num_iterations = int(input("Enter number of additional training iterations (default: 30000): ") or "30000")
                    learning_rate = float(input("Enter initial learning rate (default: 0.005): ") or "0.005")
                    scale_ellipsoid = float(input("Enter scale factor for parameter uncertainty (default: 0.1): ") or "0.1")
                    reg_strength = float(input("Enter regularization strength (default: 0.01): ") or "0.01")
                except ValueError:
                    print("Invalid input. Using default values.")
                    num_iterations = 30000
                    learning_rate = 0.005
                    scale_ellipsoid = 0.1
                    reg_strength = 0.01
                print("\nContinuing training with loaded controller...")
                cost_matrices = create_default_cost_matrices()
                trained_params_robust, losses_robust = trainer_robust.train_DR(
                    cost_matrices=cost_matrices,
                    num_iterations=num_iterations,
                    T=300,
                    initial_learning_rate=learning_rate,
                    scale_ellipsoid=scale_ellipsoid, 
                    reg_strength=reg_strength
                )
                
                # Plot training losses
                plt.figure(figsize=(8, 5))
                plt.plot(losses_robust)
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.title('Additional Robust Controller Training Progress')
                plt.ylim(0, 1000)
                plt.grid(True)
                plt.tight_layout()
                plt.savefig('robust_additional_training_loss.png')
                plt.close()
                
                # Save the further trained controller
                save_path = trainer_robust.save_controller(trained_params_robust, suffix='robust_continued')
                print(f"\nFurther trained controller saved to: {save_path}")
            

            else:
                print("\nStep 2: Training robust controller from scratch...")

                try:
                    num_iterations = int(input("Enter number of additional training iterations (default: 30000): ") or "30000")
                    learning_rate = float(input("Enter initial learning rate (default: 0.005): ") or "0.005")
                    scale_ellipsoid = float(input("Enter scale factor for parameter uncertainty (default: 0.1): ") or "0.1")
                    reg_strength = float(input("Enter regularization strength (default: 0.01): ") or "0.01")
                except ValueError:
                    print("Invalid input. Using default values.")
                    num_iterations = 30000
                    learning_rate = 0.005
                    scale_ellipsoid = 0.1
                    reg_strength = 0.01

                trainer_robust = CartPoleTrainer(
                    estimated_params,
                    hidden_layers=[64, 64, 32],
                    noise_std=0.01
                )
                cost_matrices = create_default_cost_matrices()
                trained_params_robust, losses_robust = trainer_robust.train_DR(
                    cost_matrices=cost_matrices,
                    num_iterations=num_iterations,
                    T=300,
                    initial_learning_rate=learning_rate,
                    scale_ellipsoid=scale_ellipsoid, 
                    reg_strength=reg_strength
                )

                # plot training losses of robust controller
                plt.figure(figsize=(12, 5))
                plt.plot(losses_robust)
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.title('Robust Controller Training Progress')
                plt.ylim(0, 1000)
                plt.grid(True)
                plt.tight_layout()
                plt.savefig('training_losses_robust.png')
                plt.close()

                # Save the controllers
                save_path_robust = trainer_robust.save_controller(trained_params_robust, suffix='robust')
                print(f"Robust controller saved to: {save_path_robust}")
        else:
            # 1. Estimate dynamics parameters
            print("\nStep 1: Estimating dynamics parameters...")
            estimated_params, true_params = estimate_dyn_main()
            
            # 2. Train controllers using estimated parameters
            print("\nStep 2: Training controllers with estimated parameters...")
            
            try:
                    num_iterations = int(input("Enter number of additional training iterations (default: 30000): ") or "30000")
                    learning_rate = float(input("Enter initial learning rate (default: 0.005): ") or "0.005")
                    scale_ellipsoid = float(input("Enter scale factor for parameter uncertainty (default: 0.1): ") or "0.1")
                    reg_strength = float(input("Enter regularization strength (default: 0.01): ") or "0.01")
            except ValueError:
                    print("Invalid input. Using default values.")
                    num_iterations = 30000
                    learning_rate = 0.005
                    scale_ellipsoid = 0.1
                    reg_strength = 0.01
            
            # Train nominal controller
            print("Training nominal controller...")
            trainer_nominal = CartPoleTrainer(
                estimated_params,
                hidden_layers=[64, 64, 32],
                noise_std=0.01
            )
            
            cost_matrices = create_default_cost_matrices()
            trained_params_nominal, losses_nominal = trainer_nominal.train(
                cost_matrices=cost_matrices,
                num_iterations=num_iterations,
                T=300,
                initial_learning_rate=learning_rate
            )
            
            # Train robust controller
            print("\nTraining robust controller...")
            trainer_robust = CartPoleTrainer(
                estimated_params,
                hidden_layers=[64, 64, 32],
                noise_std=0.01
            )
            
            trained_params_robust, losses_robust = trainer_robust.train_DR(
                cost_matrices=cost_matrices,
                num_iterations=num_iterations,
                T=300,
                initial_learning_rate=learning_rate,
                scale_ellipsoid=scale_ellipsoid, 
                reg_strength=reg_strength
            )
        
            # Plot training losses
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(losses_nominal)
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('Nominal Controller Training Progress')
            plt.ylim(0, 1000)
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            plt.plot(losses_robust)
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('Robust Controller Training Progress')
            plt.ylim(0, 1000)
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('training_losses_comparison.png')
            plt.close()
            
            # Save the controllers
            save_path_nominal = trainer_nominal.save_controller(trained_params_nominal, suffix='nominal')
            save_path_robust = trainer_robust.save_controller(trained_params_robust, suffix='robust')
            print(f"\nNominal controller saved to: {save_path_nominal}")
            print(f"Robust controller saved to: {save_path_robust}")

    # 3. Simulate with true parameters
    print("\nStep 3: Creating side-by-side animation of controllers...")
    # simulate_trained_controller(trainer_nominal.controller_fn, trained_params_nominal, duration=20.0, dt=0.02, save_animation=False)
    # simulate_trained_controller(trainer_robust.controller_fn, trained_params_robust, duration=20.0, dt=0.02, save_animation=False)
    compare_controllers(trainer_nominal.controller_fn, trained_params_nominal, trainer_robust.controller_fn, trained_params_robust, duration=20.0, dt=0.02, save_animation=False, labels=["Nominal Controller", "Robust Controller"])


if __name__ == "__main__":
    train_with_estimated_dynamics()
