import argparse
from QLearning import QLearning
from Plotter import Plotter
from World import World

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run Q-Learning Algorithm.')
    parser.add_argument('--data', required=True, help='Path to the data file')
    parser.add_argument('--gamma', type=float, default=1, help='Discount factor gamma')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Exploration rate epsilon')
    parser.add_argument('--iteration', type=int, default=10000, help='Number of iterations for Q-Learning')
    parser.add_argument('--plot', action='store_true', help='Whether to plot the results')

    args = parser.parse_args()

    # Check if data file is empty
    if not args.data:
        print("Error: Data file path is required.")
        return 1

    # Initialize World and q_learning objects
    world = World()
    q_learning = QLearning()

    # Load world parameters from file
    if not world.load_world_parameters_from_file(args.data, True):
        print("Error: Failed to load world parameters from file.")
        return 1

    # Set gamma if provided
    if args.gamma is not None:
        if not world.set_gamma(args.gamma):
            print("Error: Failed to set gamma.")
            return 1

    # Set epsilon if provided
    if args.epsilon is not None:
        world.set_epsilon(args.epsilon)

    # Set iteration count if provided
    if args.iteration is not None:
        q_learning.set_iteration(args.iteration)
        q_learning.is_iteration_defined_by_user = True

    # Print and construct world
    world.print_world_parameters()
    world.construct_world()

    # Run Q-Learning
    q_learning.start(world)

    # Display world and Q-values
    world.display_world()
    world.display_q_values()

    # Plot if requested
    if args.plot:
        Plotter.plot(q_learning.saved_state_utilities)

if __name__ == "__main__":
    main()
