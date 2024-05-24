import argparse
from ValueIterationAlgorithm import ValueIterationAlgorithm
from Plotter import Plotter
from World import World

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run Value Iteration Algorithm.')
    parser.add_argument('--data', required=True, help='Path to the data file')
    parser.add_argument('--gamma', type=float, default=0.9, help='Discount factor gamma')
    parser.add_argument('--plot', action='store_true', help='Whether to plot the results')

    args = parser.parse_args()

    # Check if data file is empty
    if not args.data:
        print("Error: Data file path is required.")
        return 1

    # Initialize World object
    world = World()
    value_iteration_algorithm = ValueIterationAlgorithm()

    # Load world parameters from file
    if not world.load_world_parameters_from_file(args.data, False):
        print("Error: Failed to load world parameters from file.")
        return 1

    # Set gamma if provided
    if args.gamma is not None:
        if not world.set_gamma(args.gamma):
            print("Error: Failed to set gamma.")
            return 1

    # Print and construct world
    world.print_world_parameters()
    world.construct_world()

    # Run Value Iteration Algorithm
    value_iteration_algorithm.start(world)

    # Display world
    world.display_world()

    # Plot if requested
    if args.plot:
        Plotter.plot(value_iteration_algorithm.saved_state_utilities)

if __name__ == "__main__":
    main()
