import argparse

class CommandLineParser:
    def __init__(self):
        self.data_file = ""
        self.has_gamma = False
        self.gamma = None
        self.has_epsilon = False
        self.epsilon = None
        self.has_iteration = False
        self.iteration = None
        self.should_plot = False

    @staticmethod
    def parse_value_iteration(args):
        parser = argparse.ArgumentParser(description='Parse command line arguments for Value Iteration.')
        parser.add_argument('data_file', type=str, help='Data file path')
        parser.add_argument('-g', '--gamma', type=float, help='Gamma value')
        parser.add_argument('-p', '--plot', action='store_true', help='Should plot')

        parsed_args = parser.parse_args(args)
        result = CommandLineParser()
        result.data_file = parsed_args.data_file
        result.has_gamma = parsed_args.gamma is not None
        result.gamma = parsed_args.gamma
        result.should_plot = parsed_args.plot
        return result

    @staticmethod
    def parse_q_learning(args):
        parser = argparse.ArgumentParser(description='Parse command line arguments for Q-Learning.')
        parser.add_argument('data_file', type=str, help='Data file path')
        parser.add_argument('-g', '--gamma', type=float, help='Gamma value')
        parser.add_argument('-e', '--epsilon', type=float, help='Epsilon value')
        parser.add_argument('-i', '--iteration', type=int, help='Number of iterations')
        parser.add_argument('-p', '--plot', action='store_true', help='Should plot')

        parsed_args = parser.parse_args(args)
        result = CommandLineParser()
        result.data_file = parsed_args.data_file
        result.has_gamma = parsed_args.gamma is not None
        result.gamma = parsed_args.gamma
        result.has_epsilon = parsed_args.epsilon is not None
        result.epsilon = parsed_args.epsilon
        result.has_iteration = parsed_args.iteration is not None
        result.iteration = parsed_args.iteration
        result.should_plot = parsed_args.plot
        return result

# Example usage
# if __name__ == "__main__":
#     import sys
#     # Replace 'value_iteration' or 'q_learning' with actual method names and arguments
#     print(CommandLineParser.parse_value_iteration(sys.argv[1:]))
#     print(CommandLineParser.parse_q_learning(sys.argv[1:]))
