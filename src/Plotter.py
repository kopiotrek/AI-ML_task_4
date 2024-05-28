import matplotlib.pyplot as plt

class ValueIterationAlgorithm:
    class StateData:
        def __init__(self):
            self.utilities = []
            self.x = 0
            self.y = 0

class QLearning:
    class StateData:
        def __init__(self):
            self.utilities = []
            self.x = 0
            self.y = 0

class Plotter:
    @staticmethod
    def plot(data):
        colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
            "#bcbd22", "#17becf", "indigo", "lime", "blue", "olive", "darkorchid", "black"
        ]

        # Accessing the utilities from the first dictionary in the list
        iterations = list(range(len(data[0]['utilities'])))

        plt.figure(figsize=(12.8, 7.2))

        for i, state_utility in enumerate(data):
            utilities = state_utility['utilities']
            label = f"({state_utility['x'] + 1},{state_utility['y'] + 1})"
            plt.plot(iterations, utilities, color=colors[i % len(colors)], label=label)

        plt.title("The value iteration algorithm")
        plt.xlabel("Number of iterations")
        plt.ylabel("Utility estimates")
        plt.legend(loc="lower right")
        plt.show()