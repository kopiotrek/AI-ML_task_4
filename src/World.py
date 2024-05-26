import os
import random
import math
import json
import numpy as np
import sys

class World:

    class Cell:
        def __init__(self):
            self.state = " "  # Name of the state
            self.utility = 0.0  # Utility value with 4 decimal places
            self.policy = ' '  # Policy: <, >, v, ^, or empty
            self.reward = 0.0  # Reward for state

            # Below fields are used in Q learning
            self.q = {
                '^': 0.0,
                '<': 0.0,
                '>': 0.0,
                'v': 0.0
            }

            self.n = {
                '^': 0,
                '<': 0,
                '>': 0,
                'v': 0
            }

    def __init__(self):
        self.width_x = 0  # Defines the horizontal world size
        self.height_y = 0  # Defines the vertical world size
        self.start_x = 0  # Specifies the horizontal coordinate of the start state
        self.start_y = 0  # Specifies the vertical coordinate of the start state
        self.p = [0.0, 0.0, 0.0]  # Uncertainty distribution
        self.reward = 0.0  # Default reward parameter
        self.gamma = 0.0  # Discounting parameter
        self.epsilon = 0.0  # Exploration parameter
        self.terminal_states = []  # Terminal states (X,Y) and their reward
        self.special_states = []  # Special states (X,Y) and their reward
        self.forbidden_states = []  # Forbidden states (X,Y)
        self.constructed_world = []

    def check_file_validity(self, file_name):
        if not os.path.exists(file_name):
            print(f"  Error: File {file_name} does not exist.", file=sys.stderr)
            return False

        has_w, has_p, has_r, has_t = False, False, False, False
        with open(file_name, 'r') as infile:
            for line in infile:
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == 'W':
                    if len(parts) != 3:
                        print(f"  Error: Invalid world dimensions definition after W option in file {file_name}", file=sys.stderr)
                        return False
                    has_w = True
                elif parts[0] == 'P':
                    if len(parts) != 4:
                        print(f"  Error: Invalid uncertainty distribution definition after P option in file {file_name}", file=sys.stderr)
                        return False
                    has_p = True
                elif parts[0] == 'R':
                    if len(parts) != 2:
                        print(f"  Error: Invalid reward value definition after R option in file {file_name}", file=sys.stderr)
                        return False
                    has_r = True
                elif parts[0] == 'T':
                    if len(parts) != 4:
                        print(f"  Error: Invalid terminal state definition after T option in file {file_name}", file=sys.stderr)
                        return False
                    has_t = True

        if not has_w:
            print(f"  Error: Mandatory option W is missing in file {file_name}", file=sys.stderr)
            return False
        if not has_p:
            print(f"  Error: Mandatory option P is missing in file {file_name}", file=sys.stderr)
            return False
        if not has_r:
            print(f"  Error: Mandatory option R is missing in file {file_name}", file=sys.stderr)
            return False
        if not has_t:
            print(f"  Error: Mandatory option T is missing in file {file_name}", file=sys.stderr)
            return False

        return True

    def check_parameters_validity(self):
        # Check if start state is defined within world dimensions
        if self.start_x <= 0 or self.start_x > self.width_x or self.start_y <= 0 or self.start_y > self.height_y:
            print(f"  Error: Start state ({self.start_x},{self.start_y}) is outside world dimensions", file=sys.stderr)
            return False

        # Check if p1, p2, p3 >= 0.0 <= 1.0 and p1+p2+p3 <= 1.0
        if self.p[0] < 0.0 or self.p[0] > 1.0:
            print("  Error: Invalid uncertainty specified for p1, should be in the range [0.0, 1.0]", file=sys.stderr)
            return False
        elif self.p[1] < 0.0 or self.p[1] > 1.0:
            print("  Error: Invalid uncertainty specified for p2, should be in the range [0.0, 1.0]", file=sys.stderr)
            return False
        elif self.p[2] < 0.0 or self.p[2] > 1.0:
            print("  Error: Invalid uncertainty specified for p3, should be in the range [0.0, 1.0]", file=sys.stderr)
            return False
        elif (self.p[0] + self.p[1] + self.p[2]) > 1.0:
            print("  Error: Uncertainty distribution sums to more than 1.0.", file=sys.stderr)
            return False

        # Check if gamma is in range (0.0, 1.0]
        if self.gamma <= 0.0 or self.gamma > 1.0:
            print("  Error: Gamma should be in the range (0.0, 1.0]", file=sys.stderr)
            return False

        # Check if terminal states are defined within world dimensions
        for x_t, y_t, _ in self.terminal_states:
            if x_t <= 0 or x_t > self.width_x or y_t <= 0 or y_t > self.height_y:
                print(f"  Error: Terminal state ({x_t},{y_t}) is outside world dimensions", file=sys.stderr)
                return False

        # Check if special states are defined within world dimensions
        for x_s, y_s, _ in self.special_states:
            if x_s <= 0 or x_s > self.width_x or y_s <= 0 or y_s > self.height_y:
                print(f"  Error: Special state ({x_s},{y_s}) is outside world dimensions", file=sys.stderr)
                return False

        # Check if forbidden states are defined within world dimensions
        for x_f, y_f in self.forbidden_states:
            if x_f <= 0 or x_f > self.width_x or y_f <= 0 or y_f > self.height_y:
                print(f"  Error: Forbidden state ({x_f},{y_f}) is outside world dimensions", file=sys.stderr)
                return False

        return True

    def load_world_parameters_from_file(self, file_name, is_q_learning):
        if not self.check_file_validity(file_name):
            return False

        with open(file_name, 'r') as infile:
            is_start_in_file = False
            for line in infile:
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == 'W':
                    self.width_x, self.height_y = int(parts[1]), int(parts[2])
                elif parts[0] == 'S':
                    self.start_x, self.start_y = int(parts[1]), int(parts[2])
                    is_start_in_file = True
                elif parts[0] == 'P':
                    self.p = [float(parts[1]), float(parts[2]), float(parts[3])]
                elif parts[0] == 'R':
                    self.reward = float(parts[1])
                elif parts[0] == 'G':
                    self.gamma = float(parts[1])
                elif parts[0] == 'E':
                    self.epsilon = float(parts[1])
                elif parts[0] == 'T':
                    x, y, reward = int(parts[1]), int(parts[2]), float(parts[3])
                    self.terminal_states.append((x, y, reward))
                elif parts[0] == 'B':
                    x, y, reward = int(parts[1]), int(parts[2]), float(parts[3])
                    self.special_states.append((x, y, reward))
                elif parts[0] == 'F':
                    x, y = int(parts[1]), int(parts[2])
                    self.forbidden_states.append((x, y))

        if not is_start_in_file and not is_q_learning:
            self.start_x = 1
            self.start_y = 1
            print(f"  Info: S option isn't present in {file_name} file. It has been set to default (1,1) value")
        elif not is_start_in_file and is_q_learning:
            available_states = [
                (x, y) for x in range(1, self.width_x + 1) for y in range(1, self.height_y + 1)
                if (x, y) not in self.forbidden_states and not self.is_in_terminal_states((x, y)) and not self.is_in_special_states((x, y))
            ]

            if available_states:
                random_state = random.choice(available_states)
                self.start_x, self.start_y = random_state
                print(f"  Info: S option isn't present in {file_name} file. It has been selected randomly at ({self.start_x}, {self.start_y})")

        return self.check_parameters_validity()

    def is_in_terminal_states(self, state):
        for terminal_state in self.terminal_states:
            if terminal_state[:2] == state:
                return True
        return False

    def is_in_special_states(self, state):
        for special_state in self.special_states:
            if special_state[:2] == state:
                return True
        return False

    def print_world_parameters(self):
        print("\n  World Parameters:")
        print(f"  Width X: {self.width_x}")
        print(f"  Height Y: {self.height_y}")
        print(f"  Start X: {self.start_x}")
        print(f"  Start Y: {self.start_y}")
        print(f"  Uncertainty Distribution P: {self.p[0]}(^), {self.p[1]}(<), {self.p[2]}(>)")
        print(f"  Reward: {self.reward}")
        print(f"  Discounting Parameter Gamma: {self.gamma}")
        print(f"  Exploration Parameter Epsilon: {self.epsilon}")
        print("  (T) Terminal States:", end="")
        for x, y, reward in self.terminal_states:
            print(f" ({x},{y},{reward})", end="")
        print("\n  (B) Special States:", end="")
        for x, y, reward in self.special_states:
            print(f" ({x},{y},{reward})", end="")
        print("\n  (F) Forbidden States:", end="")
        for x, y in self.forbidden_states:
            print(f" ({x},{y})", end="")
        print("\n")

    def set_gamma(self, gamma):
        if gamma <= 0.0 or gamma > 1.0:
            print("  Error: Gamma should be in the range (0.0, 1.0]", file=sys.stderr)
            return False
        self.gamma = gamma
        return True

    def display_world(self):
        if not self.constructed_world:
            return

        height = len(self.constructed_world[0])
        width = len(self.constructed_world)

        # Determine the maximum width needed for each cell, considering utility values with 4 decimal places
        max_chars = max(len(f"{cell.utility:.4f}") for row in self.constructed_world for cell in row) + 1

        # Generate horizontal line
        line = "=" * ((max_chars + 3) * width + 1) + "\n"

        # Print the grid
        print(line, end="")
        for j in range(height):
            print("║", end="")
            for i in range(width):
                print(f" {self.constructed_world[i][j].utility:>{max_chars}.4f} ║", end="")
            print("\n", end="")
            print(line, end="")

    def display_q_values(self):
        if not self.constructed_world:
            return

        height = len(self.constructed_world[0])
        width = len(self.constructed_world)

        directions = ['^', '<', '>', 'v']

        # Determine the maximum width needed for each cell, considering Q-values with 4 decimal places
        max_chars = max(len(f"{cell.q[dir]:.4f}") for row in self.constructed_world for cell in row for dir in directions) + 2

        # Generate horizontal line
        line = "=" * ((max_chars + 4) * width + 5) + "\n"

        # Print the Q-value grid
        print("\n")
        for row in range(height, 0, -1):
            print(line, end="")
            for direction in directions:
                print("║", end="")
                for i in range(width):
                    q_value = self.constructed_world[i][row-1].q[direction]
                    print(f" {direction} {q_value:>{max_chars}.4f} ║", end="")
                print("\n", end="")
            print(line, end="")
        print("\n")


    def construct_world(self):
        world = [[self.Cell() for _ in range(self.height_y)] for _ in range(self.width_x)]

        for x in range(1, self.width_x + 1):
            for y in range(1, self.height_y + 1):
                cell = self.Cell()

                cell.reward = self.reward

                for direction in cell.q:
                    cell.q[direction] = self.reward

                for (tx, ty, tr) in self.terminal_states:
                    if x == tx and y == ty:
                        cell.state = "T"
                        cell.utility = tr
                        cell.reward = cell.utility
                        for direction in cell.q:
                            cell.q[direction] = cell.reward

                for (sx, sy, sr) in self.special_states:
                    if x == sx and y == sy:
                        cell.state = "B"
                        cell.reward = sr

                for (fx, fy) in self.forbidden_states:
                    if x == fx and y == fy:
                        cell.state = "F"
                        cell.reward = 0.0
                        for direction in cell.q:
                            cell.q[direction] = cell.reward

                world[x - 1][y - 1] = cell

        world[self.start_x - 1][self.start_y - 1].state = "S"
        self.constructed_world = world

    def get_coordinates_of_state(self, target_state):
        for i in range(len(self.constructed_world)):
            for j in range(len(self.constructed_world[i])):
                if self.constructed_world[i][j].state == target_state:
                    return (i, j)
        return (0, 0)
    
    def set_epsilon(self, new_epsilon):
        self.epsilon = new_epsilon

    def get_p(self):
        return(self.p)
    
    def get_reward(self):
        return(self.reward)
    
    def get_gamma(self):
        return(self.gamma)
    
    def get_epsilon(self):
        return(self.epsilon)
    
    def get_constructed_world(self):
        return(self.constructed_world)
    
    def update_constructed_world(self, new_world):
        self.constructed_world = new_world