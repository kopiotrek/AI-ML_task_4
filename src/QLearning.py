import numpy as np
import random
from World import World

class Cell:
    def __init__(self):
        self.state = ''
        self.policy = ''
        self.utility = 0.0
        self.n = {}
        self.q = {}

class QLearning:
    def __init__(self):
        self.width = 0
        self.height = 0
        self.gamma = 0.0
        self.reward = 0.0
        self.epsilon = 0.0
        self.iteration = 0
        self.is_iteration_defined_by_user = False
        self.p = []
        self.actions = ['^', '<', '>', 'v']
        self.constructed_world = []
        self.saved_state_utilities = []

    @staticmethod
    def is_position_out_of_the_world(x, y, width, height):
        return x < 0 or x >= width or y < 0 or y >= height

    @staticmethod
    def is_position_forbidden(x, y, constructed_world):
        return constructed_world[x][y].state == "F"

    @staticmethod
    def is_position_terminal(x, y, constructed_world):
        return constructed_world[x][y].state == "T"

    def update_probability(self, action):
        if action == '^':
            return self.p[0]
        elif action == '<':
            return self.p[1]
        elif action == '>':
            return self.p[2]
        elif action == 'v':
            return 1.0 - self.p[0] - self.p[1] - self.p[2]
        else:
            return 0

    def update_cell_policy(self, x, y, new_policy):
        self.constructed_world[x][y].policy = new_policy

    def update_cell_utility(self, x, y, new_utility):
        self.constructed_world[x][y].utility = new_utility

    def update_frequency(self, x, y, current_action):
        if current_action in self.constructed_world[x][y].n:
            self.constructed_world[x][y].n[current_action] += 1
        else:
            self.constructed_world[x][y].n[current_action] = 1

    def update_actions(self, desired_orientation):
        if desired_orientation == '^':
            return [(0, 1), (-1, 0), (1, 0), (0, -1)]
        elif desired_orientation == '<':
            return [(-1, 0), (0, -1), (0, 1), (1, 0)]
        elif desired_orientation == '>':
            return [(1, 0), (0, 1), (0, -1), (-1, 0)]
        elif desired_orientation == 'v':
            return [(0, -1), (1, 0), (-1, 0), (0, 1)]
        else:
            return [(0, 0), (0, 0), (0, 0), (0, 0)]

    def update_position_changes(self, action, current_orientation):
        actions = self.update_actions(current_orientation)
        if action == '^':
            return actions[0]
        elif action == '<':
            return actions[1]
        elif action == '>':
            return actions[2]
        elif action == 'v':
            return actions[3]
        else:
            return (0, 0)

    def get_state_reward(self, x, y):
        return self.constructed_world[x][y].reward

    def get_q(self, x, y, current_action):
        return self.constructed_world[x][y].q.get(current_action, 0)

    def get_frequency_of_action(self, x, y, current_action):
        return self.constructed_world[x][y].n.get(current_action, 0)

    def get_best_policy_and_max_q(self, x, y):
        best_policy = ' '
        max_q = float('-inf')
        for policy, q in self.constructed_world[x][y].q.items():
            if q > max_q:
                max_q = q
                best_policy = policy
        return best_policy, max_q

    def calculate_new_position(self, x, y, dx, dy):
        return x + dx, y + dy

    def sum_and_remove_duplicates(self, points):
        sum_map = {}
        for point in points:
            key = (point['x'], point['y'])
            if key in sum_map:
                sum_map[key] += point['p']
            else:
                sum_map[key] = point['p']

        points.clear()

        for key, value in sum_map.items():
            if value!= 0.0:
                points.append({'x': key[0], 'y': key[1], 'p': value})

    def calculate_new_positions_possibilities_for_all_actions(self, x, y, action):
        points = []
        for action_i in self.actions:
            dx, dy = self.update_position_changes(action_i, action)
            new_x, new_y = self.calculate_new_position(x, y, dx, dy)
            p_current = self.update_probability(action_i)

            point = {'x': new_x, 'y': new_y, 'p': p_current}
            if self.is_position_out_of_the_world(new_x, new_y, self.width, self.height) or \
                    self.is_position_forbidden(new_x, new_y, self.constructed_world):
                point['x'] -= dx
                point['y'] -= dy
            points.append(point)
        self.sum_and_remove_duplicates(points)
        return points

    def generate_random_action(self, current_policy):
        if random.random() < self.epsilon or current_policy == ' ':
            return random.choice(['^', '<', '>', 'v'])
        else:
            return current_policy

    def execute_agent_move(self, x, y, possible_moves):
        total_prob = sum([move['p'] for move in possible_moves])
        rand_val = random.uniform(0, total_prob)
        cumulative_prob = 0.0

        for move in possible_moves:
            cumulative_prob += move['p']
            if rand_val <= cumulative_prob:
                return move['x'], move['y']

        return x, y

    def display_progress_bar(self, current_iteration, total_iterations, bar_width=50):
        progress = current_iteration / total_iterations
        filled_width = int(progress * bar_width)

        print(f"  QLearning: [{'=' * filled_width}{'>' if filled_width < bar_width else ''}{' ' * (bar_width - filled_width)}] "
              f"{int(progress * 100)}%\r", end='')
        print("\033[K", end='')  # Clear line

    def init_saved_state_utilities(self):
        for y in range(self.height):
            for x in range(self.width):
                state_data = {'x': x, 'y': y, 'utilities': []}
                self.saved_state_utilities.append(state_data)
                self.save_state_utility(x, y)

    def save_state_utility(self, x, y):
        index = x + y * self.width
        self.saved_state_utilities[index]['utilities'].append(self.constructed_world[x][y].utility)

    def start(self, world):
        self.p = world.get_p()
        self.reward = world.get_reward()
        self.gamma = world.get_gamma()
        self.epsilon = world.get_epsilon()
        self.constructed_world = world.get_constructed_world()

        self.width = len(self.constructed_world)
        self.height = len(self.constructed_world[0])

        if not self.is_iteration_defined_by_user:
            self.iteration = 10000

        self.init_saved_state_utilities()

        for i in range(self.iteration):
            self.display_progress_bar(i + 1, self.iteration)
            x, y = world.get_coordinates_of_state("S")
            current_x = x
            current_y = y
            while True:
                if self.is_position_terminal(current_x, current_y, self.constructed_world):
                    break

                current_action = self.generate_random_action(self.constructed_world[current_x][current_y].policy)
                possible_moves = self.calculate_new_positions_possibilities_for_all_actions(current_x, current_y, current_action)
                new_x, new_y = self.execute_agent_move(current_x, current_y, possible_moves)

                self.update_frequency(current_x, current_y, current_action)

                alpha = 1.0 / self.get_frequency_of_action(current_x, current_y, current_action)
                old_q = self.get_q(current_x, current_y, current_action)

                new_best_policy, new_max_q = self.get_best_policy_and_max_q(new_x, new_y)

                if self.is_position_terminal(new_x, new_y, self.constructed_world):
                    new_max_q = self.get_state_reward(new_x, new_y)

                new_q = self.get_state_reward(current_x, current_y) + self.gamma * new_max_q
                self.constructed_world[current_x][current_y].q[current_action] = old_q + alpha * (new_q - old_q)
                if not self.is_position_terminal(new_x, new_y, self.constructed_world):
                    self.update_cell_policy(new_x, new_y, new_best_policy)

                current_best_policy, current_max_q = self.get_best_policy_and_max_q(current_x, current_y)

                self.update_cell_utility(current_x, current_y, current_max_q)
                current_x = new_x
                current_y = new_y

            for yy in range(self.height):
                for xx in range(self.width):
                    self.save_state_utility(xx, yy)

        print("\n\n")
        world.update_constructed_world(self.constructed_world)

    def set_iteration(self, new_iteration):
        self.iteration = new_iteration
