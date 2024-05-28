class World:
    def __init__(self):
        self.p = []
        self.reward = 0.0
        self.gamma = 0.0
        self.constructed_world = []

    def get_p(self):
        return self.p

    def get_reward(self):
        return self.reward

    def get_gamma(self):
        return self.gamma

    def get_constructed_world(self):
        return self.constructed_world

    def update_constructed_world(self, constructed_world):
        self.constructed_world = constructed_world
    
class Cell:
    def __init__(self, state=' ', reward=0.0, utility=0.0, policy=''):
        self.state = state
        self.reward = reward
        self.utility = utility
        self.policy = policy

class ValueIterationAlgorithm:
    def __init__(self):
        self.saved_state_utilities = []
        self.p = []
        self.reward = 0.0
        self.gamma = 0.0
        self.constructed_world = []
        self.width = 0
        self.height = 0
        self.actions = ['^', '<', '>', 'v']

    @staticmethod
    def is_position_out_of_the_world(x, y, width, height):
        return x < 0 or x >= width or y < 0 or y >= height

    @staticmethod
    def is_position_forbidden(x, y, constructed_world):
        return constructed_world[x][y].state == 'F'

    @staticmethod
    def is_position_terminal(x, y, constructed_world):
        return constructed_world[x][y].state == 'T'

    @staticmethod
    def is_position_special(x, y, constructed_world):
        return constructed_world[x][y].state == 'B'

    @staticmethod
    def calculate_new_position(x, y, dx, dy):
        return x + dx, y + dy

    def update_probability(self, action):
        if action == '^':
            return self.p[0]
        elif action == '<':
            return self.p[1]
        elif action == '>':
            return self.p[2]
        elif action == 'v':
            remaining_prob = 1.0 - sum(self.p)
            return 0 if remaining_prob < 1e-10 else remaining_prob
        else:
            return 0


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
        
        # Mapping action string to corresponding index
        action_index = self.actions.index(action)
        
        return actions[action_index]

    def calculate_new_utility(self, action_utilities, x, y):
        return self.constructed_world[x][y].reward + self.gamma * max(action_utilities)

    def get_best_policy(self, action_utilities):
        max_index = action_utilities.index(max(action_utilities))
        return self.actions[max_index]

    def update_cell_utility(self, x, y, new_utility):
        self.constructed_world[x][y].utility = new_utility

    def update_cell_policy(self, x, y, new_policy):
        self.constructed_world[x][y].policy = new_policy

    def init_saved_state_utilities(self):
        for y in range(self.height):
            for x in range(self.width):
                state_data = {"x": x, "y": y, "utilities": []}
                self.saved_state_utilities.append(state_data)
                self.save_state_utility(x, y)

    def save_state_utility(self, x, y):
        index = y * self.width + x
        self.saved_state_utilities[index]["utilities"].append(self.constructed_world[x][y].utility)

    def calculate_utilities_for_all_actions(self, x, y, action, action_utilities):
        utility = 0.0
        for action_i in self.actions:
            dx, dy = self.update_position_changes(action_i, action)
            new_x, new_y = self.calculate_new_position(x, y, dx, dy)
            p_current = self.update_probability(action_i)
            if self.is_position_out_of_the_world(new_x, new_y, self.width, self.height) or \
               self.is_position_forbidden(new_x, new_y, self.constructed_world):
                utility += p_current * self.constructed_world[x][y].utility
            else:
                utility += p_current * self.constructed_world[new_x][new_y].utility
        action_utilities.append(utility)


    def start(self, world):
        self.p = world.get_p()
        self.reward = world.get_reward()
        self.gamma = world.get_gamma()
        self.constructed_world = world.get_constructed_world()

        self.width = len(self.constructed_world)
        self.height = len(self.constructed_world[0])

        self.init_saved_state_utilities()

        action_utilities = []
        stop_condition = False
        max_delta = float('inf')

        while not stop_condition:
            current_max_delta = 0.0
            for y in range(self.height):
                for x in range(self.width):
                    if self.is_position_terminal(x, y, self.constructed_world) or \
                       self.is_position_forbidden(x, y, self.constructed_world):
                        self.save_state_utility(x, y)
                        continue

                    for action in self.actions:
                        self.calculate_utilities_for_all_actions(x, y, action, action_utilities)

                    new_policy = self.get_best_policy(action_utilities)
                    new_utility = self.calculate_new_utility(action_utilities, x, y)

                    action_utilities.clear()

                    utility_delta = abs(new_utility - self.constructed_world[x][y].utility)
                    if utility_delta > current_max_delta:
                        current_max_delta = utility_delta
                    self.update_cell_utility(x, y, new_utility)
                    self.update_cell_policy(x, y, new_policy)
                    self.save_state_utility(x, y)

            stop_condition = current_max_delta < 0.0001 or stop_condition
            max_delta = max(max_delta, current_max_delta)

        world.update_constructed_world(self.constructed_world)
