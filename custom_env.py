import random
import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import GradientTape, expand_dims
from tqdm import tqdm

class EnvironmentWrapper(gym.Env):
    def __init__(self):
        super(EnvironmentWrapper, self).__init__()
        # Define the observation space as a binary array with 145 elements
        self.observation_space = spaces.MultiBinary(145)

        # Define the action space as discrete actions {0, 1, 2, 3}
        self.action_space = spaces.Discrete(4)  # 4 actions: 0, 1, 2, 3

        self.board = game()

    def reset(self, seed = None):
        # Reset the environment and return the initial observation
        self.board.restart()
        self.board.move_enemies()
        obs = np.array(self.board.get_data(), dtype=np.int8)
        return obs, {}

    def step(self, action):
        r = self.board.move_agent(action)
        observation = np.array(self.board.get_data(), dtype=np.int8)
        done = not(self.board.is_alive())
        info = {}  # Additional information (optional)

        return observation, r, done, done, info


test = False


class cell:
    '''
    class to create a single cell
    '''

    def __init__(self, x_coord=0, y_coord=0, state="", number=-1):
        self.position = (x_coord, y_coord)
        self.number = number  # to enumerate all cells in a board. The O-perimeter is not numbered
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.state = state
        self.nb = []

    def update(self, new_state):
        self.state = new_state


class game:
    '''
    Main class for the game.
    standard: Will enable the standard board (in terms of size, obstacles, number of enemies, position of agent and enemies).
    O_prob: For non-standard random board, this is the probability that a cell will be an obstacle.
    '''

    def __init__(self, standard=True, x_dim=25, y_dim=25, O_prob=0.2, Food_number=15, Enemy_number=4, I_pos=(19, 13),
                 Energy_units=40, reinforcement_natural_death=-0.7):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.O_prob = O_prob
        self.Food_number = Food_number
        self.food_found = 0
        self.food = []
        self.Enemy_number = Enemy_number
        self.enemies = []
        self.I = 0
        self.energy = Energy_units
        self.board = [[] for i in range(x_dim + 2)]  # contains a list per row of the board
        self.alive = True
        self.time = 0
        self.collision = False  # indicates whether or not the previous attempted move caused a collision with an obstacle.
        self.prev = [0, 0, 0, 0]  # indicates the last attempted move.
        self.reinforcement_natural_death = reinforcement_natural_death
        for x in range(1, x_dim + 1):
            self.board[x].append(cell(x, 0, 'O'))
            for y in range(1, y_dim + 1):
                self.board[x].append(cell(x, y, '', (x - 1) * y_dim + y))
            self.board[x].append(cell(x, y_dim + 1, 'O'))
        for x in [0, x_dim + 1]:
            for y in range(y_dim + 2):
                self.board[x].append(cell(x, y, 'O'))
        self.board[I_pos[0]][I_pos[1]].state = 'I'
        self.I = self.board[I_pos[0]][I_pos[1]]
        if standard:
            for i in [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 20), (1, 21), (1, 22), (1, 23), (1, 24),
                      (1, 25), (2, 1), (2, 25), (3, 1), (3, 4), (3, 5), (3, 9), (3, 10), (3, 12), (3, 13), (3, 14),
                      (3, 16), (3, 17), (3, 21), (3, 22), (3, 25), (4, 1), (4, 3), (4, 4), (4, 5), (4, 9), (4, 10),
                      (4, 12), (4, 14), (4, 16), (4, 17), (4, 21), (4, 22), (4, 23), (4, 25), (5, 1), (5, 3), (5, 4),
                      (5, 5), (5, 21), (5, 22), (5, 23), (5, 25), (6, 1), (6, 25), (9, 3), (9, 4), (9, 9), (9, 10),
                      (9, 11), (9, 12), (9, 14), (9, 15), (9, 16), (9, 17), (9, 22), (9, 23), (10, 3), (10, 4), (10, 9),
                      (10, 10), (10, 11), (10, 15), (10, 16), (10, 17), (10, 22), (10, 23), (11, 9), (11, 10), (11, 16),
                      (11, 17), (12, 3), (12, 4), (12, 9), (12, 17), (12, 22), (12, 23), (13, 3), (13, 13), (13, 23)]:
                self.board[i[0]][i[1]].state = 'O'
                self.board[26 - i[0]][i[1]].state = 'O'
            for i in [(2, 13), (7, 7), (7, 13), (7, 19)]:
                self.board[i[0]][i[1]].state = 'E'
                self.enemies.append(self.board[i[0]][i[1]])
        else:
            '''
            Will follow. This allows for non-standard board design.
            '''
        left = Food_number
        while left != 0:
            pos = (random.randint(1, x_dim), random.randint(1, y_dim))
            if self.board[pos[0]][pos[1]].state == '':
                self.board[pos[0]][pos[1]].state = '\$'
                self.food.append(self.board[pos[0]][pos[1]])
                left -= 1
        for x in range(1, x_dim + 1):
            for y in range(1, y_dim + 1):
                self.board[x][y].nb = [self.board[x - 1][y], self.board[x + 1][y], self.board[x][y - 1],
                                       self.board[x][y + 1]]
        # remains to set the nb of the perimeter cells (needed for sensors)
        for y in range(1, y_dim + 1):
            self.board[0][y].nb = [self.board[0][y - 1], self.board[0][y + 1]]
            self.board[x_dim + 1][y].nb = [self.board[0][y - 1], self.board[0][y + 1]]
        for x in range(1, x_dim + 1):
            self.board[x][0].nb = [self.board[x - 1][0], self.board[x + 1][0]]
            self.board[x][y_dim + 1].nb = [self.board[x - 1][0], self.board[x + 1][0]]
        self.board[0][0].nb = [self.board[0][1], self.board[1][0]]
        self.board[0][y_dim + 1].nb = [self.board[0][y_dim], self.board[1][y_dim + 1]]
        self.board[x_dim + 1][0].nb = [self.board[x_dim][0], self.board[x_dim + 1][1]]
        self.board[x_dim + 1][y_dim + 1].nb = [self.board[x_dim + 1][y_dim], self.board[x_dim][y_dim + 1]]
        self.visual = [self.vis()]

    def is_alive(self):
        return self.alive

    def update_board(self):
        self.move_agent()
        self.move_enemies()
        self.visual.append(self.vis())

    def simulate(self):
        while self.alive and self.Food_number > 0:
            self.update_board()

    def move_agent(self, direction=0):
        '''
        New_pos is the direction, 0 corresponding to north, 1 to east, etc.
        Get reinforcement via return statement.
        '''
        self.time += 1
        new_pos = direction
        if not (self.alive) or self.energy == 0:
            self.alive = False
            return -1
        if new_pos == 0:
            new_pos = self.board[self.I.x_coord - 1][self.I.y_coord]
        elif new_pos == 1:
            new_pos = self.board[self.I.x_coord][self.I.y_coord + 1]
        elif new_pos == 2:
            new_pos = self.board[self.I.x_coord + 1][self.I.y_coord]
        elif new_pos == 3:
            new_pos = self.board[self.I.x_coord][self.I.y_coord - 1]
        else:
            raise Exception('Not a valid move.')
        if new_pos.x_coord == self.I.x_coord - 1:
            self.prev = [1, 0, 0, 0]
        elif new_pos.x_coord == self.I.x_coord + 1:
            self.prev = [0, 1, 0, 0]
        elif new_pos.y_coord == self.I.y_coord - 1:
            self.prev = [0, 0, 1, 0]
        else:
            self.prev = [0, 0, 0, 1]
        if new_pos.state == '\$':
            self.energy += 14
            self.food_found += 1
            self.I.state = ''
            self.I = new_pos
            self.I.state = 'I'
            self.food.remove(self.I)
            self.Food_number -= 1
            self.collision = False
            return 0.4
        if new_pos.state == 'E':
            self.energy -= 1
            self.alive = False
            self.collision = False
            return -1
        if new_pos.state == 'O':
            self.energy -= 1
            self.collision = True
            if self.energy == 0:
                self.alive = False
                return self.reinforcement_natural_death
            return 0
        if new_pos.state == '':
            self.collision = False
            self.energy -= 1
            if self.energy == 0:
                self.alive = False
                return self.reinforcement_natural_death
            self.I.state = ''
            self.I = new_pos
            self.I.state = 'I'
            return 0

    def T_dist(self, enemy):
        '''
        calculates the T-distance between an enemy and the agent. enemy is a cell object.
        '''
        dist = ((enemy.x_coord - self.I.x_coord) ** 2 + (enemy.y_coord - self.I.y_coord) ** 2) ** 0.5
        if (dist <= 4):
            return 15 - dist
        elif dist <= 15:
            return 9 - dist / 2
        else:
            return 1

    def w_angle(self, enemy, newpos):
        '''
        calculates the angle between a possible new position newpos of an enemy and the agent. Arguments are cell objects.
        '''
        a = (newpos.x_coord - enemy.x_coord, newpos.y_coord - enemy.y_coord)
        b = (self.I.x_coord - enemy.x_coord, self.I.y_coord - enemy.y_coord)
        dot = (a[0] * b[0] + a[1] * b[1]) / ((a[0] ** 2 + a[1] ** 2) ** 0.5 * (b[0] ** 2 + b[1] ** 2) ** 0.5)
        angle = math.degrees(math.acos(dot))

        return (180 - abs(angle)) / 180

    def move_enemies(self):
        new_enemies = []
        for enemy in self.enemies:
            if random.random() <= 0.8:
                choices = [x for x in enemy.nb if x.state == '' or x.state == 'I']
                prob = []
                for choice in choices:
                    prob.append(math.exp(0.33 * self.w_angle(enemy, choice) * self.T_dist(enemy)))
                if len(choices) == 0:
                    new_enemies.append(enemy)
                    continue
                newpos = random.choices(choices, weights=prob)[0]
                if newpos == self.I:
                    self.alive = False
                newpos.state = 'E'
                enemy.state = ''
                new_enemies.append(newpos)
            else:
                new_enemies.append(enemy)
        self.enemies = new_enemies

    def x_sensor(self, cell):
        '''
        returns the 5 neighboring cells in a list.
        '''
        return ([cell] + [nb for nb in cell.nb])

    def o_sensor(self, cell):
        l = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if 0 <= cell.x_coord + i <= self.x_dim + 1 and 0 <= cell.y_coord + j <= self.y_dim + 1:
                    l.append(self.board[cell.x_coord + i][cell.y_coord + j])
        return l

    def y_sensor(self, cell):
        l = self.o_sensor(cell)
        for i in [-2, 2]:
            if 0 <= cell.x_coord + i <= self.x_dim + 1:
                l.append(self.board[cell.x_coord + i][cell.y_coord])
            if 0 <= cell.y_coord + i <= self.y_dim + 1:
                l.append(self.board[cell.x_coord][cell.y_coord + i])
        return l

    def obstacle_sensor(self, cell):
        '''
        returns (1,0)-information about obstacles in cells that lie in the obstacle sensor array, starting with the cell on the top of the array
        and then moving to the bottom, where each row is sorted left to right.
        All sensors are using the global action representation.
        '''
        l = []
        for i in range(-4, 5):
            for j in range(-4 + abs(i), 5 - abs(i)):
                if (i, j) == (0, 0):
                    continue
                else:
                    if 0 <= cell.x_coord + i <= self.x_dim + 1 and 0 <= cell.y_coord + j <= self.y_dim + 1:
                        if self.board[cell.x_coord + i][cell.y_coord + j].state == 'O':
                            l.append(1)
                        else:
                            l.append(0)
                    else:
                        l.append(1)
        return l

    def enemy_sensor(self, cell):
        '''
        same as above. return list ordered s.t. cells close to the agent rather come first, i.e. first X-sensors, then inner O-sensors, then outer O-sensors.
        '''
        x = cell.x_coord
        y = cell.y_coord
        Xpos = []
        for i in range(-2, 3):
            for j in range(-2 + abs(i), 3 - abs(i)):
                if (i, j) != (0, 0):
                    Xpos.append((x + i, y + j))
        Opos = []
        for i in [-2, 2]:
            for j in [-2, 2]:
                Opos += [(x + i, y + j)]
        for i in [-4, 4]:
            for j in [-2, 0, 2]:
                Opos += [(x + i, y + j), (x + j, y + i)]
        for i in [-6, 6]:
            Opos += [(x, y + i), (x + i, y)]
        l = []
        for pos in Xpos:
            if 0 <= pos[0] <= self.x_dim + 1 and 0 <= pos[1] <= self.y_dim + 1:
                xsens = self.x_sensor(self.board[pos[0]][pos[1]])
                if 'E' in [nb.state for nb in xsens]:
                    l.append(1)
                else:
                    l.append(0)
            else:
                l.append(0)
        for pos in Opos:
            if 0 <= pos[0] <= self.x_dim + 1 and 0 <= pos[1] <= self.y_dim + 1:
                osens = self.o_sensor(self.board[pos[0]][pos[1]])
                if 'E' in [nb.state for nb in osens]:
                    l.append(1)
                else:
                    l.append(0)
            else:
                l.append(0)
        return l

    def food_sensor(self, cell):
        '''
        as above.
        '''
        x = cell.x_coord
        y = cell.y_coord
        l = []
        Xpos = []
        for i in range(-2, 3):
            for j in range(-2 + abs(i), 3 - abs(i)):
                if (i, j) != (0, 0):
                    Xpos.append((x + i, y + j))
        Opos = []
        for i in [-2, 2]:
            for j in [-2, 2]:
                Opos += [(x + i, y + j)]
        for i in [-4, 4]:
            for j in [-2, 0, 2]:
                Opos += [(x + i, y + j), (x + j, y + i)]
        for i in [-6, 6]:
            Opos += [(x, y + i), (x + i, y)]
        Ypos = []
        for i in [2, 4, 6, 8]:
            Ypos += [(x + i, y + 10 - i), (x + i, y - 10 + i), (x - i, y + 10 - i), (x - i, y - 10 + i)]
        Ypos += [(x - 10, y), (x + 10, y), (x, y - 10), (x, y + 10)]
        for pos in Xpos:
            if 0 <= pos[0] <= self.x_dim + 1 and 0 <= pos[1] <= self.y_dim + 1:
                xsens = self.x_sensor(self.board[pos[0]][pos[1]])
                if '\$' in [nb.state for nb in xsens]:
                    l.append(1)
                else:
                    l.append(0)
            else:
                l.append(0)
        for pos in Opos:
            if 0 <= pos[0] <= self.x_dim + 1 and 0 <= pos[1] <= self.y_dim + 1:
                osens = self.o_sensor(self.board[pos[0]][pos[1]])
                if '\$' in [nb.state for nb in osens]:
                    l.append(1)
                else:
                    l.append(0)
            else:
                l.append(0)
        for pos in Ypos:
            if 0 <= pos[0] <= self.x_dim + 1 and 0 <= pos[1] <= self.y_dim + 1:
                ysens = self.y_sensor(self.board[pos[0]][pos[1]])
                if '\$' in [nb.state for nb in ysens]:
                    l.append(1)
                else:
                    l.append(0)
            else:
                l.append(0)
        return l

    def energy_sensor(self):
        '''
        returns the 16 bit energy sensor input for the nn. Each sensor stands for an energy level of up to 4 more energy than the previous one, e.g.
        if the energy lies between 1 and 4, the first sensor is activated; the second one instead if it lies between 5 and 8, etc.
        '''
        l = [0 for i in range(16)]
        if self.energy > 0:
            l[min(15, (self.energy - 1) // 4)] = 1
        return l

    def collision_sensor(self):
        return [0] if self.collision else [1]

    def prev_move_sensor(self):
        return self.prev

    def get_data(self, direction=0):
        '''
        returns all relevant data for the nn as a binary list.
        '''
        return self.turn_input_even_more(self.enemy_sensor(self.I) + self.food_sensor(self.I) + self.obstacle_sensor(
            self.I) + self.energy_sensor() + self.prev_move_sensor() + self.collision_sensor(), direction)

    def turn_obstacle_data(self, data):
        '''
        turns the data by 90 degrees clockwise, i.e. instead of the merit of moving north, the nn will compute the merit of moving east. can be applied
        to data repeatedly for other directions.
        '''
        l = [24, 16, 23, 31, 9, 15, 22, 30, 36, 4, 8, 14, 21, 29, 35, 39, 1, 3, 7, 13, 28, 34, 38, 40, 2, 6, 12, 20, 27,
             33, 37, 5, 11, 19, 26, 32, 10, 18, 25, 17]
        turned_data = []
        for i in l:
            turned_data.append(data[i - 1])
        return turned_data

    def turn_enemy_data(self, data):
        '''
        as above
        '''
        l = [8, 4, 7, 11, 1, 3, 10, 12, 2, 6, 9, 5, 14, 16, 13, 15, 24, 21, 26, 19, 28, 17, 18, 27, 20, 25, 22, 23, 30,
             31, 32, 29]
        turned_data = []
        for i in l:
            turned_data.append(data[i - 1])
        return turned_data

    def turn_food_data(self, data):
        '''
        as above
        '''
        l = [8, 4, 7, 11, 1, 3, 10, 12, 2, 6, 9, 5, 14, 16, 13, 15, 24, 21, 26, 19, 28, 17, 18, 27, 20, 25, 22, 23, 30,
             31, 32, 29, 46, 48, 45, 47, 42, 44, 41, 43, 38, 40, 37, 39, 34, 36, 33, 35, 52, 51, 49, 50]
        turned_data = []
        for i in l:
            turned_data.append(data[i - 1])
        return turned_data

    def turn_prev_move(self, move):
        if move == [0, 0, 0, 0]:
            return [0, 0, 0, 0]
        elif move == [1, 0, 0, 0]:
            return [0, 0, 1, 0]
        elif move == [0, 1, 0, 0]:
            return [0, 0, 0, 1]
        elif move == [0, 0, 1, 0]:
            return [0, 1, 0, 0]
        elif move == [0, 0, 0, 1]:
            return [1, 0, 0, 0]
        else:
            raise Exception('Not a move')

    def turn_input_data(self, data):
        '''
        turns the 145 bit input data by 90 degrees.
        '''
        return self.turn_enemy_data(data[:32]) + self.turn_food_data(data[32:84]) + self.turn_obstacle_data(
            data[84:124]) + data[124:140] + self.turn_prev_move(data[140:144]) + [data[144]]

    def turn_input_even_more(self, data, n):
        '''
        turns the input n times clockwise.
        '''
        for i in range(n):
            data = self.turn_input_data(data)
        return data

    def restart(self, standard=True, x_dim=25, y_dim=25, O_prob=0.2, Food_number=15, Enemy_number=4, I_pos=(19, 13),
                Energy_units=40):  # restarts the board into standard setup.
        self.__init__(standard, x_dim, y_dim, O_prob, Food_number, Enemy_number, I_pos, Energy_units,
                      self.reinforcement_natural_death)

    def vis(self):
        '''
        Writes the current state of the board into a tex table.
        '''
        if self.alive:
            status='Alive'
        else:
            status='Dead'
        string="\\begin{table}[h!]\n \\begin{adjustbox}{max width=\\textwidth}\n \\begin{tabular}{"
        string=string+28*"c "+"} \n"
        for i in self.board:
            for j in i:
                string+=j.state
                string+=' &'

            string+="\\\\ \n"
        string+="\\end{tabular}\n \\end{adjustbox}\n \\end{table} Energy: "+str(self.energy)+'\n'+status
        return string


class Q_Agent:
    def __init__(self, discount_factor=0.9, temperature=0.01, H=30, eta=0.001, board=game(), n=None):
        '''
        An TD learning Agent is initialized with
        discount factor - for TD learning
        temerature - action selection, the smaller means actions are chosen according to policy network and higher means at random
        H_e, H_p - hidden layer units
        eta_e, eta_p - learningrate Adam
        board - a game is passed on
        eval_n, policy_n - load saved models
        '''
        self.discount_factor = discount_factor
        self.temperature = temperature
        self.eta = eta
        self.rounds_trained = 0  # only needed for teaching

        if n is None:
            self.net = keras.Sequential([
                layers.Dense(units=H, activation=self.sigmoid, input_shape=[145]),
                layers.Dense(units=1, activation=self.sigmoid)
            ])
        else:
            self.net = keras.models.load_model(n)

        # 0 gives Nourth, 1 E, 2 S, 3 W

        self.optimizer = keras.optimizers.Adam(learning_rate=self.eta)

        self.loss_funktion = keras.losses.Huber()

        self.board = board

    def sigmoid(self, x):
        '''
        activation function for NN
        '''
        return (keras.activations.sigmoid(x) - 0.5) * 2

    def select_action(self, show_prob=False, comp_max=False):
        '''
        action selection function
        Output: chosen direction and input turned according to that direction.
        '''
        prob = []
        for i in range(4):
            x = np.exp(self.net(expand_dims(self.board.get_data(i), 0)) / self.temperature)
            prob.append(x[0, 0])
        if show_prob:
            print(prob)
        chosen_dir = random.choices([0, 1, 2, 3], weights=prob)[0]
        if comp_max:
            return (prob.index(max(prob)), self.board.get_data(prob.index(max(prob))))
        return (chosen_dir, self.board.get_data(chosen_dir))

    def train(self, rounds=10, temperature_l=None):
        '''
        train agents with
        rounds - amount of training rounds
        analyse - save analyse data
        temperature_l - list of temeratures of length rounds, if None temperature is set to 0.01
        '''
        real_reinforcement_list = np.zeros([rounds])
        lifetime = np.zeros([rounds])
        food_found = np.zeros([rounds])
        energy = np.zeros([rounds])
        for i in tqdm(range(rounds)):
            real_reinforcement = 0
            if temperature_l is None:
                # self.temperature= max(min(10/(i+1),0.3),0.01)
                self.temperature = 0.01
            else:
                self.temperature = temperature_l[i]

            # play moves until agent dies
            while (self.board.is_alive()):
                self.board.move_enemies()
                x = self.board.get_data()
                direction, turned_x = self.select_action()

                r = self.board.move_agent(direction=direction)

                real_reinforcement = r + self.discount_factor * real_reinforcement

                loss_funktion = self.loss_funktion

                l = []
                for j in range(4):
                    l.append(self.net(expand_dims(self.board.get_data(j), 0))[0][0])

                u_prime = np.array([[r]]) + self.discount_factor * max(l)

                # Backprop net
                with GradientTape() as tape:
                    old_u = self.net(expand_dims(turned_x, 0))
                    loss_value = loss_funktion(u_prime, old_u)

                # Update the weights of net to minimize the loss value.
                gradients = tape.gradient(loss_value, self.net.trainable_weights)
                self.optimizer.apply_gradients(zip(gradients, self.net.trainable_weights))

            real_reinforcement_list[i] = real_reinforcement
            lifetime[i] = self.board.time
            food_found[i] = self.board.food_found
            energy[i] = self.board.energy

            self.board.restart()

        return (real_reinforcement_list, lifetime, food_found, energy)

    def get_plot_data(self, rounds=300):

        found_list = []
        test_env = [game() for i in range(50)]

        l = []
        for j in range(50):
            self.board = game(food_Pos=test_env[j].food_Pos)
            l.append(self.run_no_vis()[2])
        found_list.append(sum(l) / 50)

        for i in tqdm(range(rounds // 20)):
            self.train(rounds=20)
            l = []
            for j in range(50):
                self.board = game(food_Pos=test_env[j].food_Pos)
                l.append(self.run_no_vis()[2])
            found_list.append(sum(l) / 50)

        return found_list

    def get_plot_data_teaching(self, rounds=300, teaching=True, prio=False, onpolicy=False, pl=0.01):
        found_list = []
        test_env = [game() for i in range(50)]

        l = []
        for j in range(50):
            self.board = game(food_Pos=test_env[j].food_Pos)
            l.append(self.run_no_vis()[2])
        found_list.append(sum(l) / 50)

        for i in tqdm(range(rounds // 20)):
            if not onpolicy:
                self.train_teaching(rounds=20, teaching=teaching, prio=prio)
            else:
                self.train_teaching_on_policy(rounds=20, teaching=teaching, prio=prio, pl=pl)
            l = []
            for j in range(50):
                self.board = game(food_Pos=test_env[j].food_Pos)
                l.append(self.run_no_vis()[2])
            found_list.append(sum(l) / 50)

        return found_list

    def get_avg_r(self, rounds = 10):
        '''
        runs agent with temperature 0.01 and returns average reward and food collected
        '''
        self.temperature = 0.01
        total_reward = 0
        total_food = 0
        for i in range(rounds):
            reward = 0
            food = 0
            while (self.board.is_alive()):
                self.board.move_enemies()
                # self.board.visual.append(self.board.vis())
                x = self.board.get_data()

                direction, _ = self.select_action(comp_max=True)
                r = self.board.move_agent(direction=direction)
                if r == 0.4:
                    food += 1
                reward += r
            total_reward += reward
            total_food += food
            self.board.restart()
        total_reward /= rounds
        total_food /= rounds

        return total_reward, total_food

    def run(self):
        '''
        runs agent with temperature 0.01 and returns a visual
        '''
        vis = []
        self.temperature = 0.01
        while (self.board.is_alive()):
            self.board.move_enemies()
            # self.board.visual.append(self.board.vis())
            x = self.board.get_data()
            e = self.net(expand_dims(x, 0))

            direction, _ = self.select_action(comp_max=True)
            r = self.board.move_agent(direction=direction)

            vis.append(self.board.vis())

        self.board.restart()
        return vis

    def choose_experience(self, m):  # Appendix B from the paper.
        w = min(3, 1 + 0.02 * m)
        rand = random.random()
        return int(m * math.log(1 + rand * (math.e ** w - 1)) / w)

    def teaching_prob(self, round):
        return (2 / 5 * math.e ** (-round / 100) + 1 / 10)

    def replay_number(self, round):
        if round < 14:
            return 12
        if round < 26:
            return 11
        if round < 38:
            return 10
        if round < 51:
            return 9
        if round < 76:
            return 8
        if round < 101:
            return 7
        if round < 151:
            return 6
        if round < 201:
            return 5
        return 4

    def train_teaching_on_policy(self, rounds=10, temperature_l=None, teaching=True, prio=False, pl=0.01):
        '''
        only replays actions that are on policy, i.e. probability of >=pl
        if PER is performed, this doesn't make a difference, i.e. with PER there is no differentiation between policy/non policy actions
        train agents with
        rounds - amount of training rounds
        analyse - save analyse data
        temperature_l - list of temeratures of length rounds, if None temperature is set to 0.01
        incorporates experience replay from the most recont 100 actions and teaching
        experiences are always saved as [old state (not turned),direction,reinforcement,new state (not turned)] (+[td error] potentially)
        If teaching=False, then only experience replay will be performed.
        If prio=True, then prioritized exp. replay will be performed instead of regular exp. replay.
        '''
        recent_exp = []

        real_reinforcement_list = np.zeros([rounds])
        lifetime = np.zeros([rounds])
        food_found = np.zeros([rounds])
        energy = np.zeros([rounds])
        for i in range(rounds):
            real_reinforcement = 0
            if temperature_l is None:
                # self.temperature= max(min(10/(i+1),0.3),0.01)
                self.temperature = 0.01
            else:
                self.temperature = temperature_l[i]

            # play moves until agent dies
            while (self.board.is_alive()):
                self.board.move_enemies()
                x = self.board.get_data()
                direction, turned_x = self.select_action()

                r = self.board.move_agent(direction=direction)

                real_reinforcement = r + self.discount_factor * real_reinforcement

                loss_funktion = self.loss_funktion

                l = []
                for j in range(4):
                    l.append(self.net(expand_dims(self.board.get_data(j), 0))[0][0])

                u_prime = np.array([[r]]) + self.discount_factor * max(l)

                # Backprop net
                with GradientTape() as tape:
                    old_u = self.net(expand_dims(turned_x, 0))
                    loss_value = loss_funktion(u_prime, old_u)

                # Update the weights of net to minimize the loss value.
                gradients = tape.gradient(loss_value, self.net.trainable_weights)
                self.optimizer.apply_gradients(zip(gradients, self.net.trainable_weights))

                recent_exp.append([x, direction, r, self.board.get_data(), abs(loss_value)])
                if len(recent_exp) > 100:
                    recent_exp = recent_exp[1:]

            real_reinforcement_list[i] = real_reinforcement
            lifetime[i] = self.board.time
            food_found[i] = self.board.food_found
            energy[i] = self.board.energy

            if not prio:

                for h in range(self.replay_number(
                        self.rounds_trained)):  # replays past experiences. replays only policy actions.
                    all_off_policy = False
                    not_allowed = []
                    done = False
                    while not done:
                        k = self.choose_experience(len(recent_exp) - len(not_allowed))
                        exp = recent_exp[
                            k + len([x for x in not_allowed if x <= k + len([y for y in not_allowed if y <= x])])]
                        prob = []
                        for i in range(4):
                            x = np.exp(
                                self.net(expand_dims(self.board.turn_input_even_more(exp[0], i), 0)) / self.temperature)
                            prob.append(x[0, 0])
                        if prob[exp[1]] / sum(prob) >= pl:
                            self.train_experience(exp)
                            done = True
                        else:
                            not_allowed.append(
                                k + len([x for x in not_allowed if x <= k + len([y for y in not_allowed if y <= x])]))
                            if len(not_allowed) == len(recent_exp):
                                done = True
                                print("No policy actions left")
                                all_off_policy = True
                    if all_off_policy == True:
                        break

            if prio:

                probs = [(recent_exp[t][-1] + 0.001) ** 0.7 for t in range(len(recent_exp))]
                w_list = [(len(recent_exp) * probs[t] / sum(probs)) ** (-0.5 - 0.5 * self.rounds_trained / rounds) for t
                          in range(len(recent_exp))]  # contains the weights w_i; missing '/max w_i'
                for h in range(self.replay_number(self.rounds_trained)):
                    exp = random.choices(recent_exp, weights=probs)[0]
                    t = recent_exp.index(exp)
                    new_delta = self.train_experience(exp, w_list[t] / max(w_list))
                    recent_exp[t][-1] = new_delta
                    probs[t] = (recent_exp[t][-1] + 0.001) ** 0.7
                    w_list[t] = (len(recent_exp) * probs[t] / sum(probs)) ** (-0.5 - 0.5 * self.rounds_trained / rounds)

            if teaching:

                prob = self.teaching_prob(self.rounds_trained)
                lists = [l1, l2, l3, l4, l5, l6, l7]
                l = random.choice(lists)
                for exp in l:
                    if prob >= random.random():
                        self.train_experience(exp)

            self.rounds_trained += 1
            self.board.restart()

        return (real_reinforcement_list, lifetime, food_found, energy)

    def train_teaching(self, rounds=10, temperature_l=None, teaching=True, prio=False):
        '''
        train agents with
        rounds - amount of training rounds
        analyse - save analyse data
        temperature_l - list of temeratures of length rounds, if None temperature is set to 0.01
        incorporates experience replay from the most recont 100 actions and teaching
        experiences are always saved as [old state (not turned),direction,reinforcement,new state (not turned)] (+[td error] potentially)
        If teaching=False, then only experience replay will be performed.
        If prio=True, then prioritized exp. replay will be performed instead of regular exp. replay.
        '''
        recent_exp = []

        real_reinforcement_list = np.zeros([rounds])
        lifetime = np.zeros([rounds])
        food_found = np.zeros([rounds])
        energy = np.zeros([rounds])
        for i in range(rounds):
            real_reinforcement = 0
            if temperature_l is None:
                # self.temperature= max(min(10/(i+1),0.3),0.01)
                self.temperature = 0.01
            else:
                self.temperature = temperature_l[i]

            # play moves until agent dies
            while (self.board.is_alive()):
                self.board.move_enemies()
                x = self.board.get_data()
                direction, turned_x = self.select_action()

                r = self.board.move_agent(direction=direction)

                real_reinforcement = r + self.discount_factor * real_reinforcement

                loss_funktion = self.loss_funktion

                l = []
                for j in range(4):
                    l.append(self.net(expand_dims(self.board.get_data(j), 0))[0][0])

                u_prime = np.array([[r]]) + self.discount_factor * max(l)

                # Backprop net
                with GradientTape() as tape:
                    old_u = self.net(expand_dims(turned_x, 0))
                    loss_value = loss_funktion(u_prime, old_u)

                # Update the weights of net to minimize the loss value.
                gradients = tape.gradient(loss_value, self.net.trainable_weights)
                self.optimizer.apply_gradients(zip(gradients, self.net.trainable_weights))

                recent_exp.append([x, direction, r, self.board.get_data(), abs(loss_value)])
                if len(recent_exp) > 100:
                    recent_exp = recent_exp[1:]

            real_reinforcement_list[i] = real_reinforcement
            lifetime[i] = self.board.time
            food_found[i] = self.board.food_found
            energy[i] = self.board.energy

            if not prio:

                for h in range(self.replay_number(self.rounds_trained)):  # replays past experiences
                    k = self.choose_experience(len(recent_exp))
                    self.train_experience(recent_exp[k])

            if prio:

                probs = [(recent_exp[t][-1] + 0.001) ** 0.7 for t in range(len(recent_exp))]
                w_list = [(len(recent_exp) * probs[t] / sum(probs)) ** (-0.5 - 0.5 * self.rounds_trained / rounds) for t
                          in range(len(recent_exp))]  # contains the weights w_i; missing '/max w_i'
                for h in range(self.replay_number(self.rounds_trained)):
                    exp = random.choices(recent_exp, weights=probs)[0]
                    t = recent_exp.index(exp)
                    new_delta = self.train_experience(exp, w_list[t] / max(w_list))
                    recent_exp[t][-1] = new_delta
                    probs[t] = (recent_exp[t][-1] + 0.001) ** 0.7
                    w_list[t] = (len(recent_exp) * probs[t] / sum(probs)) ** (-0.5 - 0.5 * self.rounds_trained / rounds)

            if teaching:

                prob = self.teaching_prob(self.rounds_trained)
                lists = [l1, l2, l3, l4, l5, l6, l7]
                l = random.choice(lists)
                for exp in l:
                    if prob >= random.random():
                        self.train_experience(exp)

            self.rounds_trained += 1
            self.board.restart()

        return (real_reinforcement_list, lifetime, food_found, energy)

    def train_experience(self, experience, weight=1):
        '''
        train agent on one experience [old state,direction,reinforcement,new state]
        '''

        x = experience[0]
        direction, turned_x = experience[1], self.board.turn_input_even_more(x, experience[1])

        r = experience[2]

        loss_funktion = self.loss_funktion

        l = []
        for j in range(4):
            l.append(self.net(expand_dims(self.board.turn_input_even_more(experience[3], j), 0))[0][0])

        u_prime = np.array([[r]]) + self.discount_factor * max(l)

        # Backprop net
        with GradientTape() as tape:
            old_u = self.net(expand_dims(turned_x, 0))
            loss_value = loss_funktion(u_prime, old_u) * weight

        # Update the weights of net to minimize the loss value.
        gradients = tape.gradient(loss_value, self.net.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.net.trainable_weights))

        return (abs(loss_value))

    def run_no_vis(self, temperature=0.01, visual=False):
        '''
        runs agent
        Output:
        real reinforcemnt
        time lasted
        food found
        energy left at the end
        visual
        '''
        self.temperature = temperature
        real_reinforcement = 0
        vis = []
        while (self.board.is_alive()):
            self.board.move_enemies()

            x = self.board.get_data()
            e = self.net(expand_dims(x, 0))

            direction, _ = self.select_action(comp_max=True)
            r = self.board.move_agent(direction=direction)
            real_reinforcement = r + self.discount_factor * real_reinforcement
            if visual:
                vis.append(self.board.vis())

        stats = (real_reinforcement, self.board.time, self.board.food_found, self.board.energy, vis)
        self.board.restart()
        return stats


def multiple_agents_comp(n=10, rounds=300, teaching=False, prio=False, onpolicy=True, pl=0.01):
    l = []
    for i in tqdm(range(n)):
        a = Q_Agent()
        found_list = a.get_plot_data_teaching(rounds=rounds, teaching=teaching, prio=prio, onpolicy=onpolicy, pl=pl)
        l.append(found_list)
        print(found_list)
    minmaxlist = []
    avglist = []
    for i in range(len(l[0])):
        minvalue = min([l[j][i] for j in range(len(l))])
        maxvalue = max([l[j][i] for j in range(len(l))])
        minmaxlist.append([minvalue, maxvalue])
        avglist.append(sum([l[j][i] for j in range(len(l))]) / len(l))

    return (minmaxlist, avglist, l)