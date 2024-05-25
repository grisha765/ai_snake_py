import random
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import time
import curses

# Определение констант
GRID_SIZE = 20
GRID_WIDTH = 24  # Учитывая границы
GRID_HEIGHT = 24  # Учитывая границы
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
BLANK = ' '
SNAKE_BODY = '@'
FOOD = '*'
WALL = '#'  # Символ границы

# Код игры "Змейка"
class SnakeGame:
    def __init__(self):
        self.snake = [(5, 5)]
        self.direction = RIGHT
        self.food = self.new_food()
        self.score = 0

    def new_food(self):
        food = (random.randint(1, GRID_WIDTH - 2), random.randint(1, GRID_HEIGHT - 2))  # Измененный диапазон, чтобы питание не спавнилось на границах
        while food in self.snake:
            food = (random.randint(1, GRID_WIDTH - 2), random.randint(1, GRID_HEIGHT - 2))
        return food

    def move(self):
        new_head = (self.snake[0][0] + (self.direction == RIGHT) - (self.direction == LEFT),
                    self.snake[0][1] + (self.direction == DOWN) - (self.direction == UP))
        if new_head[0] <= 0 or new_head[0] >= GRID_WIDTH - 1 or new_head[1] <= 0 or new_head[1] >= GRID_HEIGHT - 1 or new_head in self.snake:  # Учитываем границы
            return False
        self.snake.insert(0, new_head)
        if self.snake[0] == self.food:
            self.score += 1
            self.food = self.new_food()
        else:
            self.snake.pop()
        return True

    def get_state(self):
        state = np.zeros((GRID_HEIGHT, GRID_WIDTH))
        for segment in self.snake:
            state[segment[1]][segment[0]] = 1
        state[self.food[1]][self.food[0]] = -1
        return state.flatten()

    def step(self, action):
        if action == UP and self.direction != DOWN:
            self.direction = UP
        elif action == DOWN and self.direction != UP:
            self.direction = DOWN
        elif action == LEFT and self.direction != RIGHT:
            self.direction = LEFT
        elif action == RIGHT and self.direction != LEFT:
            self.direction = RIGHT
        if not self.move():
            return -1, True  # Reward, Done
        return 0, False

# Машинное обучение (Q-learning)
class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_shape=(self.state_size,), activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state.reshape(1, -1))  # Преобразование входных данных в тензор
        return np.argmax(act_values[0])  # Returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state.reshape(1, -1))[0]))  # Преобразование входных данных в тензор
            target_f = self.model.predict(state.reshape(1, -1))  # Преобразование входных данных в тензор
            target_f[0][action] = target
            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)  # Преобразование входных данных в тензор
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Основной цикл игры
def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(1)
    agent = QLearningAgent(GRID_WIDTH * GRID_HEIGHT, 4)
    game = SnakeGame()
    batch_size = 32
    episodes = 1000

    for e in range(episodes):
        state = game.get_state()
        done = False
        while not done:
            action = agent.act(state)
            reward, done = game.step(action)
            next_state = game.get_state()
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            # Вывод состояния игры и действия бота
            print("Action:", action)
            print("Score:", game.score)
            print("Snake:", game.snake)
            print("Food:", game.food)
            print()
            
            # Отрисовка игры
            stdscr.clear()
            for y in range(GRID_HEIGHT):
                for x in range(GRID_WIDTH):
                    if x == 0 or x == GRID_WIDTH - 1 or y == 0 or y == GRID_HEIGHT - 1:  # Рисуем границы
                        stdscr.addstr(y, x, WALL)
                    elif (x, y) in game.snake:
                        stdscr.addstr(y, x, SNAKE_BODY)
                    elif (x, y) == game.food:
                        stdscr.addstr(y, x, FOOD)
                    else:
                        stdscr.addstr(y, x, BLANK)
            stdscr.refresh()
            
            # Задержка, чтобы уменьшить скорость игры
            time.sleep(0.1)
            
        print("episode: {}/{}, score: {}, e: {:.2}"
              .format(e, episodes, game.score, agent.epsilon))
        game = SnakeGame()

if __name__ == "__main__":
    curses.wrapper(main)
