# ai_snake_py
A reinforcement learning implementation of the classic Snake game using Q-learning with a neural network. The game environment is created using curses for terminal-based gameplay.
### Initial Setup

1. Clone the repository: Clone this repository using git clone.
2. Create Virtual Env: Create a Python Virtual Env venv to download the required dependencies and libraries.
3. Download Dependencies: Download the required dependencies into the Virtual Env venv using pip.

```shell
git clone https://github.com/grisha765/ai_snake_py.git
cd ai_snake_py
python3 -m venv venv
venv/bin/pip install tensorflow
```

### Run Game

1. Start the Game: Start the game from the venv virtual environment.

```shell
venv/bin/python main.py
```

### Features

1. Reinforcement Learning: Utilizes Q-learning with a neural network to train the snake agent.
2. Dynamic Gameplay: The snake learns to navigate and grow in the grid environment.
3. Terminal-based Interface: The game uses curses to render the game in the terminal.
