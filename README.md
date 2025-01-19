# TicTacToe - AlphaZero

This project implements a Tic-Tac-Toe AI powered by an AlphaZero-inspired algorithm. The AI trains itself through self-play and uses a combination of Monte Carlo Tree Search (MCTS) and deep neural networks to achieve strong gameplay performance.

## Features

- **Self-Play**: The AI plays games against itself to learn optimal strategies.
- **Monte Carlo Tree Search (MCTS)**: Guides decision-making with simulated gameplay.
- **Deep Neural Networks**: Evaluates board states and guides the MCTS.
- **Flexible Training**: Train the AI from scratch or continue training from a saved model.

## Repository Structure

- **`eval.py`**: The main entry point to evaluate the trained model. Use this to play against the AI or test its performance.
- **`alpha_zero.py`**: Core implementation of the AlphaZero algorithm, including MCTS and policy/value updates.
- **`game.py`**: Defines the Tic-Tac-Toe game logic, including the rules and board representation.
- **`model_builder.py`**: Contains the neural network architecture used for predicting policies and values.
- **`train.py`**: Handles the training pipeline, including self-play, data generation, and model updates.

## Getting Started

### Prerequisites

- Python 3.7+
- Required Python packages (install using `pip install -r requirements.txt`):
    - NumPy
    - TensorFlow or PyTorch (depending on the implementation in `model_builder.py`)
    - Other utilities (refer to the `requirements.txt` file for the complete list)

### Installation

1. Clone the repository:
    
    ```
    git clone https://github.com/yourusername/tictactoe-alphazero.git
    cd tictactoe-alphazero
    ```
    
2. Install the dependencies:
    
    ```
    pip install -r requirements.txt
    ```
    

### Running the Project

### Playing Against the AI

Run the `eval.py` script to play a game against the trained AI:

```
python eval.py
```

### Training the AI

To train the AI from scratch, use the `train.py` script:

```
python train.py
```

This will generate self-play data, update the neural network model, and improve the AI over time.

### Modifying the Game or Model

- Adjust game rules or board size in `game.py`.
- Modify the neural network architecture in `model_builder.py`.

## How It Works

1. **Self-Play**: The AI generates gameplay data by playing against itself, guided by the MCTS.
2. **Neural Network**: The network predicts the policy (best moves) and value (win probability) for a given board state.
3. **Training**: The model is updated using gameplay data to improve its predictions.
4. **MCTS**: During gameplay, MCTS simulates possible outcomes to guide the AI's decisions.
