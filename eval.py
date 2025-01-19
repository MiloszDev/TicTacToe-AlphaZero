import torch
import numpy as np
from game import TicTacToe
from model_builder import ResNet
from alpha_zero import MCTS

tictactoe = TicTacToe()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(tictactoe, 4, 64, device=device)
model.load_state_dict(torch.load('model_2.pt', map_location=device))
model.eval()

args = {
    'C': 2,
    'num_searches': 60,
    'num_iterations': 3,
    'num_selfPlay_iterations': 500,
    'num_epochs': 4,
    'batch_size': 64,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3
}
mcts = MCTS(tictactoe, args, model)

state = tictactoe.get_initial_state()
player = 1

while True:
    print("\nCurrent Board State:")
    print(state)

    valid_moves = tictactoe.get_valid_moves(state)
    print("Valid moves:", [i for i in range(tictactoe.action_size) if valid_moves[i]])

    value, is_terminal = tictactoe.get_value_and_terminated(state, None)
    if np.all(state == 0):
        is_terminal = False
        value = 0

    if is_terminal:
        print("Game Over!")
        if value == 1:
            print(f"Player {player} won!")
        elif value == 0:
            print("It's a draw!")
        break

    if player == 1:
        while True:
            try:
                action = int(input("Enter your move (0-8): "))
                if valid_moves[action] == 1:
                    break
                else:
                    print("Invalid move. Try again.")
            except (ValueError, IndexError):
                print("Invalid input. Enter a number between 0 and 8.")
    else:
        neutral_state = tictactoe.change_perspective(state, player)
        mcts_probs = mcts.search(neutral_state)
        action = np.argmax(mcts_probs)
        print(f"AI chooses action: {action}")

    state = tictactoe.get_next_state(state, action, player)
    player = tictactoe.get_opponent(player)
