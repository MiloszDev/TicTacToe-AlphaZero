import math
import torch
import random
import numpy as np
from tqdm import trange
from torch.nn import functional as F

class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visit_count=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        self.children = []
        self.visit_count = visit_count
        self.value_sum = 0

    def is_fully_expanded(self):
        return len(self.children) > 0

    def select(self):
        return max(self.children, key=self.get_ucb)

    def get_ucb(self, child):
        q_value = 0 if child.visit_count == 0 else 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior

    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.game.get_next_state(self.state.copy(), action, 1)
                child_state = self.game.change_perspective(child_state, -1)
                self.children.append(Node(self.game, self.args, child_state, self, action, prob))
        return self.children[-1]

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        if self.parent:
            self.parent.backpropagate(self.game.get_opponent_value(value))


class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.inference_mode()
    def search(self, state):
        root = Node(self.game, self.args, state, visit_count=1)

        policy, _ = self.model(torch.tensor(self.game.get_encoded_state(state), device=self.model.device).unsqueeze(0))
        policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)

        valid_moves = self.game.get_valid_moves(state)
        policy *= valid_moves
        policy /= np.sum(policy)
        root.expand(policy)

        for _ in range(self.args['num_searches']):
            node = root
            while node.is_fully_expanded():
                node = node.select()

            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)

            if not is_terminal:
                policy, value = self.model(torch.tensor(self.game.get_encoded_state(node.state), device=self.model.device).unsqueeze(0))
                policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves
                policy /= np.sum(policy)

                value = value.item()
                node.expand(policy)

            node.backpropagate(value)

        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        return action_probs / np.sum(action_probs)


class AlphaZero:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)

    def selfPlay(self):
        memory = []
        player = 1
        state = self.game.get_initial_state()

        while True:
            neutral_state = self.game.change_perspective(state, player)
            action_probs = self.mcts.search(neutral_state)

            memory.append((neutral_state, action_probs, player))

            temperature_action_probs = action_probs ** (1 / self.args['temperature'])
            action = np.random.choice(self.game.action_size, p=temperature_action_probs / temperature_action_probs.sum())

            state = self.game.get_next_state(state, action, player)
            value, is_terminal = self.game.get_value_and_terminated(state, action)

            if is_terminal:
                return [(self.game.get_encoded_state(hist_neutral_state), hist_action_probs, self.game.get_opponent_value(value) if hist_player != player else value)
                        for hist_neutral_state, hist_action_probs, hist_player in memory]

            player = self.game.get_opponent(player)

    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            batch = memory[batchIdx: batchIdx + self.args['batch_size']]
            state, policy_targets, value_targets = zip(*batch)

            state_tensor = torch.tensor(np.array(state), dtype=torch.float32, device=self.model.device)
            policy_targets_tensor = torch.tensor(np.array(policy_targets), dtype=torch.float32, device=self.model.device)
            value_targets_tensor = torch.tensor(np.array(value_targets).reshape(-1, 1), dtype=torch.float32, device=self.model.device)

            out_policy, out_value = self.model(state_tensor)
            policy_loss = F.cross_entropy(out_policy, policy_targets_tensor)
            value_loss = F.mse_loss(out_value, value_targets_tensor)

            self.optimizer.zero_grad()
            (policy_loss + value_loss).backward()
            self.optimizer.step()

    def learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []

            self.model.eval()
            for _ in trange(self.args['num_selfPlay_iterations']):
                memory.extend(self.selfPlay())

            self.model.train()
            for _ in trange(self.args['num_epochs']):
                self.train(memory)

            torch.save(self.model.state_dict(), f"./models/model_{iteration}.pt")
