# DQLearner.py
#
# DQN RL learner based on PyTorch DQN implementation

import torch as tc
import torch.nn as nn
import torch.optim as optim

class DQLearner(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super(DQLearner, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),      
            nn.Linear(64, output_size)
        )



    def forward(self, x):
        return self.network(x)


    
