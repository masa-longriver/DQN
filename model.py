import torch
import torch.nn as nn
import pfrl

class QFunction(nn.Module):
    def __init__(self, config):
        super().__init__()
        obs_size  = config['env']['n_cand'] * 2
        n_actions = config['env']['n_cand']
        model_config = config['model']

        self.fc1 = nn.Linear(obs_size, model_config['hidden_size1'])
        self.fc2 = nn.Linear(model_config['hidden_size1'], model_config['hidden_size2'])
        self.fc3 = nn.Linear(model_config['hidden_size2'], n_actions)

        self.act = nn.SiLU()  # Swish
    
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)

        return pfrl.action_value.DiscreteActionValue(x)