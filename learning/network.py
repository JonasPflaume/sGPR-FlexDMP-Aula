import torch.nn as nn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLP(nn.Module):
    def __init__(self, input_d, output_d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_d, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, output_d)
        )
        self.net.to(device)
        
    def forward(self, x):
        return self.net(x)

class AccPolicy(nn.Module):
    def __init__(self, input_d, output_d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_d, 16),
            nn.ReLU(),
            nn.Linear(16, 12),
            nn.ReLU(),
            nn.Linear(12, 6),
            nn.ReLU(),
            nn.Linear(6, output_d)
        )
        self.net.to(device)
        
    def forward(self, x):
        return self.net(x)


def setParams(network:torch.nn.Module, decay:float) -> list:
    ''' function to set weight decay
    '''
    params_dict = dict(network.named_parameters())
    params=[]
    weights=[]

    for key, value in params_dict.items():
        if key[-4:] == 'bias':
            params += [{'params':value,'weight_decay':0.0}]
        else:             
            params +=  [{'params': value,'weight_decay':decay}]
    return params
    