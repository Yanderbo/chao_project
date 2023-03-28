import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size=16, hidden_dim=[32, 16]):
        super(MLP, self).__init__()
        self.input_size, self.hidden_dim = input_size, hidden_dim
        self.input = nn.Linear(self.input_size, self.hidden_dim[0])
        module_list = list()
        for h in range(len(hidden_dim) - 1):
            module_list.append(nn.Linear(self.hidden_dim[h], self.hidden_dim[h + 1]))
            module_list.append(nn.ReLU())
        self.hidden_list = nn.Sequential(*module_list)
        self.predict = nn.Linear(self.hidden_dim[-1], 1)

    def forward(self, x):
        out = self.input(x)
        out = self.hidden_list(out)
        out = self.predict(out) 
        out = out.squeeze(-1)
        return out