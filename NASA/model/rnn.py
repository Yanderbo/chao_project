import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers=2, n_class=1, mode='RNN'):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        if mode == 'RNN':
            self.cell = nn.RNN(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
            self.predict = nn.Linear(hidden_dim, n_class)
        if mode == 'GRU':
            self.cell = nn.GRU(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
            self.predict = nn.Linear(hidden_dim, n_class)
        if mode == 'LSTM':
            self.cell = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
            self.predict = nn.Linear(hidden_dim, n_class)
        if mode == 'BiLSTM':
            self.cell = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
            self.predict = nn.Linear(2 * hidden_dim, n_class)
    def forward(self, x):           # x shape: (batch_size, seq_len, input_size)
        # print(x.size())           # [15, 1, 16]
        out, _ = self.cell(x)
        # print(out.size())         # [15, 1, 128]
        out = self.predict(out)
        # print(out.size())         # [15, 1, 1]      
        y = out[:, -1, :]           # [15, 1] y shape: (batch_size, n_class=1)
        # print(y.size())
        return y

    # def forward(self, x):           # x shape: (batch_size, seq_len, input_size)
    #     out, _ = self.cell(x)
    #     last_hidden_states = out[:, -1]      
    #     y = self.predict(last_hidden_states)            # y shape: (batch_size, n_class=1)
    #     return y
