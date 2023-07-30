import torch
import torch.nn as nn


device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")


class RNNNet(nn.Module):

    def __init__(self, seq_len, hidden_size=40, num_layers=2, batch_size=500):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size=seq_len, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,
                          nonlinearity='relu')
        # self.h0 = torch.zeros(num_layers, batch_size, hidden_size).to(device)
        # print('init h0:', self.h0.shape)
        self.output_layer = nn.Sequential(nn.ReLU(),
                                          nn.Linear(hidden_size, seq_len))

    def forward(self, x):
        # x: (N, L, input_dim)
        # h0: (D*num_layers, batch_size, hidden_size)
        # outputs: (batch_size, L, D*hidden_size)
        # h: (D*num_layers, batch_size, hidden_size)
        # outputs, h = self.rnn(x, self.h0)
        outputs, h = self.rnn(x)
        # print('h:', h.shape)
        outputs = self.output_layer(outputs)
        return outputs
