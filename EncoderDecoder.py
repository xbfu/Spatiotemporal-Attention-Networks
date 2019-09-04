import torch
from torch import nn
import torch.cuda as cuda
from Module import ScaledDotProductAttention, MultiHeadAttention, PositionalWiseFeedForward

# Hyper Parameters
TIME_STEP = 12
HIDDEN_SIZE = 32
INPUT_SIZE = 6
d_model = 512
heads = 8


class EncoderLayer(nn.Module):

    def __init__(self, model_dim=d_model, num_heads=heads, ffn_dim=2048, dropout=0.0):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs):

        # self attention
        context, attention = self.attention(inputs, inputs, inputs)

        # feed forward network
        output = self.feed_forward(context)

        return output, attention


class Encoder(nn.Module):

    def __init__(self, num_layers=1, model_dim=d_model, num_heads=heads, ffn_dim=2048, dropout=0.0):
        super(Encoder, self).__init__()
        self.model_dim = model_dim

        self.encoder_layers = nn.ModuleList(
          [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)])
        self.linear = nn.Linear(1, model_dim)
        self.out = nn.Linear(TIME_STEP, 1)

        self.relu = nn.ReLU(True)

    def forward(self, inputs):
        output = self.linear(inputs)

        for encoder in self.encoder_layers:
            output, attention = encoder(output)

        x = output[:, 0].unsqueeze(1)

        return x


class Encoder2(nn.Module):
    def __init__(self):
        super(Encoder2, self).__init__()

        self.rnn = nn.RNN(
            input_size=d_model,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, h_state = self.rnn(x, h_state)
        return r_out, h_state


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )
        self.relu = nn.ReLU()
        self.out = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, input, hidden, ):
        output, hidden = self.rnn(input, hidden)
        out = self.out(output[:, -1, :])
        return out, hidden, output