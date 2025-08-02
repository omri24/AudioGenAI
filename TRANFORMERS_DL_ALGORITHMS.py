import numpy as np
import torch
from torch import nn
import math

#positional encoding taken from https://medium.com/@hunter-j-phillips/positional-encoding-7a93db4109e6

class PositionalEncoding(nn.Module):
  def __init__(self, d_model: int, dropout: float = 0.1, max_length: int = 5000):
    """
    Args:
      d_model:      dimension of embeddings
      dropout:      randomly zeroes-out some of the input
      max_length:   max sequence length
    """
    # inherit from Module
    super().__init__()

    # initialize dropout
    self.dropout = nn.Dropout(p=dropout)

    # create tensor of 0s
    pe = torch.zeros(max_length, d_model)

    # create position column
    k = torch.arange(0, max_length).unsqueeze(1)

    # calc divisor for positional encoding
    div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
    )

    # calc sine on even indices
    pe[:, 0::2] = torch.sin(k * div_term)

    # calc cosine on odd indices
    pe[:, 1::2] = torch.cos(k * div_term)

    # add dimension
    pe = pe.unsqueeze(0)

    # buffers are saved in state_dict but not trained by the optimizer
    self.register_buffer("pe", pe)

  def forward(self, x):
    """
    Args:
      x:        embeddings (batch_size, seq_length, d_model)

    Returns:
                embeddings + positional encodings (batch_size, seq_length, d_model)
    """
    # add positional encoding to the embeddings
    x = x + self.pe[:, : x.size(1)]

    # perform dropout
    return self.dropout(x)

class Transformers_Model(nn.Module):
    def __init__(self, vocab_size, num_encoder_layers, nhead, num_decoder_layers, hidden_dim, output_dim, dropout=0.5):
            super(Transformers_Model, self).__init__()
            self.emb = nn.Embedding(vocab_size, hidden_dim)
            self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
            self.transformer= nn.Transformer(
                d_model=hidden_dim,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True
            )
            self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, src, tgt):
        src = self.emb(src)
        tgt = self.emb(tgt)
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)

        output = self.transformer(src, tgt)
        output = self.fc(output)
        return torch.transpose(output, 2, 1)


