import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, : x.size(1), :]
        return x


class NERTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        nhead,
        num_encoder_layers,
        dim_feedforward,
        max_seq_length,
        num_classes,
        dropout=0.1,
    ):
        super(NERTransformer, self).__init__()
        self.model_type = "Transformer"
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_encoder_layers
        )

        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.classifier.bias.data.zero_()
        self.classifier.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len]
            src_mask: Tensor, shape [seq_len, seq_len] (optional)
            src_key_padding_mask: Tensor, shape [batch_size, seq_len] (optional) - True for padded positions
        Returns:
            output: Tensor, shape [batch_size, seq_len, num_classes]
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        src = self.dropout(src)

        output = self.transformer_encoder(
            src, mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )
        output = self.classifier(output)

        return output


if __name__ == "__main__":
    # Test the model with dummy data
    vocab_size = 100
    d_model = 32
    nhead = 4
    num_encoder_layers = 2
    dim_feedforward = 64
    max_seq_length = 50
    num_classes = 5

    model = NERTransformer(
        vocab_size,
        d_model,
        nhead,
        num_encoder_layers,
        dim_feedforward,
        max_seq_length,
        num_classes,
    )

    # Dummy input
    batch_size = 2
    seq_len = 10
    src = torch.randint(0, vocab_size, (batch_size, seq_len))

    output = model(src)
    print(f"Input shape: {src.shape}")
    print(f"Output shape: {output.shape}")

    # Check if run successful
    assert output.shape == (batch_size, seq_len, num_classes)
    print("Model test passed!")
