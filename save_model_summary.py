import torch
import json
import tiktoken
from model import NERTransformer

def get_model_summary():
    # Load tag map to get num_classes
    with open("tag_map.json", "r") as f:
        tag_map = json.load(f)
    
    num_classes = len(tag_map)
    
    # Parameters from train.py
    encoding = tiktoken.get_encoding("gpt2")
    vocab_size = encoding.n_vocab + 1
    D_MODEL = 128
    NHEAD = 4
    NUM_LAYERS = 2
    DIM_FEEDFORWARD = 256
    MAX_SEQ_LEN = 256

    model = NERTransformer(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        max_seq_length=MAX_SEQ_LEN,
        num_classes=num_classes,
        dropout=0.1,
    )

    summary = str(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = trainable_params + non_trainable_params

    with open("model_summary.txt", "w") as f:
        f.write("--- NER Model Summary ---\n\n")
        f.write(summary + "\n\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Trainable Parameters: {trainable_params:,}\n")
        f.write(f"Non-trainable Parameters: {non_trainable_params:,}\n")

    print(f"Model summary saved to model_summary.txt")
    print(f"Total Parameters: {total_params:,}")

if __name__ == "__main__":
    get_model_summary()
