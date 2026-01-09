import torch
import torch.nn.functional as F
import json
import tiktoken
from model import NERTransformer
import sys


def interactive_inference():
    # Load metadata
    try:
        with open("tag_map.json", "r") as f:
            tag_map = json.load(f)
        encoding = tiktoken.get_encoding("gpt2")
    except FileNotFoundError:
        print(
            "Error: Required files (tag_map.json) not found. Please train the model first."
        )
        return

    inv_tag_map = {v: k for k, v in tag_map.items()}
    PAD_ID = encoding.n_vocab
    vocab_size = encoding.n_vocab + 1

    # Model Hyperparameters (must match train.py)
    D_MODEL = 128
    NHEAD = 4
    NUM_LAYERS = 2
    DIM_FEEDFORWARD = 256
    MAX_SEQ_LEN = 256
    NUM_CLASSES = len(tag_map)

    # Device
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Initialize and load model
    model = NERTransformer(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        max_seq_length=MAX_SEQ_LEN,
        num_classes=NUM_CLASSES,
    ).to(device)

    try:
        model.load_state_dict(torch.load("ner_model.pth", map_location=device))
        model.eval()
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Error: ner_model.pth not found. Please train the model first.")
        return

    print("\n" + "=" * 60)
    print("NER INFERENCE TOOL - TIKTOKEN EDITION")
    print("Enter a sentence to analyze entities (Diseases/Medications).")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 60 + "\n")

    DEFAULT_EXAMPLE = "The patient, a 62-year-old female with a history of hypertension and diabetes type 2, was admitted for acute pneumonia. We initiated treatment with amoxicillin and monitored for signs of sepsis."

    while True:
        text = input("\nInput Sentence (press Enter for default): ").strip()
        if not text:
            text = DEFAULT_EXAMPLE
            print(f'Using default: "{text}"')

        if text.lower() in ["quit", "exit"]:
            break

        # Tokenization matches training
        token_ids = encoding.encode(text)
        subwords = [encoding.decode([tid]) for tid in token_ids]

        # Prepare input tensor
        input_tensor = torch.tensor([token_ids]).to(device)

        # Inference
        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=-1)[0]

        # Get Top 3 predictions per token
        top_probs, top_indices = torch.topk(probs, k=min(3, NUM_CLASSES), dim=-1)

        print(f'\nAnalysis for: "{text}"')
        print("-" * 100)
        print(
            f"{'Subword':<20} | {'Rank 1 (Prob)':<25} | {'Rank 2 (Prob)':<25} | {'Rank 3 (Prob)':<25}"
        )
        print("-" * 100)

        for i in range(len(subwords)):
            row_parts = []
            for j in range(3):
                tag_name = inv_tag_map.get(top_indices[i][j].item(), "Unknown")
                prob_val = top_probs[i][j].item()
                row_parts.append(f"{tag_name} ({prob_val:.4f})")

            print(
                f"{subwords[i]:<20} | {row_parts[0]:<25} | {row_parts[1]:<25} | {row_parts[2]:<25}"
            )

        print("-" * 100)


if __name__ == "__main__":
    interactive_inference()
