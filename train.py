import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import json
import collections
from model import NERTransformer
import os
from tokenizers import Tokenizer


def load_data(filename):
    data = []
    with open(filename, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def build_tag_map(data):
    tags = set()
    for item in data:
        tags.update(item["tags"])

    tag_map = {"[PAD]": 0}  # 0 will be ignored in loss using ignore_index
    # We also need a generic "O" if it's not present (it usually is)
    # Let's ensure O is there and mapped normally if strict_mapping isn't key

    # We want consistent mapping.
    sorted_tags = sorted(list(tags))
    if "O" in sorted_tags:
        sorted_tags.remove("O")
        sorted_tags.insert(0, "O")  # Put O first after PAD usually

    for tag in sorted_tags:
        if tag != "[PAD]":
            tag_map[tag] = len(tag_map)

    return tag_map


class NERDataset(Dataset):
    def __init__(self, data, tokenizer, tag_map):
        self.data = data
        self.tokenizer = tokenizer
        self.tag_map = tag_map
        self.pad_token_id = tokenizer.token_to_id("[PAD]")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item["tokens"]
        tags = item["tags"]

        token_ids = []
        label_ids = []

        # [CLS]
        token_ids.append(self.tokenizer.token_to_id("[CLS]"))
        label_ids.append(self.tag_map["O"])  # Use 'O' for special tokens

        for word, tag in zip(tokens, tags):
            # Encode word without special tokens
            enc = self.tokenizer.encode(word, add_special_tokens=False)
            subwords = enc.ids

            if not subwords:
                # Fallback for weird cases or empty strings
                continue

            token_ids.extend(subwords)

            # Map tag
            current_tag_id = self.tag_map[tag]

            # Logic: B-X -> I-X for subwords. I-X -> I-X. O -> O.
            if tag.startswith("B-"):
                label_ids.append(current_tag_id)  # First subword gets B-

                # Determine I- tag
                i_tag_str = "I-" + tag[2:]
                i_tag_id = self.tag_map.get(i_tag_str, current_tag_id)

                label_ids.extend([i_tag_id] * (len(subwords) - 1))
            else:
                label_ids.extend([current_tag_id] * len(subwords))

        # [SEP]
        token_ids.append(self.tokenizer.token_to_id("[SEP]"))
        label_ids.append(self.tag_map["O"])

        return torch.tensor(token_ids), torch.tensor(label_ids)


def collate_fn(batch):
    tokens, tags = zip(*batch)
    # We need to know the pad value for tokenizer. usually 0 if defined first in special tokens
    # But let's check input
    # In train_tokenizer we put [PAD] first, so id is 0.

    padded_tokens = pad_sequence(tokens, batch_first=True, padding_value=0)
    padded_tags = pad_sequence(
        tags, batch_first=True, padding_value=0
    )  # 0 is [PAD] in tag_map too
    return padded_tokens, padded_tags


def train():
    # Setup
    train_data = load_data("train.jsonl")
    test_data = load_data("test.jsonl")

    # Load Tokenizer
    tokenizer = Tokenizer.from_file("tokenizer.json")
    vocab_size = tokenizer.get_vocab_size()

    # Build Tag Map from TRAIN data
    tag_map = build_tag_map(train_data)
    inv_tag_map = {v: k for k, v in tag_map.items()}

    print(f"Vocab size: {vocab_size}")
    print(f"Num tags: {len(tag_map)}")

    # Hyperparameters
    BATCH_SIZE = 16
    D_MODEL = 128  # Increased d_model for subword complexity
    NHEAD = 4
    NUM_LAYERS = 2
    DIM_FEEDFORWARD = 256
    MAX_SEQ_LEN = 200  # Subwords increase length
    LR = 0.001
    EPOCHS = 15  # Increased epochs for scheduler

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device.")
    else:
        device = torch.device("cpu")
        print("Using CPU device.")

    # DataLoaders # Pass tokenizer
    train_dataset = NERDataset(train_data, tokenizer, tag_map)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )

    test_dataset = NERDataset(test_data, tokenizer, tag_map)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # Model
    model = NERTransformer(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        max_seq_length=MAX_SEQ_LEN,  # Not strictly enforced in arch (only PE), but good reference
        num_classes=len(tag_map),
        dropout=0.1,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore [PAD]
    optimizer = optim.AdamW(
        model.parameters(), lr=LR
    )  # AdamW is better for Transformers

    # Scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, epochs=EPOCHS, steps_per_epoch=len(train_loader)
    )

    # Logs
    with open("training_log.txt", "w") as log_file:
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            for tokens, tags in train_loader:
                tokens, tags = tokens.to(device), tags.to(device)

                optimizer.zero_grad()

                # Check seq length, if > max_len, truncate?
                # Our collate pads, but PE is fixed size.
                # If batch is longer than PE Max Len, it will crash.
                # Just clip for safety if needed, or rely on dataset gen not being huge.
                if tokens.size(1) > MAX_SEQ_LEN:
                    tokens = tokens[:, :MAX_SEQ_LEN]
                    tags = tags[:, :MAX_SEQ_LEN]

                src_key_padding_mask = tokens == 0

                output = model(tokens, src_key_padding_mask=src_key_padding_mask)
                loss = criterion(output.view(-1, len(tag_map)), tags.view(-1))

                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            # Validation
            model.eval()
            val_loss = 0
            correct_tokens = 0
            total_tokens = 0

            # Metrics for P/R (Micro-average excluding O)
            tp = 0
            fp = 0
            fn = 0
            o_id = tag_map.get(
                "O", 0
            )  # Fallback to 0 if 'O' not found (though it should be)

            monitor_samples = []

            with torch.no_grad():
                for batch_idx, (tokens, tags) in enumerate(test_loader):
                    tokens, tags = tokens.to(device), tags.to(device)
                    if tokens.size(1) > MAX_SEQ_LEN:
                        tokens = tokens[:, :MAX_SEQ_LEN]
                        tags = tags[:, :MAX_SEQ_LEN]

                    src_key_padding_mask = tokens == 0

                    output = model(tokens, src_key_padding_mask=src_key_padding_mask)
                    loss = criterion(output.view(-1, len(tag_map)), tags.view(-1))
                    val_loss += loss.item()

                    pred_tags = torch.argmax(output, dim=-1)
                    mask = tags != 0
                    correct_tokens += ((pred_tags == tags) & mask).sum().item()
                    total_tokens += mask.sum().item()

                    # Compute Precision/Recall
                    # We only care about non-pad tokens for stats
                    valid_tags = tags[mask]
                    valid_preds = pred_tags[mask]

                    for t, p in zip(valid_tags, valid_preds):
                        t = t.item()
                        p = p.item()

                        if t == o_id and p == o_id:
                            continue  # True Negative

                        if t == p:
                            # Correct prediction (and we know t != O because of logic below if we structured it differently,
                            # but here t==p. If t==O, we continued. So t!=O)
                            tp += 1
                        else:
                            # Mismatch
                            if p != o_id:
                                fp += 1  # Predicted an entity, but it was wrong (either O or different entity)
                            if t != o_id:
                                fn += 1  # Was an entity, but predicted wrong (either O or different entity)

                    if batch_idx == 0:
                        # Decode 3 samples
                        tokens_cpu = tokens.cpu().numpy()
                        tags_cpu = tags.cpu().numpy()
                        preds_cpu = pred_tags.cpu().numpy()

                        for i in range(min(3, len(tokens))):
                            # Decode back to string
                            # Note: We can decode subwords, but tags are per subword.
                            # Let's print subwords + tags to see BPE effect

                            length = (tokens[i] != 0).sum()
                            # Get tokens
                            # id_to_token might return None for some?

                            dec_tokens = [
                                tokenizer.id_to_token(tid)
                                for tid in tokens_cpu[i][:length]
                            ]
                            true_labels = [
                                inv_tag_map.get(t, "?") for t in tags_cpu[i][:length]
                            ]
                            pred_labels = [
                                inv_tag_map.get(t, "?") for t in preds_cpu[i][:length]
                            ]

                            monitor_samples.append(
                                list(zip(dec_tokens, true_labels, pred_labels))
                            )

            avg_val_loss = val_loss / len(test_loader)
            val_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0

            # P/R Calculation
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            metrics = f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f} | F1: {f1:.4f} (P: {precision:.4f}, R: {recall:.4f})"
            print(metrics)
            log_file.write(metrics + "\n")

            log_file.write(f"Sample Predictions (Epoch {epoch+1}):\n")
            for idx, sample in enumerate(monitor_samples):
                header = f"\n--- Test Sample {idx+1} ---\n"
                header += f"{'Subword':<20} {'True':<20} {'Pred':<20}\n"
                header += "-" * 60 + "\n"
                log_file.write(header)

                for w, t, p in sample:
                    line = f"{str(w):<20} {t:<20} {p:<20}\n"
                    log_file.write(line)

            log_file.write("-" * 60 + "\n\n")

    # Serialize artifacts
    torch.save(model.state_dict(), "ner_model.pth")
    with open("tag_map.json", "w") as f:
        json.dump(tag_map, f)
    # Tokenizer is already saved as tokenizer.json
    print("Model, tokenizer, and tag_map saved.")

    # Generate final prediction file
    with open("predictions.txt", "w") as f:
        f.write("--- Final BPE Model Predictions ---\n\n")
        model.eval()
        with torch.no_grad():
            for batch_idx, (tokens, tags) in enumerate(test_loader):
                tokens, tags = tokens.to(device), tags.to(device)
                if tokens.size(1) > MAX_SEQ_LEN:
                    tokens = tokens[:, :MAX_SEQ_LEN]
                    tags = tags[:, :MAX_SEQ_LEN]
                src_key_padding_mask = tokens == 0
                output = model(tokens, src_key_padding_mask=src_key_padding_mask)
                pred_tags = torch.argmax(output, dim=-1)

                tokens_cpu = tokens.cpu().numpy()
                tags_cpu = tags.cpu().numpy()
                preds_cpu = pred_tags.cpu().numpy()

                for i in range(len(tokens)):
                    length = (tokens[i] != 0).sum()
                    dec_tokens = [
                        tokenizer.id_to_token(tid) for tid in tokens_cpu[i][:length]
                    ]
                    true_labels = [
                        inv_tag_map.get(t, "?") for t in tags_cpu[i][:length]
                    ]
                    pred_labels = [
                        inv_tag_map.get(t, "?") for t in preds_cpu[i][:length]
                    ]

                    f.write(f"Example {batch_idx * BATCH_SIZE + i + 1}:\n")
                    for w, t, p in zip(dec_tokens, true_labels, pred_labels):
                        f.write(f"{str(w):<20} {t:<20} {p:<20}\n")
                    f.write("\n")


if __name__ == "__main__":
    train()
