import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import json
import collections
from model import NERTransformer
import os
import tiktoken


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

    tag_map = {"[PAD]": 0}

    sorted_tags = sorted(list(tags))
    if "O" in sorted_tags:
        sorted_tags.remove("O")
        sorted_tags.insert(0, "O")

    for tag in sorted_tags:
        if tag != "[PAD]":
            tag_map[tag] = len(tag_map)

    return tag_map


class NERDataset(Dataset):
    def __init__(self, data, encoding, tag_map):
        self.data = data
        self.enc = encoding
        self.tag_map = tag_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item["tokens"]
        tags = item["tags"]

        token_ids = []
        label_ids = []

        for word, tag in zip(tokens, tags):
            subwords = self.enc.encode(word)
            if not subwords:
                continue

            token_ids.extend(subwords)
            current_tag_id = self.tag_map[tag]

            if tag.startswith("B-"):
                label_ids.append(current_tag_id)
                i_tag_str = "I-" + tag[2:]
                i_tag_id = self.tag_map.get(i_tag_str, current_tag_id)
                label_ids.extend([i_tag_id] * (len(subwords) - 1))
            else:
                label_ids.extend([current_tag_id] * len(subwords))

        return torch.tensor(token_ids), torch.tensor(label_ids)


def collate_fn(batch, pad_id):
    tokens, tags = zip(*batch)
    padded_tokens = pad_sequence(tokens, batch_first=True, padding_value=pad_id)
    padded_tags = pad_sequence(
        tags, batch_first=True, padding_value=0
    )  # 0 is [PAD] in tag_map
    return padded_tokens, padded_tags


def train():
    train_data = load_data("train.jsonl")
    test_data = load_data("test.jsonl")

    encoding = tiktoken.get_encoding("gpt2")
    PAD_ID = encoding.n_vocab
    vocab_size = encoding.n_vocab + 1

    tag_map = build_tag_map(train_data)
    inv_tag_map = {v: k for k, v in tag_map.items()}

    print(f"Vocab size: {vocab_size} (tiktoken + pad)")
    print(f"Num tags: {len(tag_map)}")

    BATCH_SIZE = 16
    D_MODEL = 128
    NHEAD = 4
    NUM_LAYERS = 2
    DIM_FEEDFORWARD = 256
    MAX_SEQ_LEN = 256
    LR = 0.001
    EPOCHS = 10

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    train_dataset = NERDataset(train_data, encoding, tag_map)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, PAD_ID),
    )

    test_dataset = NERDataset(test_data, encoding, tag_map)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, collate_fn=lambda b: collate_fn(b, PAD_ID)
    )

    model = NERTransformer(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        max_seq_length=MAX_SEQ_LEN,
        num_classes=len(tag_map),
        dropout=0.1,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, epochs=EPOCHS, steps_per_epoch=len(train_loader)
    )

    with open("training_log.txt", "w") as log_file:
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            for tokens, tags in train_loader:
                tokens, tags = tokens.to(device), tags.to(device)
                if tokens.size(1) > MAX_SEQ_LEN:
                    tokens = tokens[:, :MAX_SEQ_LEN]
                    tags = tags[:, :MAX_SEQ_LEN]

                optimizer.zero_grad()
                src_key_padding_mask = tokens == PAD_ID
                output = model(tokens, src_key_padding_mask=src_key_padding_mask)
                loss = criterion(output.view(-1, len(tag_map)), tags.view(-1))
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            model.eval()
            val_loss = 0
            correct_tokens, total_tokens = 0, 0
            tp, fp, fn = 0, 0, 0
            o_id = tag_map.get("O", 0)
            monitor_samples = []

            with torch.no_grad():
                for batch_idx, (tokens, tags) in enumerate(test_loader):
                    tokens, tags = tokens.to(device), tags.to(device)
                    if tokens.size(1) > MAX_SEQ_LEN:
                        tokens = tokens[:, :MAX_SEQ_LEN]
                        tags = tags[:, :MAX_SEQ_LEN]

                    src_key_padding_mask = tokens == PAD_ID
                    output = model(tokens, src_key_padding_mask=src_key_padding_mask)
                    loss = criterion(output.view(-1, len(tag_map)), tags.view(-1))
                    val_loss += loss.item()

                    pred_tags = torch.argmax(output, dim=-1)
                    mask = tags != 0
                    correct_tokens += ((pred_tags == tags) & mask).sum().item()
                    total_tokens += mask.sum().item()

                    v_tags = tags[mask]
                    v_preds = pred_tags[mask]
                    for t, p in zip(v_tags, v_preds):
                        t, p = t.item(), p.item()
                        if t == o_id and p == o_id:
                            continue
                        if t == p:
                            tp += 1
                        else:
                            if p != o_id:
                                fp += 1
                            if t != o_id:
                                fn += 1

                    if batch_idx == 0:
                        t_cpu, tags_cpu, preds_cpu = (
                            tokens.cpu().numpy(),
                            tags.cpu().numpy(),
                            pred_tags.cpu().numpy(),
                        )
                        for i in range(min(3, len(tokens))):
                            length = (tokens[i] != PAD_ID).sum()
                            dec_tokens = [
                                encoding.decode([tid]) for tid in t_cpu[i][:length]
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
            val_acc = correct_tokens / total_tokens if total_tokens > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

            metrics = f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | F1: {f1:.4f} (P: {prec:.4f}, R: {rec:.4f})"
            print(metrics)
            log_file.write(metrics + "\n")

            log_file.write(f"Sample Predictions (Epoch {epoch+1}):\n")
            for idx, sample in enumerate(monitor_samples):
                header = (
                    f"\n--- Test Sample {idx+1} ---\nSubword{'':<13} True{'':<16} Pred\n"
                    + "-" * 60
                    + "\n"
                )
                log_file.write(header)
                for w, t, p in sample:
                    log_file.write(f"{str(w):<20} {t:<20} {p:<20}\n")
            log_file.write("-" * 60 + "\n\n")

    torch.save(model.state_dict(), "ner_model.pth")
    with open("tag_map.json", "w") as f:
        json.dump(tag_map, f)
    print("Model and tag_map saved.")

    # Generate predictions.txt
    print("Generating predictions.txt...")
    model.eval()
    with open("predictions.txt", "w") as f:
        f.write("--- tiktoken Model Predictions ---\n\n")
        with torch.no_grad():
            for batch_idx, (tokens, tags) in enumerate(test_loader):
                tokens, tags = tokens.to(device), tags.to(device)
                if tokens.size(1) > MAX_SEQ_LEN:
                    tokens = tokens[:, :MAX_SEQ_LEN]
                    tags = tags[:, :MAX_SEQ_LEN]

                src_key_padding_mask = tokens == PAD_ID
                output = model(tokens, src_key_padding_mask=src_key_padding_mask)
                preds = torch.argmax(output, dim=-1)

                t_cpu, tags_cpu, p_cpu = (
                    tokens.cpu().numpy(),
                    tags.cpu().numpy(),
                    preds.cpu().numpy(),
                )
                for i in range(len(tokens)):
                    length = (tokens[i] != PAD_ID).sum()
                    dec_t = [encoding.decode([tid]) for tid in t_cpu[i][:length]]
                    true_l = [inv_tag_map.get(t, "?") for t in tags_cpu[i][:length]]
                    pred_l = [inv_tag_map.get(t, "?") for t in p_cpu[i][:length]]

                    f.write(f"Example {batch_idx * BATCH_SIZE + i + 1}:\n")
                    for w, t, p in zip(dec_t, true_l, pred_l):
                        f.write(f"{str(w):<20} {t:<20} {p:<20}\n")
                    f.write("\n")
    print("predictions.txt generated.")


if __name__ == "__main__":
    train()
