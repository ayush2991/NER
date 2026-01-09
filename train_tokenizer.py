from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import json


def train_tokenizer(files):
    # Initialize a tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    # Initialize a trainer
    trainer = BpeTrainer(
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"], vocab_size=5000
    )

    # Prepare iterator
    def batch_iterator():
        for filename in files:
            with open(filename, "r") as f:
                for line in f:
                    entry = json.loads(line)
                    # Join tokens back to sentence for training
                    yield " ".join(entry["tokens"])

    # Train
    print("Training tokenizer...")
    tokenizer.train_from_iterator(batch_iterator(), trainer)

    # Save
    tokenizer.save("tokenizer.json")
    print("Tokenizer saved to tokenizer.json")


if __name__ == "__main__":
    train_tokenizer(["train.jsonl", "test.jsonl"])
