# NER Transformer Project

This project implements a Named Entity Recognition (NER) model using a Transformer architecture (PyTorch). It is designed to identify **DISEASE** and **MEDICATION** entities in medical text.

## Project Structure

- `model.py`: Defines the `NERTransformer` architecture, including custom Positional Encoding.
- `generate_data.py`: Generates synthetic training and testing data using templates and entity lists.
- `train.py`: Handles data loading, preprocessing, model training, validation, and logging.
- `predictions.txt`: Stores the model's predictions on the test set after training.
- `training_log.txt`: detailed logs of the training process, including sample outputs per epoch.

## Usage

### 1. Generate Data
First, generate the synthetic training and testing datasets. This script handles train/test splitting with controlled overlap to test generalization.
```bash
python3 generate_data.py
```
This produces `train.jsonl` and `test.jsonl`.

### 2. Train the Model
Run the training pipeline. This script trains the model, monitors progress (saved to `training_log.txt`), and saves the final model artifacts.
```bash
# Recommendation: use MPS fallback if on Apple Silicon to avoid potential operator issues
export PYTORCH_ENABLE_MPS_FALLBACK=1
python3 train.py
```

### 3. Deep Dive Inference
Use the interactive inference tool to analyze specific sentences. This shows tokenization and the top 3 predicted tags with their probabilities for each subword.
```bash
python3 inference.py
```

### 4. View Results
- **Metrics**: Check `training_log.txt` for loss, accuracy, and F1 updates.
- **Predictions**: Open `predictions.txt` to see final test set predictions.

## Architecture
- **Embedding**: Standard Token Embeddings
- **Positional Encoding**: Sinusoidal (standard Transformer PE)
- **Encoder**: PyTorch `TransformerEncoder` layers
- **Head**: Linear layer for token classification

## Dataset
The data is synthetically generated using templates mimicking medical notes (e.g., "Patient presents with symptoms of {disease}..."). It includes:
- **Noise samples**: Sentences with no entities.
- **Complex sentences**: Paragraph-like structures in the test set.
- **Unseen Entities**: A portion of the test set entities are not seen during training to evaluate zero-shot entity recognition capabilities.
