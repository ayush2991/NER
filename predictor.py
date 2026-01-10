import torch
import torch.nn.functional as F
import json
import tiktoken
import re
from model import NERTransformer

class NERPredictor:
    def __init__(self, model_path="ner_model.pth", tag_map_path="tag_map.json"):
        with open(tag_map_path, "r") as f:
            self.tag_map = json.load(f)
        self.inv_tag_map = {v: k for k, v in self.tag_map.items()}
        
        self.encoding = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.encoding.n_vocab + 1
        
        # Model Hyperparameters (matching train.py)
        D_MODEL = 128
        NHEAD = 4
        NUM_LAYERS = 2
        DIM_FEEDFORWARD = 256
        MAX_SEQ_LEN = 256
        NUM_CLASSES = len(self.tag_map)
        
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() 
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        self.model = NERTransformer(
            vocab_size=self.vocab_size,
            d_model=D_MODEL,
            nhead=NHEAD,
            num_encoder_layers=NUM_LAYERS,
            dim_feedforward=DIM_FEEDFORWARD,
            max_seq_length=MAX_SEQ_LEN,
            num_classes=NUM_CLASSES,
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def predict(self, text):
        if not text:
            return []

        # Simple regex to split by words but preserve all characters (including spaces)
        # This helps in reconstruction for the UI
        words_and_spaces = re.findall(r'\S+|\s+', text)
        
        all_token_ids = []
        word_to_subtoken_indices = []
        
        current_idx = 0
        for part in words_and_spaces:
            if part.isspace():
                word_to_subtoken_indices.append(None)
                continue
            
            subtoken_ids = self.encoding.encode(part)
            num_subtokens = len(subtoken_ids)
            all_token_ids.extend(subtoken_ids)
            word_to_subtoken_indices.append(list(range(current_idx, current_idx + num_subtokens)))
            current_idx += num_subtokens

        if not all_token_ids:
            return [{"text": part, "label": "O"} for part in words_and_spaces]

        input_tensor = torch.tensor([all_token_ids]).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            logits = output[0] # [seq_len, num_classes]

        results = []
        for part, indices in zip(words_and_spaces, word_to_subtoken_indices):
            if part.isspace():
                results.append({"text": part, "label": "O"})
                continue
            
            if indices:
                # Max pooling over logit scores for the subtokens of this word
                # Ensure indices don't exceed logits range (just in case)
                valid_indices = [idx for idx in indices if idx < logits.size(0)]
                if not valid_indices:
                    results.append({"text": part, "label": "O"})
                    continue
                    
                word_logits = logits[valid_indices] # [num_subtokens, num_classes]
                max_logits, _ = torch.max(word_logits, dim=0) # [num_classes]
                
                predicted_idx = torch.argmax(max_logits).item()
                label = self.inv_tag_map.get(predicted_idx, "O")
                
                # Clean label for visualization (e.g., B-DISEASE -> DISEASE)
                # Removing the 'B-' or 'I-' prefix if present
                clean_label = "O"
                if label != "O" and label != "[PAD]":
                    clean_label = label[2:] if len(label) > 2 and label[1] == '-' else label
                
                results.append({"text": part, "label": clean_label})
            else:
                results.append({"text": part, "label": "O"})
                
        return results

if __name__ == "__main__":
    predictor = NERPredictor()
    sample = "The patient was prescribed Metformin for Diabetes."
    res = predictor.predict(sample)
    print(f"Sample: {sample}")
    for r in res:
        if r['label'] != 'O':
            print(f"  FOUND: '{r['text']}' -> {r['label']}")
