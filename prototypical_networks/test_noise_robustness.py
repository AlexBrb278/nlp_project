"""
Test Prototypical Networks BERT Model on Noisy CLINC Dataset
Applies the same noise types as the Colab notebook and evaluates performance
"""

import json
import logging
import nltk
import os
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score

# Suppress NLTK download messages inside nlpaug/other libs
logging.getLogger('nltk').setLevel(logging.ERROR)

# Ensure NLTK data can be found and pre-download needed packages
nltk_data_path = os.path.expanduser('~/nltk_data')
os.environ['NLTK_DATA'] = nltk_data_path
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

nltk.data.path.append(nltk_data_path)
nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_path, quiet=True)
nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
nltk.download('universal_tagset', download_dir=nltk_data_path, quiet=True)

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
from torch.serialization import safe_globals
import tokenizers
import transformers.models.bert.tokenization_bert

MODEL_PATH = "./prototypical_bert_model/model.pth"
TOKENIZER_PATH = "./prototypical_bert_model/tokenizer.pth"
# METADATA_PATH = "./prototypical_bert_model/metadata.json"  # deleted, label2id rebuilt from training data instead
PROTOTYPES_PATH = "./prototypical_bert_model/prototypes.pt"
CLINC_PATH = "../oos-eval/data/data_full.json"
MAX_LENGTH = 64
BATCH_SIZE = 128  # Increased for GPU efficiency
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")


class PrototypicalBertNetwork(torch.nn.Module):
    """Same model architecture as training script"""
    def __init__(self, bert_model_name="bert-base-uncased", hidden_dim=768):
        super().__init__()
        from transformers import BertModel
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.hidden_dim = hidden_dim

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings

    def compute_prototypes(self, embeddings, labels, num_classes):
        prototypes = []
        for class_idx in range(num_classes):
            class_mask = labels == class_idx
            if class_mask.sum() > 0:
                class_embeddings = embeddings[class_mask]
                prototype = class_embeddings.mean(dim=0)
                prototypes.append(prototype)
            else:
                prototypes.append(torch.randn(self.hidden_dim, device=embeddings.device))

        return torch.stack(prototypes)


def load_clinc_test_data(clinc_path, label2id):
    """Load only test data from CLINC dataset, using label2id from training metadata."""
    with open(clinc_path, "r") as f:
        data = json.load(f)

    test_texts = [item[0] for item in data["test"]]
    test_labels = [item[1] for item in data["test"]]
    test_ids = [label2id[l] for l in test_labels]

    print(f"Test: {len(test_texts)} examples")

    return test_texts, test_ids


def add_casing_noise(texts):
    """Convert all text to uppercase"""
    return [t.upper() for t in texts]

def add_keyboard_noise(texts, aug_char_p=0.15):
    """Add keyboard typos"""
    aug = nac.KeyboardAug(aug_char_p=aug_char_p)
    results = []
    for t in texts:
        try:
            results.append(aug.augment(t)[0])
        except:
            results.append(t)
    return results

def add_spelling_noise(texts, aug_p=0.2):
    """Add spelling errors"""
    aug = naw.SpellingAug(aug_p=aug_p)
    results = []
    for t in texts:
        try:
            results.append(aug.augment(t)[0])
        except:
            results.append(t)
    return results

def add_synonym_noise(texts, aug_p=0.2):
    """Replace words with synonyms"""
    aug = naw.SynonymAug(aug_p=aug_p)
    results = []
    for t in texts:
        try:
            results.append(aug.augment(t)[0])
        except:
            results.append(t)
    return results

ABBREV_MAP = {
    "please": "pls", "you": "u", "your": "ur",
    "are": "r", "okay": "ok", "thanks": "thx",
    "tomorrow": "tmrw", "because": "bc",
    "with": "w/", "without": "w/o",
    "information": "info", "application": "app",
    "number": "num", "message": "msg",
    "account": "acct", "department": "dept",
    "appointment": "appt", "maximum": "max",
    "minimum": "min", "approximately": "approx",
}

def add_abbreviation_noise(texts):
    """Replace common words with abbreviations"""
    results = []
    for t in texts:
        words = t.split()
        new_words = [ABBREV_MAP.get(w.lower(), w) for w in words]
        results.append(" ".join(new_words))
    return results


class IntentDataset(Dataset):
    """Same dataset class as training"""
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


def evaluate_model(model, texts, labels, tokenizer, prototypes, device):
    """Evaluate model using pre-built training prototypes."""
    dataset = IntentDataset(texts, labels, tokenizer, MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["label"]

            embeddings = model(input_ids, attention_mask)
            distances = torch.cdist(embeddings, prototypes.to(device))
            preds = torch.argmin(distances, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_labels.numpy())

    return accuracy_score(all_labels, all_preds)

def main():
    print("="*70)
    print("TESTING PROTOTYPICAL NETWORKS ON NOISY CLINC DATA")
    print("="*70)

    # Pre-download NLTK data to avoid repeated downloads during noise generation
    print("\nPre-downloading NLTK data...")
    import nltk
    import os

    # Set NLTK data path to avoid repeated downloads
    nltk_data_path = os.path.expanduser('~/nltk_data')
    os.environ['NLTK_DATA'] = nltk_data_path
    if not os.path.exists(nltk_data_path):
        os.makedirs(nltk_data_path)

    # Also set environment variables that nlpaug might use
    os.environ['NLTK_DATA_PATH'] = nltk_data_path
    nltk.data.path.append(nltk_data_path)

    # Download required NLTK data
    try:
        nltk.download('wordnet', download_dir=nltk_data_path, quiet=True)
        nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_path, quiet=True)
        print("NLTK data ready.")
    except Exception as e:
        print(f"Warning: NLTK download failed: {e}")
        print("Continuing anyway...")

    # Disable any further NLTK downloader output from nlpaug or other libraries
    nltk.download = lambda *args, **kwargs: None

    # Load model and metadata
    print("\nLoading trained model...")
    model = PrototypicalBertNetwork().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    # Load tokenizer (trusted source, so weights_only=False is safe)
    tokenizer = torch.load(TOKENIZER_PATH, weights_only=False)

    # Rebuild label2id from training data using the same logic as train_prototypical_bert.py
    # (sorted unique train labels → indices), matching the mapping used when prototypes.pt was built
    with open(CLINC_PATH, "r") as f:
        _data = json.load(f)
    _train_labels = [item[1] for item in _data["train"]]
    _all_labels = sorted(set(_train_labels))
    label2id = {label: idx for idx, label in enumerate(_all_labels)}
    num_classes = len(label2id)
    print(f"Model loaded - {num_classes} classes")

    print("\nLoading saved prototypes...")
    prototypes = torch.load(PROTOTYPES_PATH, map_location=DEVICE)
    print(f"Prototypes loaded — shape: {prototypes.shape}")

    # Load test data
    print("\nLoading CLINC test data...")
    # test_texts, test_labels, label2id = load_clinc_test_data(CLINC_PATH)  # old call, rebuilt label2id from test set
    test_texts, test_labels = load_clinc_test_data(CLINC_PATH, label2id)

    # Generate noisy versions (same as Colab)
    print("\nGenerating noisy test sets...")

    print("  Creating casing noise...")
    casing_texts = add_casing_noise(test_texts)
    print("  Creating keyboard noise...")
    keyboard_texts = add_keyboard_noise(test_texts)
    print("  Creating spelling noise...")
    spelling_texts = add_spelling_noise(test_texts)
    print("  Creating synonym noise...")
    synonym_texts = add_synonym_noise(test_texts)
    print("  Creating abbreviation noise...")
    abbreviation_texts = add_abbreviation_noise(test_texts)

    noisy_variants = {
        "original": test_texts,
        "casing": casing_texts,
        "keyboard": keyboard_texts,
        "spelling": spelling_texts,
        "synonyms": synonym_texts,
        "abbreviations": abbreviation_texts,
    }

    print("Noisy variants created:")
    for name, texts in noisy_variants.items():
        print(f"  {name}: {len(texts)} examples")

    # Show examples
    print("\nExamples of noise:")
    for name in ["casing", "keyboard", "spelling", "synonyms", "abbreviations"]:
        noisy = noisy_variants[name]
        print(f"  {name}: '{test_texts[0]}' → '{noisy[0]}'")

    # Evaluate on all variants
    print("\n" + "="*70)
    print("EVALUATING MODEL ON NOISY DATA")
    print("="*70)

    results = {}
    for noise_type, texts in noisy_variants.items():
        print(f"\nEvaluating on {noise_type} data...")
        acc = evaluate_model(model, texts, test_labels, tokenizer, prototypes, DEVICE)
        results[noise_type] = acc
        print(f"  {noise_type}: {acc:.2f}")

    # Calculate degradation metrics
    print("\n" + "="*70)
    print("PERFORMANCE DEGRADATION ANALYSIS")
    print("="*70)

    clean_acc = results["original"]
    print(f"  Original: {clean_acc:.2f}")
    print(f"{'Noise Type':<15} {'Accuracy':>10} {'PDR':>8} {'ERM':>8}")
    print("-" * 45)

    for noise_type, acc in results.items():
        if noise_type == "original":
            print(f"  {noise_type}: {acc:.2f}")
        else:
            pdr = (clean_acc - acc) / clean_acc * 100  # Performance Degradation Rate
            erm = (1 - acc) / (1 - clean_acc)  # Error Rate Multiplier
            print(f"  {noise_type}: {acc:.2f} ({pdr:.1f}%, {erm:.2f})")
    # Save results
    output_results = {
        "noise_evaluation": results,
        "degradation_analysis": {
            "baseline_accuracy": clean_acc,
            "noise_types_tested": list(results.keys()),
        }
    }

    with open("./prototypical_bert_model/noise_evaluation_results.json", "w") as f:
        json.dump(output_results, f, indent=2)

    print("Results saved to: ./prototypical_bert_model/noise_evaluation_results.json")
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()