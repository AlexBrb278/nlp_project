"""
Prototypical Networks with BERT for Intent Classification on CLINC Dataset
Trains on full CLINC dataset with same hyperparameters as standard BERT baseline
"""

import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================
CLINC_PATH = "../oos-eval/data/data_full.json"  # Adjust path as needed
OUTPUT_DIR = "./prototypical_bert_model"
MAX_LENGTH = 64
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_EVAL = 64
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")


# ============================================================================
# DATA LOADING
# ============================================================================
def load_clinc_data(clinc_path):
    """Load CLINC OOS dataset"""
    with open(clinc_path, "r") as f:
        data = json.load(f)

    train_texts = [item[0] for item in data["train"]]
    train_labels = [item[1] for item in data["train"]]

    test_texts = [item[0] for item in data["test"]]
    test_labels = [item[1] for item in data["test"]]

    val_texts = [item[0] for item in data["val"]]
    val_labels = [item[1] for item in data["val"]]

    # Create label mapping
    all_labels = list(set(train_labels))
    all_labels.sort()
    label2id = {label: idx for idx, label in enumerate(all_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    train_ids = [label2id[l] for l in train_labels]
    test_ids = [label2id[l] for l in test_labels]
    val_ids = [label2id[l] for l in val_labels]

    print(f"Train: {len(train_texts)} examples")
    print(f"Test:  {len(test_texts)} examples")
    print(f"Val:   {len(val_texts)} examples")
    print(f"Number of intents: {len(all_labels)}")
    print(f"Sample: '{train_texts[0]}' -> {train_labels[0]}")

    return (
        train_texts, train_ids,
        test_texts, test_ids,
        val_texts, val_ids,
        label2id, id2label
    )


# ============================================================================
# DATASET CLASS
# ============================================================================
class IntentDataset(Dataset):
    """PyTorch dataset for intent classification"""
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


# ============================================================================
# PROTOTYPICAL NETWORKS MODEL
# ============================================================================
class PrototypicalBertNetwork(nn.Module):
    """
    Prototypical Networks with BERT encoder
    Uses BERT embeddings and computes prototypes for each class
    """
    def __init__(self, bert_model_name="bert-base-uncased", hidden_dim=768):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.hidden_dim = hidden_dim

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
        Returns:
            embeddings: (batch_size, hidden_dim) - [CLS] token representations
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        # Use [CLS] token representation as sentence embedding
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings

    def compute_prototypes(self, embeddings, labels, num_classes):
        """
        Compute class prototypes from embeddings
        Args:
            embeddings: (batch_size, hidden_dim)
            labels: (batch_size,)
            num_classes: int
        Returns:
            prototypes: (num_classes, hidden_dim)
        """
        prototypes = []
        for class_idx in range(num_classes):
            class_mask = labels == class_idx
            if class_mask.sum() > 0:
                class_embeddings = embeddings[class_mask]
                prototype = class_embeddings.mean(dim=0)
                prototypes.append(prototype)
            else:
                # Random prototype if class not in batch
                prototypes.append(torch.randn(self.hidden_dim, device=embeddings.device))

        return torch.stack(prototypes)


# ============================================================================
# TRAINING
# ============================================================================
def train_epoch(model, train_loader, optimizer, num_classes, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # Forward pass
        embeddings = model(input_ids, attention_mask)
        prototypes = model.compute_prototypes(embeddings, labels, num_classes)

        # Compute distances to prototypes
        distances = torch.cdist(embeddings, prototypes)  # (batch_size, num_classes)
        logits = -distances  # Use negative distance as logits (closer = higher score)

        # Loss: cross-entropy
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Track predictions
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def evaluate(model, eval_loader, num_classes, device):
    """Evaluate on validation/test set"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # Forward pass
            embeddings = model(input_ids, attention_mask)
            prototypes = model.compute_prototypes(embeddings, labels, num_classes)

            # Compute distances
            distances = torch.cdist(embeddings, prototypes)
            logits = -distances

            # Predictions
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================
def main():
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n" + "="*60)
    print("Loading CLINC dataset...")
    print("="*60)
    train_texts, train_ids, test_texts, test_ids, val_texts, val_ids, label2id, id2label = load_clinc_data(CLINC_PATH)
    num_classes = len(label2id)

    # Tokenizer
    print("\nLoading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Datasets
    print("Creating datasets...")
    train_dataset = IntentDataset(train_texts, train_ids, tokenizer, MAX_LENGTH)
    val_dataset = IntentDataset(val_texts, val_ids, tokenizer, MAX_LENGTH)
    test_dataset = IntentDataset(test_texts, test_ids, tokenizer, MAX_LENGTH)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_EVAL, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_EVAL, shuffle=False)

    # Model
    print("\nInitializing Prototypical Networks model...")
    model = PrototypicalBertNetwork(bert_model_name="bert-base-uncased").to(DEVICE)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # Training loop
    print("\n" + "="*60)
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    print("="*60)

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, num_classes, DEVICE)
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc*100:.2f}%")

        # Validate
        val_acc = evaluate(model, val_loader, num_classes, DEVICE)
        print(f"  Val Accuracy: {val_acc*100:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"  ✓ Best model updated (val_acc: {val_acc*100:.2f}%)")

    # Load best model and evaluate on test set
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print("\n" + "="*60)
    print("Evaluating on test set...")
    print("="*60)
    test_acc = evaluate(model, test_loader, num_classes, DEVICE)
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")

    # Save model and metadata
    print("\n" + "="*60)
    print("Saving model...")
    print("="*60)
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/model.pth")
    torch.save(tokenizer, f"{OUTPUT_DIR}/tokenizer.pth")
    
    metadata = {
        "test_accuracy": float(test_acc),
        "best_val_accuracy": float(best_val_acc),
        "num_classes": num_classes,
        "id2label": id2label,
        "label2id": label2id,
        "hyperparameters": {
            "num_epochs": NUM_EPOCHS,
            "batch_size_train": BATCH_SIZE_TRAIN,
            "batch_size_eval": BATCH_SIZE_EVAL,
            "learning_rate": LEARNING_RATE,
            "max_length": MAX_LENGTH,
        }
    }
    
    with open(f"{OUTPUT_DIR}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model saved to {OUTPUT_DIR}/")
    print(f"Test Accuracy: {test_acc*100:.2f}%")


if __name__ == "__main__":
    main()
