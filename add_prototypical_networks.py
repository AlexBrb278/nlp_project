"""
Evaluate a trained PrototypicalBertNetwork using prototypical inference
on the noisy test set.

Usage:
    python evaluate_model.py <model_dir> <output_dir>

model_dir must contain model.pth and tokenizer.pth (torch.save format).
"""

import argparse
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModel
from torch.serialization import safe_globals
import transformers.models.bert.tokenization_bert
from sklearn.metrics import accuracy_score

CLINC_PATH = "oos-eval/data/data_full.json"
NOISY_PATH = "noisy_test_set2.json"
MAX_LENGTH = 64
BATCH_SIZE = 64
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PrototypicalBertNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.hidden_dim = 768

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return out.last_hidden_state[:, 0, :]


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts, self.labels, self.tokenizer = texts, labels, tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], add_special_tokens=True,
                             max_length=MAX_LENGTH, padding="max_length",
                             truncation=True, return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": torch.tensor(self.labels[idx], dtype=torch.long)}


def get_embeddings(model, texts, labels, tokenizer):
    loader = DataLoader(TextDataset(texts, labels, tokenizer), batch_size=BATCH_SIZE, shuffle=False)
    embs, lbls = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            e = model(batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE))
            embs.append(e.cpu())
            lbls.extend(batch["label"].tolist())
    return torch.cat(embs), lbls


def build_prototypes(embeddings, labels, num_classes):
    protos = torch.zeros(num_classes, embeddings.shape[1])
    for c in range(num_classes):
        mask = torch.tensor(labels) == c
        if mask.sum() > 0:
            protos[c] = embeddings[mask].mean(dim=0)
    return protos


def evaluate(model, texts, labels, tokenizer, prototypes):
    embs, true = get_embeddings(model, texts, labels, tokenizer)
    preds = torch.argmin(torch.cdist(embs, prototypes.cpu()), dim=1).tolist()
    return accuracy_score(true, preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir",       help="Directory containing the model")
    parser.add_argument("output_dir",      help="Directory to save results.json and embeddings")
    parser.add_argument("--embeddings",    help="Path to saved embeddings .pt file (skips recomputing)", default=None)
    parser.add_argument("--model_format",  choices=["pytorch", "huggingface"], default="pytorch",
                                           help="pytorch: model.pth + tokenizer.pth  |  huggingface: save_pretrained format")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Device: {DEVICE}")

    if args.model_format == "pytorch":
        with safe_globals([transformers.models.bert.tokenization_bert.BertTokenizer]):
            tokenizer = torch.load(os.path.join(args.model_dir, "tokenizer.pth"), weights_only=False)
        model = PrototypicalBertNetwork().to(DEVICE)
        model.load_state_dict(torch.load(os.path.join(args.model_dir, "model.pth"), map_location=DEVICE))
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        model = PrototypicalBertNetwork().to(DEVICE)
        model.bert = AutoModel.from_pretrained(args.model_dir).to(DEVICE)
    print("Model loaded.")

    with open(CLINC_PATH) as f:
        data = json.load(f)
    train_labels_str = [item[1] for item in data["train"]]
    label2id  = {l: i for i, l in enumerate(sorted(set(train_labels_str)))}
    train_ids = [label2id[l] for l in train_labels_str]
    test_ids  = [label2id[item[1]] for item in data["test"]]

    emb_save_path = os.path.join(args.output_dir, "train_embeddings.pt")
    if args.embeddings:
        print(f"Loading embeddings from {args.embeddings} ...")
        saved = torch.load(args.embeddings)
        train_embs, train_lbls = saved["embeddings"], saved["labels"]
    else:
        print(f"Computing embeddings for {len(train_ids)} training examples ...")
        train_embs, train_lbls = get_embeddings(model, [item[0] for item in data["train"]], train_ids, tokenizer)
        torch.save({"embeddings": train_embs, "labels": train_lbls}, emb_save_path)
        print(f"Embeddings saved to {emb_save_path}")

    prototypes = build_prototypes(train_embs, train_lbls, len(label2id))

    with open(NOISY_PATH) as f:
        noisy = json.load(f)

    results, clean_acc = {}, None
    for noise_type, texts in noisy.items():
        acc = evaluate(model, texts, test_ids, tokenizer, prototypes)
        results[noise_type] = round(acc, 4)
        if noise_type == "original":
            clean_acc = acc
        pdr = (clean_acc - acc) / clean_acc * 100 if clean_acc and noise_type != "original" else 0
        erm = (1 - acc) / (1 - clean_acc) if clean_acc and noise_type != "original" and clean_acc < 1 else 1
        marker = f"  PDR={pdr:.1f}%  ERM={erm:.2f}x" if noise_type != "original" else ""
        print(f"  {noise_type:<15} {acc*100:.2f}%{marker}")

    out_path = os.path.join(args.output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump({"model": args.model_dir, "results": results}, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()

