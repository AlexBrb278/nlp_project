import argparse
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BertModel,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

try:
    from transformers import CanineModel, CanineTokenizer
    _CANINE_AVAILABLE = True
except ImportError:
    _CANINE_AVAILABLE = False

_nltk_path = os.path.expanduser("~/nltk_data")
os.makedirs(_nltk_path, exist_ok=True)

if _nltk_path not in nltk.data.path:
    nltk.data.path.append(_nltk_path)

for pkg in ("wordnet", "averaged_perceptron_tagger", "punkt"):
    nltk.download(pkg, download_dir=_nltk_path, quiet=True)

try:
    import nlpaug.augmenter.char as nac
    import nlpaug.augmenter.word as naw
    _NLPAUG_AVAILABLE = True
except ImportError:
    _NLPAUG_AVAILABLE = False
    print("[WARNING] nlpaug not installed — keyboard/spelling/synonym noise disabled.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

ROOT        = Path(__file__).parent
CLINC_PATH  = ROOT / "dataset" / "data_full.json"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {DEVICE}")

NUM_CLASSES = 150


@dataclass
class ExperimentConfig:
    use_canine:       bool = False
    use_contrastive:  bool = False
    use_prototypical: bool = False

    num_epochs:         int   = 10
    batch_size:         int   = 32
    lr:                 float = 2e-5
    weight_decay:       float = 0.01
    warmup_ratio:       float = 0.06
    contrastive_lambda: float = 0.3
    temperature:        float = 0.07
    dropout:            float = 0.1
    num_workers:        int   = 0

    @property
    def max_len(self) -> int:
        return 128 if self.use_canine else 64

    def name(self) -> str:
        parts = []
        if self.use_canine:       parts.append("K")
        if self.use_contrastive:  parts.append("C")
        if self.use_prototypical: parts.append("P")
        return "+".join(parts) if parts else "baseline"

    def __str__(self) -> str:
        return (f"ExperimentConfig(canine={self.use_canine}, "
                f"contrastive={self.use_contrastive}, "
                f"proto={self.use_prototypical})")


def load_clinc_data(path: Path = CLINC_PATH) -> Dict:
    with open(path) as f:
        raw = json.load(f)

    in_scope_labels = sorted({item[1] for item in raw["train"]})
    label2id = {lbl: idx for idx, lbl in enumerate(in_scope_labels)}
    id2label  = {idx: lbl for lbl, idx in label2id.items()}

    def _parse(split_key):
        texts, labels = [], []
        for text, intent in raw[split_key]:
            texts.append(text)
            labels.append(label2id[intent])
        return texts, labels

    train_texts, train_labels = _parse("train")
    val_texts,   val_labels   = _parse("val")
    test_texts,  test_labels  = _parse("test")

    log.info(f"Data loaded — train: {len(train_texts)}, "
             f"val: {len(val_texts)}, test: {len(test_texts)}")
    return dict(
        train_texts=train_texts, train_labels=train_labels,
        val_texts=val_texts,     val_labels=val_labels,
        test_texts=test_texts,   test_labels=test_labels,
        label2id=label2id,       id2label=id2label,
    )


ABBREV_MAP = {
    "please": "pls", "you": "u", "your": "ur", "are": "r",
    "okay": "ok", "thanks": "thx", "tomorrow": "tmrw", "because": "bc",
    "with": "w/", "without": "w/o", "information": "info",
    "application": "app", "number": "num", "message": "msg",
    "account": "acct", "department": "dept", "appointment": "appt",
    "maximum": "max", "minimum": "min", "approximately": "approx",
}

_TTS_SUBS = [
    ("their", "there"), ("there", "their"), ("to", "too"),
    ("too", "to"), ("for", "four"), ("two", "to"),
    ("know", "no"), ("no", "know"), ("right", "write"),
    ("write", "right"), ("hear", "here"), ("here", "hear"),
]


def add_casing_noise(texts: List[str]) -> List[str]:
    return [t.upper() for t in texts]


def add_keyboard_noise(texts: List[str], aug_char_p: float = 0.15) -> List[str]:
    if not _NLPAUG_AVAILABLE:
        return texts
    aug = nac.KeyboardAug(aug_char_p=aug_char_p)
    out = []
    for t in texts:
        try:
            out.append(aug.augment(t)[0])
        except Exception:
            out.append(t)
    return out


def add_spelling_noise(texts: List[str], aug_p: float = 0.2) -> List[str]:
    if not _NLPAUG_AVAILABLE:
        return texts
    aug = naw.SpellingAug(aug_p=aug_p)
    out = []
    for t in texts:
        try:
            out.append(aug.augment(t)[0])
        except Exception:
            out.append(t)
    return out


def add_synonym_noise(texts: List[str], aug_p: float = 0.2) -> List[str]:
    if not _NLPAUG_AVAILABLE:
        return texts
    aug = naw.SynonymAug(aug_p=aug_p)
    out = []
    for t in texts:
        try:
            out.append(aug.augment(t)[0])
        except Exception:
            out.append(t)
    return out


def add_abbreviation_noise(texts: List[str]) -> List[str]:
    out = []
    for t in texts:
        words = t.split()
        out.append(" ".join(ABBREV_MAP.get(w.lower(), w) for w in words))
    return out


def add_tts_noise(texts: List[str]) -> List[str]:
    out = []
    for t in texts:
        for src, tgt in _TTS_SUBS:
            t = t.replace(f" {src} ", f" {tgt} ")
        out.append(t)
    return out


NOISE_FNS = {
    "clean":    lambda ts: ts,
    "casing":   add_casing_noise,
    "keyboard": add_keyboard_noise,
    "spelling": add_spelling_noise,
    "synonyms": add_synonym_noise,
    "abbrev":   add_abbreviation_noise,
    "tts":      add_tts_noise,
}

_kb_aug = nac.KeyboardAug(aug_char_p=0.15) if _NLPAUG_AVAILABLE else None
_sp_aug = naw.SpellingAug(aug_p=0.2)       if _NLPAUG_AVAILABLE else None


def generate_noisy_text(text: str) -> str:
    if not _NLPAUG_AVAILABLE:
        return text
    choice = random.choices(["keyboard", "spelling"], weights=[0.7, 0.3])[0]
    try:
        if choice == "keyboard":
            return _kb_aug.augment(text)[0]
        else:
            return _sp_aug.augment(text)[0]
    except Exception:
        return text


def _build_tokenizer(config: ExperimentConfig):
    if config.use_canine:
        if not _CANINE_AVAILABLE:
            raise ImportError("transformers>=4.18 required for CanineTokenizer")
        return CanineTokenizer.from_pretrained("google/canine-s")
    return BertTokenizer.from_pretrained("bert-base-uncased")


def _tokenize(tokenizer, text: str, max_len: int) -> Dict[str, torch.Tensor]:
    enc = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return {k: v.squeeze(0) for k, v in enc.items()
            if k in ("input_ids", "attention_mask")}


class UnifiedDataset(Dataset):
    def __init__(
        self,
        texts:          List[str],
        labels:         List[int],
        tokenizer,
        config:         ExperimentConfig,
        generate_noisy: bool = False,
    ):
        self.texts          = texts
        self.labels         = labels
        self.tokenizer      = tokenizer
        self.config         = config
        self.generate_noisy = generate_noisy

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text  = self.texts[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        clean = _tokenize(self.tokenizer, text, self.config.max_len)

        if self.generate_noisy:
            noisy_text = generate_noisy_text(text)
            noisy = _tokenize(self.tokenizer, noisy_text, self.config.max_len)
            return {
                "clean_ids":  clean["input_ids"],
                "clean_mask": clean["attention_mask"],
                "noisy_ids":  noisy["input_ids"],
                "noisy_mask": noisy["attention_mask"],
                "label":      label,
            }

        return {
            "input_ids":      clean["input_ids"],
            "attention_mask": clean["attention_mask"],
            "label":          label,
        }


class UnifiedIntentModel(nn.Module):
    HIDDEN = 768

    def __init__(self, config: ExperimentConfig, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.config      = config
        self.num_classes = num_classes
        self.dropout     = nn.Dropout(config.dropout)

        if config.use_canine:
            if not _CANINE_AVAILABLE:
                raise ImportError("transformers>=4.18 required for CanineModel")
            self.encoder = CanineModel.from_pretrained("google/canine-s")
        else:
            self.encoder = BertModel.from_pretrained("bert-base-uncased")

        if not config.use_prototypical:
            self.classifier = nn.Linear(self.HIDDEN, num_classes)

    def get_embedding(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.dropout(out.last_hidden_state[:, 0, :])

    def compute_prototypes(
        self,
        embeddings:  torch.Tensor,
        labels:      torch.Tensor,
        num_classes: int,
    ) -> torch.Tensor:
        prototypes = []
        for class_idx in range(num_classes):
            mask = labels == class_idx
            if mask.sum() > 0:
                prototypes.append(embeddings[mask].mean(dim=0))
            else:
                prototypes.append(
                    torch.randn(self.HIDDEN, device=embeddings.device) * 0.01
                )
        return torch.stack(prototypes)

    @staticmethod
    def contrastive_loss(
        clean_emb:   torch.Tensor,
        noisy_emb:   torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        batch_size = clean_emb.size(0)
        clean_norm = F.normalize(clean_emb, dim=-1)
        noisy_norm = F.normalize(noisy_emb, dim=-1)

        combined = torch.cat([clean_norm, noisy_norm], dim=0)
        sim = torch.matmul(combined, combined.T) / temperature

        diag_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=sim.device)
        sim.masked_fill_(diag_mask, float("-inf"))

        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(0, batch_size),
        ]).to(sim.device)

        return F.cross_entropy(sim, labels)

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        noisy_ids:      Optional[torch.Tensor] = None,
        noisy_mask:     Optional[torch.Tensor] = None,
        labels:         Optional[torch.Tensor] = None,
        prototypes:     Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        clean_emb = self.get_embedding(input_ids, attention_mask)

        if self.config.use_prototypical:
            assert prototypes is not None, (
                "prototypes must be provided when use_prototypical=True; "
                "call build_epoch_prototypes() before forward()"
            )
            distances = torch.cdist(clean_emb, prototypes)
            logits = -distances
        else:
            logits = self.classifier(clean_emb)

        loss = None
        if labels is not None:
            ce_loss = F.cross_entropy(logits, labels)
            loss = ce_loss

            if self.config.use_contrastive:
                assert noisy_ids is not None, "noisy_ids required when use_contrastive=True"
                noisy_emb = self.get_embedding(noisy_ids, noisy_mask)
                cl_loss = self.contrastive_loss(
                    clean_emb, noisy_emb, self.config.temperature
                )
                loss = ce_loss + self.config.contrastive_lambda * cl_loss

        return logits, loss


@torch.no_grad()
def build_epoch_prototypes(
    model:       UnifiedIntentModel,
    loader:      DataLoader,
    num_classes: int,
    device:      torch.device,
) -> torch.Tensor:
    model.eval()
    all_embs:   List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for batch in loader:
        ids  = batch.get("clean_ids",  batch.get("input_ids")).to(device)
        mask = batch.get("clean_mask", batch.get("attention_mask")).to(device)
        embs = model.get_embedding(ids, mask)
        all_embs.append(embs.cpu())
        all_labels.append(batch["label"])

    all_embs   = torch.cat(all_embs,   dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return model.compute_prototypes(all_embs, all_labels, num_classes).to(device)


def train_epoch(
    model:      UnifiedIntentModel,
    loader:     DataLoader,
    optimizer,
    scheduler,
    config:     ExperimentConfig,
    device:     torch.device,
    prototypes: Optional[torch.Tensor] = None,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        optimizer.zero_grad()

        labels = batch["label"].to(device)

        if config.use_contrastive:
            input_ids      = batch["clean_ids"].to(device)
            attention_mask = batch["clean_mask"].to(device)
            noisy_ids      = batch["noisy_ids"].to(device)
            noisy_mask     = batch["noisy_mask"].to(device)
        else:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            noisy_ids = noisy_mask = None

        _, loss = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            noisy_ids=noisy_ids,
            noisy_mask=noisy_mask,
            labels=labels,
            prototypes=prototypes.to(device) if prototypes is not None else None,
        )

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model:      UnifiedIntentModel,
    loader:     DataLoader,
    config:     ExperimentConfig,
    device:     torch.device,
    prototypes: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    model.eval()
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"]

        logits, _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prototypes=prototypes.to(device) if prototypes is not None else None,
        )
        preds = logits.argmax(dim=-1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean()

    return {"accuracy": float(accuracy)}


@torch.no_grad()
def evaluate_all_noise(
    model:       UnifiedIntentModel,
    test_texts:  List[str],
    test_labels: List[int],
    tokenizer,
    config:      ExperimentConfig,
    device:      torch.device,
    prototypes:  Optional[torch.Tensor] = None,
) -> Dict[str, Dict[str, float]]:
    results = {}
    for noise_name, noise_fn in NOISE_FNS.items():
        log.info(f"  Evaluating noise: {noise_name}")
        noisy_texts = noise_fn(test_texts)
        ds = UnifiedDataset(noisy_texts, test_labels, tokenizer, config,
                            generate_noisy=False)
        loader = DataLoader(ds, batch_size=config.batch_size * 2,
                            shuffle=False, num_workers=config.num_workers)
        results[noise_name] = evaluate(model, loader, config, device, prototypes)

    return results


def run_experiment(
    config: ExperimentConfig,
    data:   Dict,
    device: torch.device,
) -> Dict:
    log.info("=" * 70)
    log.info(f"Running experiment: {config.name()}  ({config})")
    log.info("=" * 70)

    save_dir = RESULTS_DIR / config.name()
    save_dir.mkdir(exist_ok=True)

    tokenizer = _build_tokenizer(config)

    train_ds = UnifiedDataset(
        data["train_texts"], data["train_labels"], tokenizer, config,
        generate_noisy=config.use_contrastive,
    )
    val_ds = UnifiedDataset(
        data["val_texts"], data["val_labels"], tokenizer, config,
        generate_noisy=False,
    )
    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=True,  num_workers=config.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=config.batch_size * 2,
                              shuffle=False, num_workers=config.num_workers)

    model = UnifiedIntentModel(config, num_classes=NUM_CLASSES).to(device)

    total_steps  = len(train_loader) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    optimizer    = AdamW(model.parameters(), lr=config.lr,
                         weight_decay=config.weight_decay)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_acc = 0.0
    history = []

    for epoch in range(1, config.num_epochs + 1):
        t0 = time.time()

        prototypes = None
        if config.use_prototypical:
            log.info(f"  Epoch {epoch}: building prototypes...")
            model.eval()
            prototypes = build_epoch_prototypes(
                model, train_loader, NUM_CLASSES, device
            )
            model.train()

        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            config, device, prototypes,
        )

        val_metrics = evaluate(model, val_loader, config, device, prototypes)
        val_acc     = val_metrics["accuracy"]

        log.info(f"  Epoch {epoch}/{config.num_epochs} | "
                 f"loss={train_loss:.4f} | val_acc={val_acc:.4f} | "
                 f"time={time.time()-t0:.1f}s")
        history.append({"epoch": epoch, "loss": train_loss, **val_metrics})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            log.info(f"  → New best val_acc: {best_val_acc:.4f}")

    model.load_state_dict(torch.load(save_dir / "best_model.pt", map_location=device))
    model.eval()

    prototypes = None
    if config.use_prototypical:
        prototypes = build_epoch_prototypes(model, train_loader, NUM_CLASSES, device)

    log.info("  Running full noise evaluation on test set...")
    noise_results = evaluate_all_noise(
        model, data["test_texts"], data["test_labels"],
        tokenizer, config, device, prototypes,
    )

    results = {
        "config": config.name(),
        "best_val_accuracy": best_val_acc,
        "noise_results": noise_results,
        "training_history": history,
    }

    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"  Results saved → {save_dir / 'results.json'}")

    return results


EXPERIMENT_SUITE = [
    ExperimentConfig(use_canine=False, use_contrastive=False, use_prototypical=False),
    ExperimentConfig(use_canine=True,  use_contrastive=False, use_prototypical=False),
    ExperimentConfig(use_canine=False, use_contrastive=True,  use_prototypical=False),
    ExperimentConfig(use_canine=False, use_contrastive=False, use_prototypical=True),
    ExperimentConfig(use_canine=False, use_contrastive=True,  use_prototypical=True),
    ExperimentConfig(use_canine=True,  use_contrastive=False, use_prototypical=True),
    ExperimentConfig(use_canine=True,  use_contrastive=True,  use_prototypical=False),
    ExperimentConfig(use_canine=True,  use_contrastive=True,  use_prototypical=True),
]


def print_results_table(all_results: Dict) -> None:
    noise_types = list(NOISE_FNS.keys())
    col_w = 10

    header = f"{'Config':<10}" + "".join(f"{n:>{col_w}}" for n in noise_types)
    print("\n" + "=" * len(header))
    print("ACCURACY BY NOISE TYPE")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for cfg_name, res in all_results.items():
        row = f"{cfg_name:<10}"
        for nt in noise_types:
            acc = res["noise_results"].get(nt, {}).get("accuracy", float("nan"))
            row += f"{acc*100:>{col_w}.1f}"
        print(row)

    print("=" * len(header))
    print("(values are percentages)\n")


def save_all_results(all_results: Dict) -> None:
    out_path = RESULTS_DIR / "combined_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"All results saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run combined model experiments")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Run a single config by name (e.g. K+C+P). Omit to run all.",
    )
    args = parser.parse_args()

    data = load_clinc_data()

    suite = EXPERIMENT_SUITE
    if args.config:
        suite = [c for c in EXPERIMENT_SUITE if c.name() == args.config]
        if not suite:
            raise ValueError(f"Unknown config '{args.config}'. "
                             f"Available: {[c.name() for c in EXPERIMENT_SUITE]}")

    all_results = {}
    for cfg in suite:
        all_results[cfg.name()] = run_experiment(cfg, data, DEVICE)

    print_results_table(all_results)
    save_all_results(all_results)


if __name__ == "__main__":
    main()
