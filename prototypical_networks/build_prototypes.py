import argparse
import json
import os
import subprocess
import webbrowser
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch.serialization import safe_globals
import tokenizers
import transformers.models.bert.tokenization_bert


# ============================================================================
# CONFIGURATION
# ============================================================================
CLINC_PATH = "../oos-eval/data/data_full.json"
OUTPUT_DIR = "./prototypical_bert_model"
MODEL_PATH = f"{OUTPUT_DIR}/model.pth"
TOKENIZER_PATH = f"{OUTPUT_DIR}/tokenizer.pth"
METADATA_PATH = f"{OUTPUT_DIR}/metadata.json"
PROTOTYPES_PATH = f"{OUTPUT_DIR}/prototypes.pt"
PROTOTYPE_META_PATH = f"{OUTPUT_DIR}/prototype_metadata.json"
QUERY_VIZ_PATH = f"{OUTPUT_DIR}/query_visualization.html"
MAX_LENGTH = 64
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# DOMAIN GROUPINGS  (same as visualize_prototypes.py)
# ============================================================================
DOMAIN_INTENTS = {
    "auto & commute": [
        "directions", "distance", "gas", "gas_type", "how_busy",
        "jump_start", "last_maintenance", "mpg", "oil_change_how",
        "oil_change_when", "schedule_maintenance", "tire_change",
        "tire_pressure", "traffic", "uber",
    ],
    "banking": [
        "account_blocked", "balance", "bill_balance", "bill_due",
        "direct_deposit", "exchange_rate", "freeze_account",
        "interest_rate", "min_payment", "order_checks", "pay_bill",
        "payday", "report_fraud", "routing", "spending_history",
        "transactions", "transfer",
    ],
    "credit cards": [
        "apr", "card_declined", "credit_limit", "credit_limit_change",
        "credit_score", "damaged_card", "expiration_date",
        "improve_credit_score", "international_fees", "new_card",
        "pin_change", "redeem_rewards", "replacement_card_duration",
        "report_lost_card", "rewards_balance",
    ],
    "home": [
        "find_phone", "insurance", "insurance_change", "make_call",
        "order", "order_status", "reset_settings", "share_location",
        "shopping_list", "shopping_list_update", "smart_home",
        "sync_device", "text", "user_name",
    ],
    "kitchen & dining": [
        "accept_reservations", "calories", "cook_time", "food_last",
        "ingredient_substitution", "ingredients_list", "meal_suggestion",
        "nutrition_info", "recipe", "restaurant_reservation",
        "restaurant_reviews", "restaurant_suggestion",
    ],
    "meta": [
        "are_you_a_bot", "change_accent", "change_ai_name",
        "change_language", "change_speed", "change_user_name",
        "change_volume", "do_you_have_pets", "how_old_are_you",
        "meaning_of_life", "what_are_your_hobbies", "what_can_i_ask_you",
        "what_is_your_name", "where_are_you_from", "whisper_mode",
        "who_do_you_work_for", "who_made_you",
    ],
    "small talk": [
        "cancel", "flip_coin", "fun_fact", "goodbye", "greeting",
        "maybe", "no", "repeat", "roll_dice", "spelling",
        "tell_joke", "thank_you", "yes",
    ],
    "travel": [
        "book_flight", "book_hotel", "cancel_reservation", "car_rental",
        "carry_on", "confirm_reservation", "flight_status",
        "international_visa", "lost_luggage", "plug_type",
        "travel_alert", "travel_notification", "travel_suggestion",
        "vaccines",
    ],
    "utility": [
        "alarm", "calculator", "calendar", "calendar_update",
        "current_location", "date", "definition", "measurement_conversion",
        "next_holiday", "reminder", "reminder_update", "time", "timer",
        "timezone", "todo_list", "todo_list_update", "translate", "weather",
    ],
    "work": [
        "application_status", "income", "meeting_schedule", "pto_balance",
        "pto_request", "pto_request_status", "pto_used", "rollover_401k",
        "schedule_meeting", "taxes", "w2",
    ],
    "entertainment": [
        "next_song", "play_music", "update_playlist", "what_song",
    ],
}

DOMAIN_COLORS = {
    "auto & commute":   "#4e79a7",
    "banking":          "#f28e2b",
    "credit cards":     "#e15759",
    "home":             "#76b7b2",
    "kitchen & dining": "#59a14f",
    "meta":             "#edc948",
    "small talk":       "#b07aa1",
    "travel":           "#ff9da7",
    "utility":          "#9c755f",
    "work":             "#bab0ac",
    "entertainment":    "#d37295",
}

_LABEL_TO_DOMAIN = {
    intent: domain
    for domain, intents in DOMAIN_INTENTS.items()
    for intent in intents
}
_LABEL_TO_COLOR = {
    intent: DOMAIN_COLORS[domain]
    for domain, intents in DOMAIN_INTENTS.items()
    for intent in intents
}


# ============================================================================
# MODEL
# ============================================================================
class PrototypicalBertNetwork(nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased", hidden_dim=768):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.hidden_dim = hidden_dim

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return outputs.last_hidden_state[:, 0, :]


# ============================================================================
# DATA
# ============================================================================
class IntentDataset(Dataset):
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
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def load_clinc_train_data(clinc_path, label2id):
    with open(clinc_path, "r") as f:
        data = json.load(f)

    texts = [item[0] for item in data["train"]]
    labels = [item[1] for item in data["train"]]
    ids = [label2id[label] for label in labels]

    return texts, ids


# ============================================================================
# PROTOTYPE BUILDING
# ============================================================================
def build_prototypes(model, tokenizer, texts, labels, num_classes, device, batch_size=64):
    dataset = IntentDataset(texts, labels, tokenizer, MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_embs, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            emb = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            all_embs.append(emb.cpu())
            all_labels.extend(batch["label"].tolist())

    all_embs = torch.cat(all_embs)
    labels_t = torch.tensor(all_labels)
    protos = torch.zeros(num_classes, model.hidden_dim)
    for c in range(num_classes):
        mask = labels_t == c
        if mask.sum() > 0:
            protos[c] = all_embs[mask].mean(dim=0)
    return protos


# ============================================================================
# INFERENCE
# ============================================================================
def classify_text(model, tokenizer, prototypes, text, device, top_k=3):
    """Returns (top_k_indices, top_k_distances, embedding_numpy_array)."""
    model.eval()
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        emb = model(input_ids, attention_mask)
        distances = torch.cdist(emb, prototypes.to(device))
        top_dists, top_idxs = torch.topk(distances[0], k=top_k, largest=False)

    return top_idxs.cpu().tolist(), top_dists.cpu().tolist(), emb.cpu().numpy()


# ============================================================================
# 2D VISUALISATION
# ============================================================================
def project_with_tsne(prototypes_np, query_emb_np):
    """Run t-SNE on prototypes + query together. Returns (proto_coords_2d, query_xy)."""
    all_vecs = np.vstack([prototypes_np, query_emb_np])   # (151, 768)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(all_vecs)                  # (151, 2)
    return coords[:-1], coords[-1]                         # proto coords, query coord


def build_proto_points(coords_2d, id2label):
    points = []
    for i, (x, y) in enumerate(coords_2d):
        label  = id2label[i]
        domain = _LABEL_TO_DOMAIN.get(label, "other")
        color  = _LABEL_TO_COLOR.get(label, "#aaaaaa")
        points.append({"x": float(x), "y": float(y),
                       "label": label, "domain": domain, "color": color})
    return points


def _make_query_html(proto_points, query_xy, query_text, pred_label, emb_np, top3=None):
    pred_domain = _LABEL_TO_DOMAIN.get(pred_label, "other")
    pred_color  = _LABEL_TO_COLOR.get(pred_label, "#aaaaaa")

    top3_data = []
    if top3:
        for rank, (lbl, dist) in enumerate(top3, 1):
            top3_data.append({
                "rank":   rank,
                "label":  lbl,
                "domain": _LABEL_TO_DOMAIN.get(lbl, "other"),
                "color":  _LABEL_TO_COLOR.get(lbl, "#aaaaaa"),
                "dist":   round(dist, 4),
            })

    data = {
        "protos": proto_points,
        "query":  {
            "x":      float(query_xy[0]),
            "y":      float(query_xy[1]),
            "text":   query_text,
            "pred":   pred_label,
            "domain": pred_domain,
            "color":  pred_color,
            "norm":   round(float(np.linalg.norm(emb_np)), 4),
            "first8": [round(float(v), 4) for v in emb_np[0, :8]],
            "px":     round(float(query_xy[0]), 4),
            "py":     round(float(query_xy[1]), 4),
            "top3":   top3_data,
        },
        "domainColors": DOMAIN_COLORS,
    }
    data_json = json.dumps(data)

    # Build HTML by concatenation to avoid any f-string / JS template conflicts.
    return (
        """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Query Visualisation</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background:#0f0f1e; color:#dde; font-family:'Segoe UI',system-ui,sans-serif;
         display:flex; flex-direction:column; align-items:center; padding:20px 16px 40px; }
  #info {
    background:#0d0d20; border:1px solid #1e2044; border-radius:9px;
    padding:12px 20px; margin-bottom:18px; font-size:0.85rem;
    display:flex; gap:20px; align-items:flex-start; flex-wrap:wrap;
  }
  #info .q-text { color:#aac; }
  #info .q-text span { color:#eef; font-weight:600; }
  #info .q-pred { font-weight:700; font-size:1rem; }
  #info .q-domain { font-size:0.78rem; color:#778; }
  #info .q-emb { font-size:0.75rem; color:#667; line-height:1.7; border-left:1px solid #1e2044; padding-left:18px; }
  #info .q-emb b { color:#99b; font-weight:600; }
  #info .q-top3 { font-size:0.78rem; color:#99b; border-left:1px solid #1e2044; padding-left:18px; }
  #info .q-top3 b { color:#ccd; font-weight:600; }
  .top3-row { display:flex; align-items:center; gap:8px; padding:2px 0; }
  .top3-rank { color:#556; font-size:0.72rem; width:16px; }
  .top3-dot  { width:9px; height:9px; border-radius:50%; flex-shrink:0; }
  .top3-label { font-weight:600; }
  .top3-dist  { color:#556; font-size:0.72rem; }
  .emb-bar-wrap { display:flex; gap:3px; align-items:flex-end; height:28px; margin-top:4px; }
  .emb-bar { width:14px; border-radius:2px 2px 0 0; position:relative; }
  .emb-bar-neg { border-radius:0 0 2px 2px; }
  #wrap { display:flex; gap:18px; align-items:flex-start; }
  canvas { background:#0d0d20; border-radius:10px; border:1px solid #1e2044; cursor:crosshair; }
  #sidebar { background:#0d0d20; border:1px solid #1e2044; border-radius:10px;
             padding:16px 18px; min-width:175px; max-width:185px; }
  #sidebar h2 { font-size:0.7rem; text-transform:uppercase; letter-spacing:0.12em;
                color:#667; margin-bottom:12px; }
  .litem { display:flex; align-items:center; gap:9px; padding:4px 6px; border-radius:5px;
           cursor:pointer; font-size:0.78rem; color:#bbc;
           transition:background 0.12s,opacity 0.12s; user-select:none; }
  .litem:hover { background:#171733; }
  .litem.hidden { opacity:0.3; }
  .ldot { width:11px; height:11px; border-radius:50%; flex-shrink:0; }
  #tip { position:fixed; background:rgba(10,10,30,0.92); border:1px solid #445;
         border-radius:7px; padding:7px 13px; pointer-events:none; display:none; z-index:9999; }
  #tip .t-label  { font-size:0.82rem; font-weight:700; color:#eef; }
  #tip .t-domain { font-size:0.73rem; color:#99b; margin-top:2px; }
  .legend-key { display:flex; align-items:center; gap:7px; margin-top:14px;
                font-size:0.75rem; color:#778; }
  .legend-key .k-dot { width:11px; height:11px; flex-shrink:0; }
</style>
</head>
<body>

<div id="info">
  <div>
    <div class="q-text">Query: <span id="q-text-val"></span></div>
    <div class="q-pred" id="q-pred-val"></div>
    <div class="q-domain" id="q-domain-val"></div>
  </div>
  <div class="q-top3">
    <b>Top 3 closest intents</b>
    <div id="q-top3-rows"></div>
  </div>
  <div class="q-emb">
    <div><b>Embedding</b> &nbsp;768-dim &nbsp;|&nbsp; norm = <span id="q-norm"></span></div>
    <div>First 8 dims: <span id="q-first8"></span></div>
    <div class="emb-bar-wrap" id="q-bars"></div>
    <div style="margin-top:4px;"><b>2D projection</b> &nbsp;x = <span id="q-px"></span> &nbsp;y = <span id="q-py"></span></div>
  </div>
</div>

<div id="wrap">
  <canvas id="c" width="820" height="700"></canvas>
  <div id="sidebar">
    <h2>Domains</h2>
    <div id="legend"></div>
    <div class="legend-key">
      <svg class="k-dot" viewBox="0 0 11 11">
        <polygon points="5.5,0 11,5.5 5.5,11 0,5.5" fill="#ffffff" stroke="#ffdd00" stroke-width="1.2"/>
      </svg>
      your query
    </div>
    <div class="legend-key">
      <svg class="k-dot" viewBox="0 0 11 11">
        <circle cx="5.5" cy="5.5" r="5" fill="none" stroke="#ffffff" stroke-width="1.8"/>
      </svg>
      predicted
    </div>
  </div>
</div>

<div id="tip">
  <div class="t-label"></div>
  <div class="t-domain"></div>
</div>

<script>
const DATA = """ + data_json + """;

const canvas = document.getElementById('c');
const ctx    = canvas.getContext('2d');
const tip    = document.getElementById('tip');
const W = canvas.width, H = canvas.height;
const PAD = 44, R = 7;

let hidden = new Set();

// ── Populate info bar ────────────────────────────────────────────────────────
const q = DATA.query;
document.getElementById('q-text-val').textContent   = '"' + q.text + '"';
document.getElementById('q-pred-val').textContent   = q.pred.replace(/_/g, ' ');
document.getElementById('q-pred-val').style.color   = q.color;
document.getElementById('q-domain-val').textContent = '● ' + q.domain;
document.getElementById('q-domain-val').style.color = q.color;
document.getElementById('q-norm').textContent   = q.norm;
document.getElementById('q-first8').textContent = '[' + q.first8.join(', ') + ']';
document.getElementById('q-px').textContent     = q.px;
document.getElementById('q-py').textContent     = q.py;

// ── Populate top-3 list ──────────────────────────────────────────────────────
(function buildTop3() {
  const wrap = document.getElementById('q-top3-rows');
  (q.top3 || []).forEach(function(item) {
    const row = document.createElement('div');
    row.className = 'top3-row';
    row.innerHTML =
      '<span class="top3-rank">#' + item.rank + '</span>' +
      '<span class="top3-dot" style="background:' + item.color + '"></span>' +
      '<span class="top3-label" style="color:' + item.color + '">' + item.label.replace(/_/g, ' ') + '</span>' +
      '<span class="top3-dist">d=' + item.dist + '</span>';
    wrap.appendChild(row);
  });
})();

// mini bar chart for the first 8 dims
(function buildBars() {
  const wrap = document.getElementById('q-bars');
  const vals = q.first8;
  const maxAbs = Math.max(...vals.map(Math.abs), 0.001);
  const H = 26;
  vals.forEach((v, i) => {
    const bar = document.createElement('div');
    const h = Math.round(Math.abs(v) / maxAbs * H);
    const col = v >= 0 ? q.color : '#e15759';
    bar.style.cssText = 'width:14px;background:' + col + ';height:' + h + 'px;' +
      'border-radius:' + (v >= 0 ? '2px 2px 0 0' : '0 0 2px 2px') + ';' +
      'align-self:' + (v >= 0 ? 'flex-end' : 'flex-start') + ';' +
      'title:dim ' + i + '=' + v;
    bar.title = 'dim ' + i + ' = ' + v;
    wrap.appendChild(bar);
  });
})();

// ── Scale all coords (protos + query) to canvas pixels once ─────────────────
(function scale() {
  const allX = DATA.protos.map(p => p.x).concat([DATA.query.x]);
  const allY = DATA.protos.map(p => p.y).concat([DATA.query.y]);
  const x0 = Math.min(...allX), x1 = Math.max(...allX);
  const y0 = Math.min(...allY), y1 = Math.max(...allY);
  const pw = W - 2*PAD, ph = H - 2*PAD;
  function sc(p) {
    p.cx = PAD + (p.x - x0) / (x1 - x0) * pw;
    p.cy = PAD + (p.y - y0) / (y1 - y0) * ph;
  }
  DATA.protos.forEach(sc);
  sc(DATA.query);
})();

// ── Draw a diamond (query marker) ────────────────────────────────────────────
function drawDiamond(cx, cy, r, fill, stroke) {
  ctx.beginPath();
  ctx.moveTo(cx,     cy - r);
  ctx.lineTo(cx + r, cy);
  ctx.lineTo(cx,     cy + r);
  ctx.lineTo(cx - r, cy);
  ctx.closePath();
  ctx.fillStyle   = fill;
  ctx.fill();
  ctx.strokeStyle = stroke;
  ctx.lineWidth   = 2;
  ctx.stroke();
}

// ── Main draw ────────────────────────────────────────────────────────────────
function draw() {
  ctx.clearRect(0, 0, W, H);

  // grid
  ctx.strokeStyle = '#14142a';
  ctx.lineWidth = 1;
  for (let i = 1; i < 9; i++) {
    const gx = PAD + (W - 2*PAD) * i / 9;
    const gy = PAD + (H - 2*PAD) * i / 9;
    ctx.beginPath(); ctx.moveTo(gx, PAD); ctx.lineTo(gx, H-PAD); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(PAD, gy); ctx.lineTo(W-PAD, gy); ctx.stroke();
  }

  // prototypes
  for (const p of DATA.protos) {
    if (hidden.has(p.domain)) continue;
    const isPred = p.label === DATA.query.pred;

    // highlight ring for the predicted prototype
    if (isPred) {
      ctx.beginPath();
      ctx.arc(p.cx, p.cy, R + 6, 0, 2*Math.PI);
      ctx.strokeStyle = '#ffffffaa';
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    ctx.beginPath();
    ctx.arc(p.cx, p.cy, isPred ? R + 2 : R, 0, 2*Math.PI);
    ctx.fillStyle   = p.color + (isPred ? 'ff' : 'bb');
    ctx.fill();
    ctx.strokeStyle = p.color;
    ctx.lineWidth   = isPred ? 2 : 1.4;
    ctx.stroke();

    // intent label
    ctx.font = '8px "Segoe UI", system-ui, sans-serif';
    ctx.textBaseline = 'middle';
    const lbl = p.label.replace(/_/g, ' ');
    const tw  = ctx.measureText(lbl).width;
    const tx  = p.cx + (isPred ? R + 5 : R + 3);
    ctx.fillStyle = 'rgba(8,8,22,0.80)';
    ctx.fillRect(tx - 1, p.cy - 6, tw + 4, 12);
    ctx.fillStyle = isPred ? '#ffffff' : p.color;
    ctx.fillText(lbl, tx + 1, p.cy);
  }

  // query diamond
  const q = DATA.query;
  drawDiamond(q.cx, q.cy, R + 4, '#ffffff', '#ffdd00');

  // "you" label on the query
  ctx.font = 'bold 9px "Segoe UI", system-ui, sans-serif';
  ctx.textBaseline = 'middle';
  const qlbl = 'you';
  const qtw  = ctx.measureText(qlbl).width;
  ctx.fillStyle = 'rgba(8,8,22,0.85)';
  ctx.fillRect(q.cx + R + 6, q.cy - 7, qtw + 6, 14);
  ctx.fillStyle = '#ffdd00';
  ctx.fillText(qlbl, q.cx + R + 9, q.cy);
}

draw();

// ── Legend ───────────────────────────────────────────────────────────────────
const legEl = document.getElementById('legend');
for (const [domain, color] of Object.entries(DATA.domainColors)) {
  const div = document.createElement('div');
  div.className = 'litem';
  div.innerHTML = '<div class="ldot" style="background:' + color + '"></div>' + domain;
  div.addEventListener('click', () => {
    if (hidden.has(domain)) hidden.delete(domain);
    else hidden.add(domain);
    div.classList.toggle('hidden', hidden.has(domain));
    draw();
  });
  legEl.appendChild(div);
}

// ── Tooltip ──────────────────────────────────────────────────────────────────
canvas.addEventListener('mousemove', e => {
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;

  // check query diamond first
  const q = DATA.query;
  if (Math.hypot(q.cx - mx, q.cy - my) < R + 6) {
    tip.querySelector('.t-label').textContent  = '"' + q.text + '"';
    tip.querySelector('.t-domain').textContent = '→ ' + q.pred.replace(/_/g, ' ');
    tip.style.borderColor = '#ffdd00';
    tip.style.left = (e.clientX + 16) + 'px';
    tip.style.top  = (e.clientY - 12) + 'px';
    tip.style.display = 'block';
    return;
  }

  let best = null, bestD = Infinity;
  for (const p of DATA.protos) {
    if (hidden.has(p.domain)) continue;
    const d = Math.hypot(p.cx - mx, p.cy - my);
    if (d < R + 5 && d < bestD) { bestD = d; best = p; }
  }
  if (best) {
    tip.querySelector('.t-label').textContent  = best.label.replace(/_/g, ' ');
    tip.querySelector('.t-domain').textContent = '● ' + best.domain;
    tip.style.borderColor = best.color;
    tip.style.left = (e.clientX + 16) + 'px';
    tip.style.top  = (e.clientY - 12) + 'px';
    tip.style.display = 'block';
  } else {
    tip.style.display = 'none';
  }
});
canvas.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
</script>
</body>
</html>"""
    )


def open_query_viz(proto_points, query_xy, query_text, pred_label, emb_np, top3=None):
    html = _make_query_html(proto_points, query_xy, query_text, pred_label, emb_np, top3=top3)
    path = Path(QUERY_VIZ_PATH)
    path.write_text(html, encoding="utf-8")
    # Under WSL the Linux webbrowser module has no registered browser.
    # Convert the path to a Windows path and open via cmd.exe instead.
    try:
        result = subprocess.run(
            ["wslpath", "-w", str(path.resolve())],
            capture_output=True, text=True, check=True,
        )
        win_path = result.stdout.strip()
        subprocess.Popen(
            ["cmd.exe", "/c", "start", "", win_path],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        webbrowser.open(path.resolve().as_uri())


# ============================================================================
# UTILITIES
# ============================================================================
def load_tokenizer(path):
    try:
        with safe_globals([transformers.models.bert.tokenization_bert.BertTokenizer]):
            tokenizer = torch.load(path, weights_only=False)
        return tokenizer
    except Exception:
        return BertTokenizer.from_pretrained("bert-base-uncased")


def load_metadata(path):
    with open(path, "r") as f:
        return json.load(f)


def save_prototypes(prototypes, path):
    torch.save(prototypes, path)


def save_prototype_metadata(id2label, label2id, path):
    with open(path, "w") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f, indent=2)


def load_prototype_metadata(path):
    with open(path, "r") as f:
        return json.load(f)


def print_info():
    print(f"Using device: {DEVICE}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Tokenizer path: {TOKENIZER_PATH}")
    print(f"Prototypes path: {PROTOTYPES_PATH}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Build and use saved prototypes for prototypical BERT inference")
    parser.add_argument("--build", action="store_true", help="Build prototypes from training data and save them")
    parser.add_argument("--interactive", action="store_true", help="Enter interactive classification mode")
    parser.add_argument("--sentence", type=str, help="Classify a single sentence and print the predicted intent")
    args = parser.parse_args()

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print_info()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"Tokenizer file not found: {TOKENIZER_PATH}")
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")

    metadata = load_metadata(METADATA_PATH)
    num_classes = metadata["num_classes"]
    id2label = {int(k): v for k, v in metadata["id2label"].items()}
    label2id = metadata["label2id"]

    tokenizer = load_tokenizer(TOKENIZER_PATH)
    model = PrototypicalBertNetwork().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    if args.build or not os.path.exists(PROTOTYPES_PATH):
        print("\nBuilding prototypes from CLINC training data...")
        train_texts, train_ids = load_clinc_train_data(CLINC_PATH, label2id)
        prototypes = build_prototypes(model, tokenizer, train_texts, train_ids, num_classes, DEVICE, batch_size=BATCH_SIZE)
        save_prototypes(prototypes, PROTOTYPES_PATH)
        save_prototype_metadata(id2label, label2id, PROTOTYPE_META_PATH)
        print(f"Saved prototypes to {PROTOTYPES_PATH}")
        print(f"Saved prototype metadata to {PROTOTYPE_META_PATH}")
    else:
        print(f"Loading saved prototypes from {PROTOTYPES_PATH}")
        prototypes = torch.load(PROTOTYPES_PATH, map_location=DEVICE)

    if args.sentence or args.interactive:
        protos_np = prototypes.cpu().numpy()

        def run_viz(text, top_idxs, top_dists, emb_np):
            label_idx  = top_idxs[0]
            pred_label = id2label[label_idx]
            # norm       = float(np.linalg.norm(emb_np))
            top3       = [(id2label[i], d) for i, d in zip(top_idxs, top_dists)]
            print(f"Top 3 closest intents:")
            for rank, (lbl, dist) in enumerate(top3, 1):
                marker = " <-- predicted" if rank == 1 else ""
                print(f"  #{rank}  {lbl:<30}  distance={dist:.4f}{marker}")
            # print(f"Embedding:        768-dim vector  |  norm={norm:.4f}  |  first 8 values: [{', '.join(f'{v:.4f}' for v in emb_np[0, :8])}]")
            print("Running t-SNE on prototypes + query...")
            proto_coords_2d, query_xy = project_with_tsne(protos_np, emb_np)
            # print(f"2D projection:    x={query_xy[0]:.4f}  y={query_xy[1]:.4f}")
            proto_points = build_proto_points(proto_coords_2d, id2label)
            open_query_viz(proto_points, query_xy, text, pred_label, emb_np, top3=top3)
            print(f"Link: {Path(QUERY_VIZ_PATH).resolve().as_uri()}")

        if args.sentence:
            top_idxs, top_dists, emb_np = classify_text(model, tokenizer, prototypes, args.sentence, DEVICE)
            run_viz(args.sentence, top_idxs, top_dists, emb_np)
            return

        print("\nEntering interactive mode. Type a sentence and press Enter. Empty line to quit.")
        while True:
            text = input("> ")
            if not text.strip():
                break
            top_idxs, top_dists, emb_np = classify_text(model, tokenizer, prototypes, text, DEVICE)
            run_viz(text, top_idxs, top_dists, emb_np)


if __name__ == "__main__":
    main()
