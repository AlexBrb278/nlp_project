"""
Interactive Prototypical Networks BERT - predict intent for typed sentences.

pytorch format:
    python test_interactive.py --model_dir prototypical_networks/prototypical_bert_model \
        --clinc_path oos-eval/data/data_full.json

huggingface format (contrastive + prototypical):
    python test_interactive.py \
        --model_dir models/contrastive_learning \
        --model_format huggingface \
        --embeddings results/contrastive_prototypical/train_embeddings.pt \
        --clinc_path oos-eval/data/data_full.json \
        --show_domains --html
"""

import argparse
import json
import logging
import os

import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, AutoModel, AutoTokenizer

logging.getLogger("nltk").setLevel(logging.ERROR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same palette as visualize_prototypes.py
DOMAIN_COLORS = {
    "auto_and_commute":   "#4e79a7",
    "banking":            "#f28e2b",
    "credit_cards":       "#e15759",
    "home":               "#76b7b2",
    "kitchen_and_dining": "#59a14f",
    "meta":               "#edc948",
    "small_talk":         "#b07aa1",
    "travel":             "#ff9da7",
    "utility":            "#9c755f",
    "work":               "#bab0ac",
    "oos":                "#667788",
}

TSNE_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>__TITLE__</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0f0f1e;
    color: #dde;
    font-family: 'Segoe UI', system-ui, sans-serif;
    display: flex;
    flex-direction: column;
    align-items: center;
    min-height: 100vh;
    padding: 24px 16px 40px;
  }
  h1 {
    font-size: 1.25rem;
    font-weight: 600;
    color: #c8d8ee;
    letter-spacing: 0.04em;
    margin-bottom: 4px;
  }
  .subtitle {
    font-size: 0.75rem;
    color: #6677aa;
    margin-bottom: 20px;
    text-align: center;
  }
  #wrap {
    display: flex;
    gap: 18px;
    align-items: flex-start;
  }
  canvas {
    background: #0d0d20;
    border-radius: 10px;
    border: 1px solid #1e2044;
    cursor: crosshair;
  }
  #sidebar {
    background: #0d0d20;
    border: 1px solid #1e2044;
    border-radius: 10px;
    padding: 16px 18px;
    min-width: 175px;
    max-width: 185px;
  }
  #sidebar h2 {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #667;
    margin-bottom: 12px;
  }
  .litem {
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 4px 6px;
    border-radius: 5px;
    cursor: pointer;
    font-size: 0.78rem;
    color: #bbc;
    transition: background 0.12s, opacity 0.12s;
    user-select: none;
  }
  .litem:hover { background: #171733; }
  .litem.hidden { opacity: 0.3; }
  .ldot {
    width: 11px; height: 11px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  #tip {
    position: fixed;
    background: rgba(10,10,30,0.92);
    border: 1px solid #445;
    border-radius: 7px;
    padding: 7px 13px;
    pointer-events: none;
    display: none;
    z-index: 9999;
  }
  #tip .t-label  { font-size: 0.82rem; font-weight: 700; color: #eef; }
  #tip .t-domain { font-size: 0.73rem; color: #99b; margin-top: 2px; }
</style>
</head>
<body>
<h1>__TITLE__</h1>
<p class="subtitle">
  Each dot = one intent prototype &bull;
  position = mean encoder [CLS] embedding projected via t-SNE &bull;
  colour = domain &bull;
  hover to inspect &bull; click legend to toggle
</p>
<div id="wrap">
  <canvas id="c" width="820" height="700"></canvas>
  <div id="sidebar">
    <h2>Domains</h2>
    <div id="legend"></div>
    <hr style="border-color:#1e2044;margin:14px 0 12px;">
    <div id="lbl-btn" style="display:flex;align-items:center;gap:9px;padding:5px 6px;border-radius:5px;cursor:pointer;font-size:0.78rem;color:#bbc;user-select:none;transition:opacity 0.12s;">
      <div style="width:11px;height:11px;border-radius:2px;background:#778;flex-shrink:0;font-size:9px;line-height:11px;text-align:center;color:#0f0f1e;">A</div>
      <span>Labels</span>
    </div>
  </div>
</div>
<div id="tip">
  <div class="t-label"></div>
  <div class="t-domain"></div>
</div>
<script>
const POINTS  = __POINTS__;
const QUERIES = __QUERIES__;
const DOMAIN_COLORS = __COLORS__;
const canvas = document.getElementById('c');
const ctx    = canvas.getContext('2d');
const tip    = document.getElementById('tip');
const W = canvas.width, H = canvas.height;
const PAD = 44, R = 7, QR = 10;
let hidden = new Set(), showLabels = true;

// Scale all points (prototypes + queries) using prototype bounds so queries
// land in the same coordinate space.
(function scale() {
  const xs = POINTS.map(p => p.x), ys = POINTS.map(p => p.y);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = Math.min(...ys), y1 = Math.max(...ys);
  const pw = W - 2*PAD, ph = H - 2*PAD;
  const toCanvas = p => {
    p.cx = PAD + (p.x - x0) / (x1 - x0) * pw;
    p.cy = PAD + (p.y - y0) / (y1 - y0) * ph;
  };
  POINTS.forEach(toCanvas);
  QUERIES.forEach(toCanvas);
})();

function drawDiamond(cx, cy, r, fill, stroke) {
  ctx.beginPath();
  ctx.moveTo(cx, cy - r);
  ctx.lineTo(cx + r, cy);
  ctx.lineTo(cx, cy + r);
  ctx.lineTo(cx - r, cy);
  ctx.closePath();
  ctx.fillStyle   = fill;   ctx.fill();
  ctx.strokeStyle = stroke; ctx.lineWidth = 2; ctx.stroke();
}

function draw() {
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = '#14142a'; ctx.lineWidth = 1;
  for (let i = 1; i < 9; i++) {
    const gx = PAD + (W-2*PAD)*i/9, gy = PAD + (H-2*PAD)*i/9;
    ctx.beginPath(); ctx.moveTo(gx,PAD); ctx.lineTo(gx,H-PAD); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(PAD,gy); ctx.lineTo(W-PAD,gy); ctx.stroke();
  }
  // prototype dots
  for (const p of POINTS) {
    if (hidden.has(p.domain)) continue;
    ctx.beginPath(); ctx.arc(p.cx, p.cy, R, 0, 2*Math.PI);
    ctx.fillStyle = p.color+'bb'; ctx.fill();
    ctx.strokeStyle = p.color; ctx.lineWidth = 1.4; ctx.stroke();
  }
  // prototype labels
  if (showLabels) {
    ctx.font = '8px "Segoe UI", system-ui, sans-serif';
    ctx.textBaseline = 'middle';
    for (const p of POINTS) {
      if (hidden.has(p.domain)) continue;
      const text = p.label.replace(/_/g,' ');
      const tw = ctx.measureText(text).width;
      const tx = p.cx+R+3, ty = p.cy;
      ctx.fillStyle = 'rgba(8,8,22,0.80)';
      ctx.fillRect(tx-1, ty-6, tw+4, 12);
      ctx.fillStyle = p.color; ctx.fillText(text, tx+1, ty);
    }
  }
  // query diamonds
  for (const q of QUERIES) {
    drawDiamond(q.cx, q.cy, QR, 'rgba(255,255,180,0.25)', '#ffffb0');
    ctx.font = 'bold 9px "Segoe UI", system-ui, sans-serif';
    ctx.textBaseline = 'middle';
    const short = q.text.length > 28 ? q.text.slice(0,26)+'…' : q.text;
    const tw = ctx.measureText(short).width;
    const tx = q.cx + QR + 4, ty = q.cy;
    ctx.fillStyle = 'rgba(8,8,22,0.85)';
    ctx.fillRect(tx-2, ty-7, tw+6, 14);
    ctx.fillStyle = '#ffffb0'; ctx.fillText(short, tx+1, ty);
  }
}
draw();

const legEl = document.getElementById('legend');
for (const [domain, color] of Object.entries(DOMAIN_COLORS)) {
  const div = document.createElement('div');
  div.className = 'litem'; div.dataset.domain = domain;
  div.innerHTML = `<div class="ldot" style="background:${color}"></div>${domain.replace(/_/g,' ')}`;
  div.addEventListener('click', () => {
    if (hidden.has(domain)) hidden.delete(domain); else hidden.add(domain);
    div.classList.toggle('hidden', hidden.has(domain)); draw();
  });
  legEl.appendChild(div);
}
document.getElementById('lbl-btn').addEventListener('click', () => {
  showLabels = !showLabels;
  document.getElementById('lbl-btn').style.opacity = showLabels ? '1' : '0.35';
  draw();
});

// Tooltip — handles both prototypes and query diamonds
canvas.addEventListener('mousemove', e => {
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX-rect.left, my = e.clientY-rect.top;
  let best = null, bestD = Infinity, isQuery = false;
  for (const p of POINTS) {
    if (hidden.has(p.domain)) continue;
    const d = Math.hypot(p.cx-mx, p.cy-my);
    if (d < R+5 && d < bestD) { bestD=d; best=p; isQuery=false; }
  }
  for (const q of QUERIES) {
    const d = Math.hypot(q.cx-mx, q.cy-my);
    if (d < QR+5 && d < bestD) { bestD=d; best=q; isQuery=true; }
  }
  if (best) {
    tip.querySelector('.t-label').textContent  = isQuery ? `"${best.text}"` : best.label.replace(/_/g,' ');
    tip.querySelector('.t-domain').textContent = isQuery ? `→ ${best.top_intent.replace(/_/g,' ')}` : '● '+best.domain.replace(/_/g,' ');
    tip.style.borderColor = isQuery ? '#ffffb0' : best.color;
    tip.style.left = (e.clientX+16)+'px'; tip.style.top = (e.clientY-12)+'px';
    tip.style.display = 'block';
  } else { tip.style.display = 'none'; }
});
canvas.addEventListener('mouseleave', () => { tip.style.display='none'; });
</script>
</body>
</html>
"""


def generate_tsne_state(prototypes, id2label, intent2domain, model_dir):
    """Run t-SNE once and return state dict for reuse when queries arrive."""
    from sklearn.manifold import TSNE
    import numpy as np

    proto_np    = prototypes.cpu().numpy()
    num_classes = proto_np.shape[0]

    print("Running t-SNE on prototypes (perplexity=30, 1000 iterations)…")
    coords = TSNE(n_components=2, perplexity=30, random_state=42,
                  max_iter=1000, verbose=0).fit_transform(proto_np)

    points = []
    for i in range(num_classes):
        label  = id2label[i]
        domain = intent2domain.get(label, "oos")
        color  = DOMAIN_COLORS.get(domain, "#667788")
        points.append({"x": float(coords[i, 0]), "y": float(coords[i, 1]),
                        "label": label, "domain": domain, "color": color})

    projector = build_tsne_projector(proto_np, coords)

    return {
        "proto_np":  proto_np,
        "coords":    coords,
        "points":    points,
        "projector": projector,
        "title":     f"Prototype Space — {os.path.basename(model_dir)} (t-SNE)",
    }


def build_tsne_projector(proto_np, coords, n_neighbors=15):
    """Fit a KNN regressor: 768-dim embedding → 2D t-SNE coords."""
    from sklearn.neighbors import KNeighborsRegressor
    reg = KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance")
    reg.fit(proto_np, coords)
    return reg


def project_query(embedding_np, projector):
    """Project a single query embedding to 2D using the fitted projector."""
    return projector.predict(embedding_np.reshape(1, -1))[0]


def save_tsne_html(tsne_state, query_points, tsne_path):
    """Re-render t-SNE HTML with current query diamonds and write to disk."""
    html = (TSNE_TEMPLATE
            .replace("__POINTS__",  json.dumps(tsne_state["points"]))
            .replace("__QUERIES__", json.dumps(query_points))
            .replace("__COLORS__",  json.dumps(DOMAIN_COLORS))
            .replace("__TITLE__",   tsne_state["title"]))
    with open(tsne_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  [t-SNE updated → {tsne_path}]")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Intent Predictions</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0f0f1e;
    color: #dde;
    font-family: 'Segoe UI', system-ui, sans-serif;
    padding: 28px 32px 48px;
    min-height: 100vh;
  }
  h1 {
    font-size: 1.15rem;
    font-weight: 600;
    color: #c8d8ee;
    letter-spacing: 0.04em;
    margin-bottom: 4px;
  }
  .subtitle {
    font-size: 0.73rem;
    color: #6677aa;
    margin-bottom: 28px;
  }
  .query-card {
    background: #0d0d20;
    border: 1px solid #1e2044;
    border-radius: 10px;
    padding: 16px 20px 18px;
    margin-bottom: 18px;
  }
  .query-text {
    font-size: 0.88rem;
    color: #99aacc;
    margin-bottom: 14px;
    font-style: italic;
  }
  .bar-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
  }
  .bar-row:last-child { margin-bottom: 0; }
  .bar-label {
    width: 200px;
    flex-shrink: 0;
    font-size: 0.78rem;
    color: #bbc;
    text-align: right;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .bar-track {
    flex: 1;
    background: #14142a;
    border-radius: 4px;
    height: 18px;
    overflow: hidden;
    position: relative;
  }
  .bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s ease;
  }
  .bar-score {
    width: 90px;
    flex-shrink: 0;
    font-size: 0.72rem;
    color: #667;
  }
  .top-marker {
    font-size: 0.7rem;
    color: #c8d8ee;
    background: #1e2044;
    border-radius: 3px;
    padding: 1px 5px;
    margin-left: 6px;
  }
</style>
</head>
<body>
<h1>Intent Predictions</h1>
<p class="subtitle">Prototypical network — nearest prototype by Euclidean distance in BERT embedding space</p>
<div id="cards">__CARDS__</div>
</body>
</html>
"""


class PrototypicalBertNetwork(nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return out.last_hidden_state[:, 0, :]


def build_prototypes(embeddings, labels, num_classes):
    protos = torch.zeros(num_classes, embeddings.shape[1])
    for c in range(num_classes):
        mask = torch.tensor(labels) == c
        if mask.sum() > 0:
            protos[c] = embeddings[mask].mean(dim=0)
    return protos


def load_domains(domains_path):
    with open(domains_path) as f:
        raw = json.load(f)
    return {intent: domain for domain, intents in raw.items() for intent in intents}


def load_model(args, clinc_path):
    if args.model_format == "pytorch":
        model = PrototypicalBertNetwork().to(DEVICE)
        model.load_state_dict(torch.load(
            os.path.join(args.model_dir, "model.pth"), map_location=DEVICE))
        tokenizer = torch.load(
            os.path.join(args.model_dir, "tokenizer.pth"), weights_only=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        model = PrototypicalBertNetwork().to(DEVICE)
        model.bert = AutoModel.from_pretrained(args.model_dir).to(DEVICE)

    model.eval()

    with open(clinc_path) as f:
        data = json.load(f)
    all_labels  = sorted(set(item[1] for item in data["train"]))
    label2id    = {l: i for i, l in enumerate(all_labels)}
    id2label    = {i: l for l, i in label2id.items()}
    num_classes = len(all_labels)

    if args.model_format == "pytorch":
        prototypes = torch.load(
            os.path.join(args.model_dir, "prototypes.pt"), map_location=DEVICE)
        print(f"Prototypes loaded — shape: {prototypes.shape}")
    else:
        if not args.embeddings:
            raise ValueError("--embeddings is required for huggingface format.")
        saved      = torch.load(args.embeddings, map_location="cpu")
        prototypes = build_prototypes(
            saved["embeddings"], saved["labels"], num_classes).to(DEVICE)
        print(f"Prototypes built from embeddings — shape: {prototypes.shape}")

    print(f"Ready — {num_classes} intents | device: {DEVICE}")
    return model, tokenizer, prototypes, id2label


def predict(text, model, tokenizer, prototypes, id2label, max_length=64, top_k=3):
    enc = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        embedding = model(input_ids, attention_mask)
        distances = torch.cdist(embedding, prototypes)[0]
        scores    = F.softmax(-distances, dim=0)

    top_ids = torch.argsort(distances)[:top_k].cpu().tolist()
    preds   = [(id2label[i], scores[i].item(), distances[i].item()) for i in top_ids]
    emb_np  = embedding[0].cpu().numpy()
    return preds, emb_np


def render_card(query, preds, intent2domain):
    max_score = max(p[1] for p in preds)
    rows = ""
    for rank, (intent, score, dist) in enumerate(preds):
        domain  = intent2domain.get(intent, "oos")
        color   = DOMAIN_COLORS.get(domain, "#667788")
        pct     = score / max_score * 100
        label   = intent.replace("_", " ")
        if intent2domain:
            label += f" <span style='color:{color};font-size:0.68rem;'>· {domain.replace('_',' ')}</span>"
        marker  = "<span class='top-marker'>top</span>" if rank == 0 else ""
        rows += f"""
    <div class="bar-row">
      <div class="bar-label">{label}{marker}</div>
      <div class="bar-track">
        <div class="bar-fill" style="width:{pct:.1f}%;background:{color};"></div>
      </div>
      <div class="bar-score">score {score:.4f}<br><span style="color:#445;">dist {dist:.2f}</span></div>
    </div>"""

    return f"""
<div class="query-card">
  <div class="query-text">"{query}"</div>
  {rows}
</div>"""


def save_html(history, intent2domain, html_path):
    cards = "".join(render_card(q, p, intent2domain) for q, p in reversed(history))
    html  = HTML_TEMPLATE.replace("__CARDS__", cards)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  [HTML updated → {html_path}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",    default="prototypical_networks/prototypical_bert_model")
    parser.add_argument("--model_format", choices=["pytorch", "huggingface"], default="pytorch")
    parser.add_argument("--embeddings",   default=None)
    parser.add_argument("--clinc_path",   default="oos-eval/data/data_full.json")
    parser.add_argument("--domains_path", default=None,
                        help="Path to domains.json (auto-detected from clinc_path dir if omitted)")
    parser.add_argument("--show_domains", action="store_true",
                        help="Print domain next to each intent in the terminal")
    parser.add_argument("--html",         action="store_true",
                        help="Save/update an HTML file after each query")
    parser.add_argument("--html_path",    default="predictions.html")
    parser.add_argument("--tsne",         action="store_true",
                        help="Generate a t-SNE prototype visualization HTML before starting")
    parser.add_argument("--tsne_path",    default="tsne_prototypes.html")
    parser.add_argument("--top_k",        type=int, default=3)
    parser.add_argument("--max_length",   type=int, default=64,
                        help="64 for BERT, 128 for CANINE")
    args = parser.parse_args()

    nltk.download("wordnet",                    quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    nltk.download("omw-1.4",                   quiet=True)
    nltk.download = lambda *a, **kw: None

    intent2domain = {}
    if args.show_domains or args.html:
        domains_path  = args.domains_path or os.path.join(
            os.path.dirname(args.clinc_path), "domains.json")
        intent2domain = load_domains(domains_path)

    print(f"Loading model ({args.model_format}) from {args.model_dir}...")
    model, tokenizer, prototypes, id2label = load_model(args, args.clinc_path)

    tsne_state   = None
    tsne_queries = []
    if args.tsne:
        tsne_state = generate_tsne_state(prototypes, id2label, intent2domain, args.model_dir)
        save_tsne_html(tsne_state, tsne_queries, args.tsne_path)
        print(f"t-SNE → {args.tsne_path}  (open in browser, refresh after each query)\n")

    if args.html:
        print(f"HTML  → {args.html_path}  (open in browser, refresh after each query)\n")

    print("Type a sentence and press Enter.  'quit' / 'exit' to stop.\n")

    history = []

    while True:
        try:
            text = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not text:
            continue
        if text.lower() in ("quit", "exit"):
            break

        preds, emb_np = predict(text, model, tokenizer, prototypes, id2label,
                                max_length=args.max_length, top_k=args.top_k)

        for rank, (intent, score, dist) in enumerate(preds, 1):
            marker     = "→" if rank == 1 else " "
            domain_str = f"  [{intent2domain.get(intent, 'oos')}]" if args.show_domains else ""
            print(f"  {marker} #{rank}  {intent:<30}  score={score:.4f}  dist={dist:.4f}{domain_str}")
        print()

        if args.html:
            history.append((text, preds))
            save_html(history, intent2domain, args.html_path)

        if args.tsne:
            pos = project_query(emb_np, tsne_state["projector"])
            tsne_queries.append({"x": float(pos[0]), "y": float(pos[1]),
                                  "text": text, "top_intent": preds[0][0]})
            save_tsne_html(tsne_state, tsne_queries, args.tsne_path)


if __name__ == "__main__":
    main()
