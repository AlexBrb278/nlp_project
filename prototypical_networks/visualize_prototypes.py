"""
Visualise the 150 CLINC intent prototypes in 2D using t-SNE.
Loads saved prototypes.pt and outputs a self-contained interactive HTML file.

Run from the prototypical_networks/ directory:
    python visualize_prototypes.py

No extra dependencies beyond sklearn, torch, numpy (already installed).
Output:  ./prototypical_bert_model/prototype_visualization.html
"""

import json
import torch
import numpy as np
from pathlib import Path
from sklearn.manifold import TSNE

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
PROTOTYPES_PATH = "./prototypical_bert_model/prototypes.pt"
METADATA_PATH   = "./prototypical_bert_model/metadata.json"
OUTPUT_HTML     = "./prototypical_bert_model/prototype_visualization.html"

# ─────────────────────────────────────────────────────────────────────────────
# CLINC DOMAIN GROUPINGS  (all 150 intents, each assigned to exactly one domain)
# ─────────────────────────────────────────────────────────────────────────────
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

# Tableau-10 palette + one extra (for 11 domains)
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

# ─────────────────────────────────────────────────────────────────────────────
# HTML TEMPLATE  (uses __POINTS__, __COLORS__, __TITLE__ as substitution tokens)
# ─────────────────────────────────────────────────────────────────────────────
HTML_TEMPLATE = r"""<!DOCTYPE html>
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
  #tip .t-label { font-size: 0.82rem; font-weight: 700; color: #eef; }
  #tip .t-domain { font-size: 0.73rem; color: #99b; margin-top: 2px; }
  #tip .t-dist   { font-size: 0.68rem; color: #667; margin-top: 1px; }
</style>
</head>
<body>
<h1>__TITLE__</h1>
<p class="subtitle">
  Each dot = one intent prototype &bull;
  position = mean BERT [CLS] embedding projected via t-SNE &bull;
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
const POINTS = __POINTS__;
const DOMAIN_COLORS = __COLORS__;

const canvas = document.getElementById('c');
const ctx    = canvas.getContext('2d');
const tip    = document.getElementById('tip');
const W = canvas.width, H = canvas.height;
const PAD = 44, R = 7;

let hidden     = new Set();
let showLabels = true;

// Scale raw t-SNE coords to canvas pixels once
(function scale() {
  const xs = POINTS.map(p => p.x), ys = POINTS.map(p => p.y);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = Math.min(...ys), y1 = Math.max(...ys);
  const pw = W - 2*PAD, ph = H - 2*PAD;
  for (const p of POINTS) {
    p.cx = PAD + (p.x - x0) / (x1 - x0) * pw;
    p.cy = PAD + (p.y - y0) / (y1 - y0) * ph;
  }
})();

function draw() {
  ctx.clearRect(0, 0, W, H);

  // subtle grid
  ctx.strokeStyle = '#14142a';
  ctx.lineWidth = 1;
  for (let i = 1; i < 9; i++) {
    const gx = PAD + (W - 2*PAD) * i / 9;
    const gy = PAD + (H - 2*PAD) * i / 9;
    ctx.beginPath(); ctx.moveTo(gx, PAD); ctx.lineTo(gx, H-PAD); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(PAD, gy); ctx.lineTo(W-PAD, gy); ctx.stroke();
  }

  // dots
  for (const p of POINTS) {
    if (hidden.has(p.domain)) continue;
    const col = p.color;
    ctx.beginPath();
    ctx.arc(p.cx, p.cy, R, 0, 2*Math.PI);
    ctx.fillStyle = col + 'bb';
    ctx.fill();
    ctx.strokeStyle = col;
    ctx.lineWidth = 1.4;
    ctx.stroke();
  }

  // labels  ("domain · intent name")
  if (showLabels) {
    ctx.font = '8px "Segoe UI", system-ui, sans-serif';
    ctx.textBaseline = 'middle';
    for (const p of POINTS) {
      if (hidden.has(p.domain)) continue;
      const text = p.label.replace(/_/g, ' ');
      const tw   = ctx.measureText(text).width;
      const tx   = p.cx + R + 3;
      const ty   = p.cy;
      ctx.fillStyle = 'rgba(8,8,22,0.80)';
      ctx.fillRect(tx - 1, ty - 6, tw + 4, 12);
      ctx.fillStyle = p.color;
      ctx.fillText(text, tx + 1, ty);
    }
  }
}

draw();

// ── Legend ──────────────────────────────────────────────────────────────────
const legEl = document.getElementById('legend');
for (const [domain, color] of Object.entries(DOMAIN_COLORS)) {
  const div = document.createElement('div');
  div.className = 'litem';
  div.dataset.domain = domain;
  div.innerHTML =
    `<div class="ldot" style="background:${color}"></div>${domain}`;
  div.addEventListener('click', () => {
    if (hidden.has(domain)) hidden.delete(domain);
    else hidden.add(domain);
    div.classList.toggle('hidden', hidden.has(domain));
    draw();
  });
  legEl.appendChild(div);
}

// ── Labels toggle ────────────────────────────────────────────────────────────
const lblBtn = document.getElementById('lbl-btn');
lblBtn.addEventListener('click', () => {
  showLabels = !showLabels;
  lblBtn.style.opacity = showLabels ? '1' : '0.35';
  draw();
});

// ── Tooltip on hover ─────────────────────────────────────────────────────────
canvas.addEventListener('mousemove', e => {
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  let best = null, bestD = Infinity;
  for (const p of POINTS) {
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
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def build_label_domain_map():
    m = {}
    for domain, intents in DOMAIN_INTENTS.items():
        for intent in intents:
            m[intent] = domain
    return m


def main():
    print("Loading prototypes…")
    prototypes = torch.load(PROTOTYPES_PATH, map_location="cpu").numpy()
    print(f"  shape: {prototypes.shape}")

    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    id2label    = {int(k): v for k, v in metadata["id2label"].items()}
    num_classes = metadata["num_classes"]

    label_to_domain = build_label_domain_map()

    unmapped = [id2label[i] for i in range(num_classes) if id2label[i] not in label_to_domain]
    if unmapped:
        print(f"  Warning – {len(unmapped)} unmapped intents: {unmapped}")
        DOMAIN_COLORS["other"] = "#cccccc"
        for intent in unmapped:
            label_to_domain[intent] = "other"

    # t-SNE ───────────────────────────────────────────────────────────────────
    print("Running t-SNE (perplexity=30, 1000 iterations)…")
    tsne   = TSNE(n_components=2, perplexity=30, random_state=42,
                  max_iter=1000, verbose=1)
    coords = tsne.fit_transform(prototypes)
    print(f"  KL divergence: {tsne.kl_divergence_:.4f}")

    # Build point list ─────────────────────────────────────────────────────────
    points = []
    for i in range(num_classes):
        label  = id2label[i]
        domain = label_to_domain.get(label, "other")
        color  = DOMAIN_COLORS.get(domain, "#cccccc")
        points.append({
            "x":      float(coords[i, 0]),
            "y":      float(coords[i, 1]),
            "label":  label,
            "domain": domain,
            "color":  color,
        })

    # Render HTML ──────────────────────────────────────────────────────────────
    title      = "Prototype Space — Intent Relationships (t-SNE)"
    html = (HTML_TEMPLATE
            .replace("__POINTS__", json.dumps(points))
            .replace("__COLORS__", json.dumps(DOMAIN_COLORS))
            .replace("__TITLE__",  title))

    out = Path(OUTPUT_HTML)
    out.write_text(html, encoding="utf-8")
    print(f"\nSaved → {out.resolve()}")
    print("Open that file in any browser.")
    print("Hover dots to see intent names; click legend to toggle domains.")


if __name__ == "__main__":
    main()
