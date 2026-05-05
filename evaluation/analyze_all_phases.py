# Analyse und Plot-Generierung für alle 5 Evaluierungsphasen.
# Erzeugt phasenspezifische Plots und phasenübergreifende Vergleiche.

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "images"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "font.family": "DejaVu Sans",
})

COLORS = {
    "gemma2": "#4285F4",
    "qwen2.5": "#EA4335",
    "mistral": "#FBBC05",
    "llama3.1": "#34A853",
    "phi4": "#FF6D01",
    "qwen2.5:14b": "#8B5CF6",
    "llama-3.3-70b-versatile": "#06B6D4",
    "gemini-2.5-flash": "#EC4899",
}

print("Lade Daten...")
df1 = pd.read_csv("data/evaluation_logs/phase1_small_models_grid.csv")
df2 = pd.read_csv("data/evaluation_logs/phase2_large_models_grid.csv")
df3 = pd.read_csv("data/evaluation_logs/phase3_head_to_head.csv")
df4 = pd.read_csv("data/evaluation_logs/phase4_local_vs_cloud.csv")
df5 = pd.read_csv("data/evaluation_logs/phase5_topk_fine_grained.csv")

# Fehlerhafte Einträge für Scoring herausfiltern

for df in [df1, df2, df3, df4, df5]:
    df["overall_score"] = pd.to_numeric(df["overall_score"], errors="coerce")

df1s = df1[not df1["is_error"]].copy()
df2s = df2[not df2["is_error"]].copy()
df3s = df3[not df3["is_error"]].copy()
df4s = df4[not df4["is_error"]].copy()
df5s = df5[not df5["is_error"]].copy()

print(f"  Phase 1: {len(df1s)}/{len(df1)} bewertet")
print(f"  Phase 2: {len(df2s)}/{len(df2)} bewertet")
print(f"  Phase 3: {len(df3s)}/{len(df3)} bewertet")
print(f"  Phase 4: {len(df4s)}/{len(df4)} bewertet")
print(f"  Phase 5: {len(df5s)}/{len(df5)} bewertet")

# Phase 1: Kleine Modelle Grid Search
print("\n=== PHASE 1 PLOTS ===")

# 1a) Durchschnittlicher Score pro Modell
fig, ax = plt.subplots(figsize=(8, 5))
p1_model = df1s.groupby("model")["overall_score"].mean().sort_values(ascending=False)
bars = ax.bar(p1_model.index, p1_model.values, color=[COLORS.get(m, "#888") for m in p1_model.index])
ax.set_ylabel("Mean Score")
ax.set_title("Phase 1: Durchschnittlicher Score pro Modell (7B-Klasse)")
ax.set_ylim(75, 100)
for bar, val in zip(bars, p1_model.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f"{val:.1f}", ha="center", va="bottom", fontsize=9)
plt.savefig(f"{OUT}/phase1_model_scores.png")
plt.close()

# 1b) Einfluss der Temperature
fig, ax = plt.subplots(figsize=(8, 5))
for model in sorted(df1s["model"].unique()):
    m = df1s[df1s["model"] == model]
    t_scores = m.groupby("temperature")["overall_score"].mean()
    ax.plot(t_scores.index, t_scores.values, marker="o", label=model, color=COLORS.get(model, "#888"))
ax.set_xlabel("Temperature")
ax.set_ylabel("Mean Score")
ax.set_title("Phase 1: Einfluss der Temperature auf den Score (7B)")
ax.legend()
ax.set_ylim(60, 100)
plt.savefig(f"{OUT}/phase1_temperature_effect.png")
plt.close()

# 1c) Einfluss von Top-K
fig, ax = plt.subplots(figsize=(8, 5))
for model in sorted(df1s["model"].unique()):
    m = df1s[df1s["model"] == model]
    k_scores = m.groupby("top_k")["overall_score"].mean()
    ax.plot(k_scores.index, k_scores.values, marker="s", label=model, color=COLORS.get(model, "#888"))
ax.set_xlabel("Top-K (Retrieval)")
ax.set_ylabel("Mean Score")
ax.set_title("Phase 1: Einfluss von Top-K auf den Score (7B)")
ax.legend()
ax.set_ylim(70, 100)
ax.set_xticks([3, 5, 7, 9])
plt.savefig(f"{OUT}/phase1_topk_effect.png")
plt.close()

# 1d) Einfluss der Repeat Penalty
fig, ax = plt.subplots(figsize=(8, 5))
for model in sorted(df1s["model"].unique()):
    m = df1s[df1s["model"] == model]
    rp_scores = m.groupby("repeat_penalty")["overall_score"].mean()
    ax.plot(rp_scores.index, rp_scores.values, marker="^", label=model, color=COLORS.get(model, "#888"))
ax.set_xlabel("Repeat Penalty")
ax.set_ylabel("Mean Score")
ax.set_title("Phase 1: Einfluss der Repeat Penalty auf den Score (7B)")
ax.legend()
ax.set_ylim(75, 100)
plt.savefig(f"{OUT}/phase1_repeat_penalty_effect.png")
plt.close()

# 1e) Heatmap: Beste Konfiguration pro Modell (temp x rp, gemittelt über top_k)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for idx, model in enumerate(sorted(df1s["model"].unique())):
    ax = axes[idx // 2][idx % 2]
    m = df1s[df1s["model"] == model]
    pivot = m.pivot_table(values="overall_score", index="repeat_penalty", columns="temperature", aggfunc="mean")
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=70, vmax=100, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{r}" for r in pivot.index])
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Repeat Penalty")
    ax.set_title(f"{model}")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i, j]:.1f}", ha="center", va="center", fontsize=8)
fig.suptitle("Phase 1: Score-Heatmap (Temperature × Repeat Penalty, gemittelt über Top-K)", fontsize=13)
fig.colorbar(im, ax=axes, shrink=0.6, label="Mean Score")
plt.savefig(f"{OUT}/phase1_heatmaps.png")
plt.close()

# 1f) Halluzinationsrate nach Temperature
fig, ax = plt.subplots(figsize=(8, 5))
hall_by_temp = df1s.groupby("temperature")["hallucination_free"].mean() * 100
ax.bar(hall_by_temp.index.astype(str), hall_by_temp.values, color=["#22c55e", "#84cc16", "#eab308", "#ef4444"])
ax.set_ylabel("Halluzinationsfrei (%)")
ax.set_xlabel("Temperature")
ax.set_title("Phase 1: Halluzinationsfreiheit nach Temperature (7B)")
ax.set_ylim(80, 102)
for i, (t, v) in enumerate(hall_by_temp.items()):
    ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=9)
plt.savefig(f"{OUT}/phase1_hallucination_by_temp.png")
plt.close()

print("  Phase 1: 6 Plots gespeichert")

# Phase 2: Große Modelle Grid Search
print("\n=== PHASE 2 PLOTS ===")

# 2a) Durchschnittlicher Score pro Modell
fig, ax = plt.subplots(figsize=(6, 5))
p2_model = df2s.groupby("model")["overall_score"].mean().sort_values(ascending=False)
bars = ax.bar(p2_model.index, p2_model.values, color=[COLORS.get(m, "#888") for m in p2_model.index])
ax.set_ylabel("Mean Score")
ax.set_title("Phase 2: Durchschnittlicher Score pro Modell (14B-Klasse)")
ax.set_ylim(85, 100)
for bar, val in zip(bars, p2_model.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f"{val:.1f}", ha="center", va="bottom", fontsize=9)
plt.savefig(f"{OUT}/phase2_model_scores.png")
plt.close()

# 2b) Einfluss von Top-K (14B)
fig, ax = plt.subplots(figsize=(8, 5))
for model in sorted(df2s["model"].unique()):
    m = df2s[df2s["model"] == model]
    k_scores = m.groupby("top_k")["overall_score"].mean()
    ax.plot(k_scores.index, k_scores.values, marker="s", label=model, color=COLORS.get(model, "#888"), linewidth=2)
ax.set_xlabel("Top-K (Retrieval)")
ax.set_ylabel("Mean Score")
ax.set_title("Phase 2: Einfluss von Top-K auf den Score (14B)")
ax.legend()
ax.set_ylim(85, 100)
ax.set_xticks([3, 5, 7, 9])
plt.savefig(f"{OUT}/phase2_topk_effect.png")
plt.close()

# 2c) Einfluss der Temperature (14B)
fig, ax = plt.subplots(figsize=(8, 5))
for model in sorted(df2s["model"].unique()):
    m = df2s[df2s["model"] == model]
    t_scores = m.groupby("temperature")["overall_score"].mean()
    ax.plot(t_scores.index, t_scores.values, marker="o", label=model, color=COLORS.get(model, "#888"), linewidth=2)
ax.set_xlabel("Temperature")
ax.set_ylabel("Mean Score")
ax.set_title("Phase 2: Einfluss der Temperature auf den Score (14B)")
ax.legend()
ax.set_ylim(80, 100)
plt.savefig(f"{OUT}/phase2_temperature_effect.png")
plt.close()

# 2d) Heatmaps (14B)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for idx, model in enumerate(sorted(df2s["model"].unique())):
    ax = axes[idx]
    m = df2s[df2s["model"] == model]
    pivot = m.pivot_table(values="overall_score", index="repeat_penalty", columns="temperature", aggfunc="mean")
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=85, vmax=100, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{r}" for r in pivot.index])
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Repeat Penalty")
    ax.set_title(f"{model}")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i, j]:.1f}", ha="center", va="center", fontsize=9)
fig.suptitle("Phase 2: Score-Heatmap (Temperature × Repeat Penalty, 14B)", fontsize=13)
fig.colorbar(im, ax=axes, shrink=0.8, label="Mean Score")
plt.savefig(f"{OUT}/phase2_heatmaps.png")
plt.close()

# 2e) GPU stability
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for idx, model in enumerate(sorted(df2s["model"].unique())):
    ax = axes[idx]
    m = df2s[df2s["model"] == model].reset_index(drop=True)
    ax.scatter(range(len(m)), m["generation_time_sec"], alpha=0.3, s=8, color=COLORS.get(model, "#888"))
    # Gleitender Durchschnitt
    window = min(50, len(m) // 5)
    if window > 1:
        rolling = m["generation_time_sec"].rolling(window).mean()
        ax.plot(range(len(m)), rolling, color="red", linewidth=1.5, label=f"Gleitender Schnitt ({window})")
    ax.set_xlabel("Evaluation #")
    ax.set_ylabel("Generation Time (s)")
    ax.set_title(f"{model}: GPU-Stabilität")
    ax.legend()
fig.suptitle("Phase 2: Generierungszeit über den Evaluierungsverlauf (GPU-Stress-Check)", fontsize=13)
plt.savefig(f"{OUT}/phase2_gpu_stability.png")
plt.close()

print("  Phase 2: 5 Plots gespeichert")

# Phase 3: Head-to-Head
print("\n=== PHASE 3 PLOTS ===")

# 3a) Leaderboard
fig, ax = plt.subplots(figsize=(10, 5))
p3_model = df3s.groupby("model")["overall_score"].mean().sort_values(ascending=True)
colors_list = [COLORS.get(m, "#888") for m in p3_model.index]
bars = ax.barh(p3_model.index, p3_model.values, color=colors_list)
ax.set_xlabel("Mean Score")
ax.set_title("Phase 3: Head-to-Head Ranking (beste Config pro Modell)")
ax.set_xlim(75, 102)
for bar, val in zip(bars, p3_model.values):
    ax.text(val + 0.3, bar.get_y() + bar.get_height()/2, f"{val:.1f}", va="center", fontsize=9)
plt.savefig(f"{OUT}/phase3_leaderboard.png")
plt.close()

# 3b) Heatmap pro Frage (alle Modelle)
fig, ax = plt.subplots(figsize=(14, 10))
pivot3 = df3s.pivot_table(values="overall_score", index="question", columns="model", aggfunc="mean")
q_short = [q[:60] + "..." if len(q) > 60 else q for q in pivot3.index]
im = ax.imshow(pivot3.values, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(len(pivot3.columns)))
ax.set_xticklabels(pivot3.columns, rotation=45, ha="right")
ax.set_yticks(range(len(pivot3.index)))
ax.set_yticklabels(q_short, fontsize=7)
for i in range(pivot3.shape[0]):
    for j in range(pivot3.shape[1]):
        v = pivot3.values[i, j]
        ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6,
                color="white" if v < 50 else "black")
ax.set_title("Phase 3: Scores pro Frage und Modell")
fig.colorbar(im, ax=ax, shrink=0.5, label="Score")
plt.savefig(f"{OUT}/phase3_question_heatmap.png")
plt.close()

# 3c) Metrik-Vergleich über alle Modelle
fig, ax = plt.subplots(figsize=(10, 5))
metrics_cols = ["hallucination_free", "required_keywords_present", "sources_match", "link_validity", "multi_intent_complete"]
metrics_labels = ["Halluzinationsfrei", "Pflicht-Keywords", "Quellen korrekt", "Links gültig", "Multi-Intent"]
models_sorted = df3s.groupby("model")["overall_score"].mean().sort_values(ascending=False).index
x = np.arange(len(metrics_labels))
width = 0.12
for i, model in enumerate(models_sorted):
    m = df3s[df3s["model"] == model]
    vals = [m[c].mean() * 100 for c in metrics_cols]
    ax.bar(x + i * width, vals, width, label=model, color=COLORS.get(model, "#888"))
ax.set_ylabel("Rate (%)")
ax.set_title("Phase 3: Metrik-Vergleich pro Modell")
ax.set_xticks(x + width * 2.5)
ax.set_xticklabels(metrics_labels, rotation=15, ha="right")
ax.legend(fontsize=7, ncol=3)
ax.set_ylim(70, 105)
plt.savefig(f"{OUT}/phase3_metrics_comparison.png")
plt.close()

# 3d) 7B vs 14B Vergleich
fig, ax = plt.subplots(figsize=(8, 5))
small_models = ["gemma2", "qwen2.5", "mistral", "llama3.1"]
large_models = ["phi4", "qwen2.5:14b"]
for model in models_sorted:
    m = df3s[df3s["model"] == model]
    score = m["overall_score"].mean()
    cat = "14B" if model in large_models else "7B"
    ax.barh(f"{model} ({cat})", score, color=COLORS.get(model, "#888"))
    ax.text(score + 0.3, f"{model} ({cat})", f"{score:.1f}", va="center", fontsize=9)
ax.set_xlabel("Mean Score")
ax.set_title("Phase 3: 7B vs 14B Modellklassen")
ax.set_xlim(75, 103)
plt.savefig(f"{OUT}/phase3_7b_vs_14b.png")
plt.close()

print("  Phase 3: 4 Plots gespeichert")

# Phase 4: Lokal vs Cloud
print("\n=== PHASE 4 PLOTS ===")

# 4a) Score-Vergleich
fig, ax = plt.subplots(figsize=(10, 5))
p4_model = df4s.groupby("model")["overall_score"].mean().sort_values(ascending=True)
colors_4 = [COLORS.get(m, "#888") for m in p4_model.index]
bars = ax.barh(p4_model.index, p4_model.values, color=colors_4)
ax.set_xlabel("Mean Score")
ax.set_title("Phase 4: Lokal vs Cloud — Score-Vergleich")
ax.set_xlim(85, 102)
for bar, val in zip(bars, p4_model.values):
    ax.text(val + 0.2, bar.get_y() + bar.get_height()/2, f"{val:.1f}", va="center", fontsize=9)
plt.savefig(f"{OUT}/phase4_score_comparison.png")
plt.close()

# 4b) Geschwindigkeitsvergleich
fig, ax = plt.subplots(figsize=(10, 5))
p4_speed = df4s.groupby("model")["generation_time_sec"].mean().sort_values()
colors_sp = [COLORS.get(m, "#888") for m in p4_speed.index]
bars = ax.barh(p4_speed.index, p4_speed.values, color=colors_sp)
ax.set_xlabel("Durchschnittliche Generierungszeit (s)")
ax.set_title("Phase 4: Lokal vs Cloud — Geschwindigkeitsvergleich")
for bar, val in zip(bars, p4_speed.values):
    ax.text(val + 0.1, bar.get_y() + bar.get_height()/2, f"{val:.1f}s", va="center", fontsize=9)
plt.savefig(f"{OUT}/phase4_speed_comparison.png")
plt.close()

# 4c) Vergleich pro Frage
fig, ax = plt.subplots(figsize=(14, 8))
pivot4 = df4s.pivot_table(values="overall_score", index="question", columns="model", aggfunc="mean")
q_short4 = [q[:55] + "..." if len(q) > 55 else q for q in pivot4.index]
im = ax.imshow(pivot4.values, cmap="RdYlGn", vmin=40, vmax=100, aspect="auto")
ax.set_xticks(range(len(pivot4.columns)))
ax.set_xticklabels(pivot4.columns, rotation=45, ha="right")
ax.set_yticks(range(len(pivot4.index)))
ax.set_yticklabels(q_short4, fontsize=7)
for i in range(pivot4.shape[0]):
    for j in range(pivot4.shape[1]):
        v = pivot4.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if v < 50 else "black")
ax.set_title("Phase 4: Scores pro Frage — Lokal vs Cloud")
fig.colorbar(im, ax=ax, shrink=0.5, label="Score")
plt.savefig(f"{OUT}/phase4_question_heatmap.png")
plt.close()

# 4d) Fehlerrate-Vergleich
fig, ax = plt.subplots(figsize=(8, 4))
err_rates = df4.groupby("model")["is_error"].mean() * 100
bars = ax.bar(err_rates.index, err_rates.values, color=[COLORS.get(m, "#888") for m in err_rates.index])
ax.set_ylabel("Fehlerrate (%)")
ax.set_title("Phase 4: Fehlerrate pro Modell")
for bar, val in zip(bars, err_rates.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f"{val:.1f}%", ha="center", fontsize=9)
ax.tick_params(axis='x', rotation=20)
plt.savefig(f"{OUT}/phase4_error_rates.png")
plt.close()

print("  Phase 4: 4 Plots gespeichert")

# Phase 5: Fine-Grained Top-K
print("\n=== PHASE 5 PLOTS ===")

# Phase 5 (K=6) mit Phase 2 Qwen2.5:14b (K=5, K=7) kombinieren
qwen14_p2 = df2s[(df2s["model"] == "qwen2.5:14b") & (df2s["top_k"].isin([5, 7]))].copy()
qwen14_p5 = df5s.copy()
qwen14_combined = pd.concat([qwen14_p2, qwen14_p5], ignore_index=True)

# 5a) Top-K=5 vs 6 vs 7 Durchschnitts-Score
fig, ax = plt.subplots(figsize=(7, 5))
k_scores = qwen14_combined.groupby("top_k")["overall_score"].mean()
bars = ax.bar(k_scores.index.astype(str), k_scores.values, color=["#8B5CF6", "#A78BFA", "#C4B5FD"])
ax.set_xlabel("Top-K (Retrieval)")
ax.set_ylabel("Mean Score")
ax.set_title("Phase 5: Qwen 2.5:14b — Fine-Grained Top-K Vergleich")
ax.set_ylim(92, 100)
for bar, val in zip(bars, k_scores.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f"{val:.2f}", ha="center", fontsize=10)
plt.savefig(f"{OUT}/phase5_topk_comparison.png")
plt.close()

# 5b) Score pro Frage: K=5 vs K=6 vs K=7
fig, ax = plt.subplots(figsize=(14, 8))
q_by_k = qwen14_combined.pivot_table(values="overall_score", index="question", columns="top_k", aggfunc="mean")
q_short5 = [q[:55] + "..." if len(q) > 55 else q for q in q_by_k.index]
x = np.arange(len(q_by_k))
width = 0.25
for i, k in enumerate(sorted(q_by_k.columns)):
    vals = q_by_k[k].values
    ax.barh(x + i * width, vals, width, label=f"K={k}", alpha=0.85)
ax.set_yticks(x + width)
ax.set_yticklabels(q_short5, fontsize=7)
ax.set_xlabel("Mean Score")
ax.set_title("Phase 5: Qwen 2.5:14b — Score pro Frage bei K=5/6/7")
ax.legend()
ax.set_xlim(50, 105)
plt.savefig(f"{OUT}/phase5_question_by_topk.png")
plt.close()

# 5c) Beste Konfiguration pro K
fig, ax = plt.subplots(figsize=(7, 5))
best_configs = []
for k in sorted(qwen14_combined["top_k"].unique()):
    subset = qwen14_combined[qwen14_combined["top_k"] == k]
    best = subset.groupby(["temperature", "repeat_penalty"])["overall_score"].mean()
    cfg = best.idxmax()
    best_configs.append({"top_k": k, "best_score": best.max(), "avg_score": subset["overall_score"].mean(),
                          "temp": cfg[0], "rp": cfg[1]})
bdf = pd.DataFrame(best_configs)
x = np.arange(len(bdf))
ax.bar(x - 0.15, bdf["best_score"], 0.3, label="Beste Config", color="#8B5CF6")
ax.bar(x + 0.15, bdf["avg_score"], 0.3, label="Durchschnitt (alle Configs)", color="#C4B5FD")
ax.set_xticks(x)
ax.set_xticklabels([f"K={int(k)}" for k in bdf["top_k"]])
ax.set_ylabel("Score")
ax.set_title("Phase 5: Beste Config vs Durchschnitt bei K=5/6/7")
ax.legend()
ax.set_ylim(90, 101)
for i, row in bdf.iterrows():
    ax.text(i - 0.15, row["best_score"] + 0.15, f"{row['best_score']:.1f}\n(t={row['temp']},rp={row['rp']})",
            ha="center", fontsize=7)
    ax.text(i + 0.15, row["avg_score"] + 0.15, f"{row['avg_score']:.1f}", ha="center", fontsize=8)
plt.savefig(f"{OUT}/phase5_best_vs_avg.png")
plt.close()

# 5d) Metriken pro K
fig, ax = plt.subplots(figsize=(8, 5))
metric_cols_5 = ["hallucination_free", "required_keywords_present", "sources_match"]
metric_labels_5 = ["Halluzinationsfrei", "Pflicht-Keywords", "Quellen korrekt"]
x = np.arange(len(metric_labels_5))
width = 0.2
for i, k in enumerate(sorted(qwen14_combined["top_k"].unique())):
    subset = qwen14_combined[qwen14_combined["top_k"] == k]
    vals = [subset[c].mean() * 100 for c in metric_cols_5]
    ax.bar(x + i * width, vals, width, label=f"K={k}")
ax.set_ylabel("Rate (%)")
ax.set_title("Phase 5: Metriken bei K=5/6/7 (Qwen 2.5:14b)")
ax.set_xticks(x + width)
ax.set_xticklabels(metric_labels_5)
ax.legend()
ax.set_ylim(85, 102)
plt.savefig(f"{OUT}/phase5_metrics_by_topk.png")
plt.close()

print("  Phase 5: 4 Plots gespeichert")

# Phasenübergreifend: Gesamtvergleich
print("\n=== CROSS-PHASE PLOTS ===")

# C1) Entwicklung: bester Score pro Phase
fig, ax = plt.subplots(figsize=(10, 5))
phase_best = {
    "Phase 1\n(7B Grid)": df1s.groupby("model")["overall_score"].mean().max(),
    "Phase 2\n(14B Grid)": df2s.groupby("model")["overall_score"].mean().max(),
    "Phase 3\n(Head-to-Head)": df3s.groupby("model")["overall_score"].mean().max(),
    "Phase 4\n(vs Cloud)": df4s.groupby("model")["overall_score"].mean().max(),
    "Phase 5\n(Top-K Fine)": qwen14_combined.groupby("top_k").apply(
        lambda g: g.groupby(["temperature", "repeat_penalty"])["overall_score"].mean().max()
    ).max(),
}
ax.plot(list(phase_best.keys()), list(phase_best.values()), "o-", color="#8B5CF6", linewidth=2, markersize=8)
ax.set_ylabel("Bester Mean Score")
ax.set_title("Evaluierungs-Evolution: Bester Score über alle Phasen")
ax.set_ylim(85, 101)
for i, (k, v) in enumerate(phase_best.items()):
    ax.annotate(f"{v:.1f}", (i, v), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=10)
plt.savefig(f"{OUT}/cross_phase_evolution.png")
plt.close()

# C2) Alle Modelle im Phasenvergleich
fig, ax = plt.subplots(figsize=(12, 6))
all_models_scores = {}
# Phase 1+2 best config
for model in df1s["model"].unique():
    best = df1s[df1s["model"] == model].groupby(["temperature", "repeat_penalty", "top_k"])["overall_score"].mean().max()
    all_models_scores[model] = {"Grid Search\n(Phase 1/2)": best}
for model in df2s["model"].unique():
    best = df2s[df2s["model"] == model].groupby(["temperature", "repeat_penalty", "top_k"])["overall_score"].mean().max()
    all_models_scores[model] = {"Grid Search\n(Phase 1/2)": best}
# Phase 3
for model in df3s["model"].unique():
    if model not in all_models_scores:
        all_models_scores[model] = {}
    all_models_scores[model]["Head-to-Head\n(Phase 3)"] = df3s[df3s["model"] == model]["overall_score"].mean()
# Phase 4 (local only)
for model in df4s[df4s["provider"] == "local"]["model"].unique():
    if model not in all_models_scores:
        all_models_scores[model] = {}
    all_models_scores[model]["vs Cloud\n(Phase 4)"] = df4s[(df4s["model"] == model) & (df4s["provider"] == "local")]["overall_score"].mean()

phases = ["Grid Search\n(Phase 1/2)", "Head-to-Head\n(Phase 3)", "vs Cloud\n(Phase 4)"]
x = np.arange(len(phases))
width = 0.12
for i, (model, scores) in enumerate(sorted(all_models_scores.items())):
    vals = [scores.get(p, np.nan) for p in phases]
    ax.bar(x + i * width, vals, width, label=model, color=COLORS.get(model, "#888"))
ax.set_ylabel("Mean Score")
ax.set_title("Modell-Entwicklung über die Phasen")
ax.set_xticks(x + width * 2.5)
ax.set_xticklabels(phases)
ax.legend(fontsize=7, ncol=3)
ax.set_ylim(80, 103)
plt.savefig(f"{OUT}/cross_phase_model_progression.png")
plt.close()

# C3) Top-K Gesamtanalyse: K=3..9 (Phase 1+2) + K=6 (Phase 5)
fig, ax = plt.subplots(figsize=(9, 5))
all_grid = pd.concat([df1s, df2s], ignore_index=True)
k_overall = all_grid.groupby("top_k")["overall_score"].mean()
# K=6 aus Phase 5 hinzufügen
k6_score = df5s["overall_score"].mean()
k_all = pd.concat([k_overall, pd.Series({6: k6_score})]).sort_index()
colors_k = ["#94a3b8", "#8B5CF6", "#A78BFA", "#94a3b8", "#94a3b8"]
bar_colors = []
for k in k_all.index:
    if k == 5:
        bar_colors.append("#22c55e")
    elif k == 6:
        bar_colors.append("#8B5CF6")
    elif k == 7:
        bar_colors.append("#3b82f6")
    else:
        bar_colors.append("#94a3b8")
bars = ax.bar(k_all.index.astype(str), k_all.values, color=bar_colors)
ax.set_xlabel("Top-K (Retrieval)")
ax.set_ylabel("Mean Score (alle Modelle)")
ax.set_title("Gesamt-Top-K-Analyse: K=3 bis K=9 (inkl. K=6 aus Phase 5)")
ax.set_ylim(82, 98)
for bar, (k, val) in zip(bars, k_all.items()):
    label = f"{val:.1f}"
    if k == 5:
        label += "\n★ Best"
    elif k == 6:
        label += "\n(Phase 5)"
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, label, ha="center", fontsize=8)
plt.savefig(f"{OUT}/cross_phase_topk_overall.png")
plt.close()

# C4) Gesamtübersicht Evaluierung
fig, ax = plt.subplots(figsize=(10, 5))
phase_data = [
    ("Phase 1\n7B Grid", len(df1), len(df1s), df1s["overall_score"].mean()),
    ("Phase 2\n14B Grid", len(df2), len(df2s), df2s["overall_score"].mean()),
    ("Phase 3\nHead-to-Head", len(df3), len(df3s), df3s["overall_score"].mean()),
    ("Phase 4\nvs Cloud", len(df4), len(df4s), df4s["overall_score"].mean()),
]
labels = [p[0] for p in phase_data]
totals = [p[1] for p in phase_data]
scores = [p[3] for p in phase_data]
x = np.arange(len(labels))
ax2 = ax.twinx()
bars = ax.bar(x, totals, 0.4, color="#8B5CF6", alpha=0.6, label="Anzahl Evals")
ax2.plot(x, scores, "o-", color="#ef4444", linewidth=2, markersize=8, label="Mean Score")
ax.set_ylabel("Anzahl Evaluierungen")
ax2.set_ylabel("Durchschnittlicher Score")
ax.set_title(f"Gesamtübersicht: {sum(totals)} Evaluierungen in vier Phasen")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc="upper left")
ax2.legend(loc="upper right")
ax2.set_ylim(80, 100)
for bar, total in zip(bars, totals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, str(total), ha="center", fontsize=9)
plt.savefig(f"{OUT}/cross_phase_overview.png")
plt.close()

print("  Phasenübergreifend: 4 Plots gespeichert")

# Zusammenfassung 
print("\n" + "=" * 70)
print("ZUSAMMENFASSUNG: Phase 5 Top-K Fine-Grained Analyse")
print("=" * 70)

print("\n--- K=5 vs K=6 vs K=7 (Qwen 2.5:14b, alle Configs) ---")
for k in [5, 6, 7]:
    subset = qwen14_combined[qwen14_combined["top_k"] == k]
    best = subset.groupby(["temperature", "repeat_penalty"])["overall_score"].mean()
    cfg = best.idxmax()
    print(f"  K={k}: avg={subset['overall_score'].mean():.2f}, best={best.max():.1f} "
          f"(temp={cfg[0]}, rp={cfg[1]}), hall_free={subset['hallucination_free'].mean()*100:.1f}%, "
          f"req_kw={subset['required_keywords_present'].mean()*100:.1f}%")

print("\n--- Per-question K=5 vs K=6 vs K=7 ---")
q5 = qwen14_combined[qwen14_combined["top_k"] == 5].groupby("question")["overall_score"].mean()
q6 = qwen14_combined[qwen14_combined["top_k"] == 6].groupby("question")["overall_score"].mean()
q7 = qwen14_combined[qwen14_combined["top_k"] == 7].groupby("question")["overall_score"].mean()

# Auf gemeinsame Fragen abgleichen
common_qs = q5.index.intersection(q6.index).intersection(q7.index)
print(f"  Verglichene Fragen: {len(common_qs)}")
print(f"\n  {'Frage':<60} {'K=5':>5} {'K=6':>5} {'K=7':>5} {'Best':>5}")
print(f"  {'-'*60} {'---':>5} {'---':>5} {'---':>5} {'---':>5}")
for q in sorted(common_qs):
    s5, s6, s7 = q5[q], q6[q], q7[q]
    best_k = [5, 6, 7][np.argmax([s5, s6, s7])]
    marker = f"K={best_k}" if max(s5, s6, s7) - min(s5, s6, s7) > 2 else "≈"
    print(f"  {q[:60]:<60} {s5:5.1f} {s6:5.1f} {s7:5.1f} {marker:>5}")
print(f"\n  {'AVERAGE':<60} {q5[common_qs].mean():5.1f} {q6[common_qs].mean():5.1f} {q7[common_qs].mean():5.1f}")

print("\n" + "=" * 70)
print("GESAMTÜBERSICHT EVALUIERUNG")
print("=" * 70)
total = len(df1) + len(df2) + len(df3) + len(df4) + len(df5)
total_errors = df1["is_error"].sum() + df2["is_error"].sum() + df3["is_error"].sum() + df4["is_error"].sum() + df5["is_error"].sum()
print(f"  Evaluierungen gesamt: {total}")
print(f"  Fehler gesamt: {int(total_errors)} ({total_errors/total*100:.2f}%)")
print(f"\n  Phase 1 (7B Grid):      {len(df1):>5} evals, avg={df1s['overall_score'].mean():.1f}")
print(f"  Phase 2 (14B Grid):     {len(df2):>5} evals, avg={df2s['overall_score'].mean():.1f}")
print(f"  Phase 3 (Head-to-Head): {len(df3):>5} evals, avg={df3s['overall_score'].mean():.1f}")
print(f"  Phase 4 (vs Cloud):     {len(df4):>5} evals, avg={df4s['overall_score'].mean():.1f}")
print(f"  Phase 5 (Top-K Fine):   {len(df5):>5} evals, avg={df5s['overall_score'].mean():.1f}")
print(f"\n  GESAMT: {total} Evaluierungen über 5 Phasen")

print(f"\nAlle {len(os.listdir(OUT))} Plots gespeichert unter {OUT}/")
