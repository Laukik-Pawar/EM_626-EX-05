"""
EX05 - Comparing text: Pro vs Con (Britannica ProCon)
Usage: python ex05_analysis.py [pro.txt] [con.txt] [stopwords_en.txt]
Outputs (in ./output): cleaned texts, metrics.json, bigram CSVs, word clouds, networks, plots.
"""
import sys, re, json, os
from collections import Counter
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

PRO = sys.argv[1] if len(sys.argv) > 1 else "pros.txt"
CON = sys.argv[2] if len(sys.argv) > 2 else "cons.txt"
STOP = sys.argv[3] if len(sys.argv) > 3 else "stopwords_en.txt"
OUT = "output"
os.makedirs(OUT, exist_ok=True)

# Words that appear everywhere because of the issue itself (edit after inspecting top words)
ISSUE_WORDS = {"artificial", "intelligence", "schoolwork", "students", "student", "school",
               "schools", "chatgpt", "tools", "tool", "pro", "con"}

stopwords = {w.strip().lower() for w in open(STOP, encoding="utf-8") if w.strip()}
stopwords |= ISSUE_WORDS


def clean(text):
    """Return (list of cleaned tokens per sentence, flat token list)."""
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"\r|\n+", " ", text)                          # end-of-line / blank lines
    text = text.lower()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sent_tokens = []
    for s in sentences:
        s = re.sub(r"[^a-z\s]", " ", s)                          # punctuation, numbers, symbols
        toks = [w for w in s.split() if len(w) >= 3 and w not in stopwords]
        if toks:
            sent_tokens.append(toks)
    return sent_tokens, [w for s in sent_tokens for w in s]


def lexical_diversity(tokens):
    return len(set(tokens)) / len(tokens) if tokens else 0.0


def bigrams(sent_tokens):
    c = Counter()
    for toks in sent_tokens:
        c.update(zip(toks, toks[1:]))
    return c


def build_network(bg, min_count=2):
    """Co-occurrence (adjacent-word) network: nodes = words, edge weight = bigram count."""
    G = nx.Graph()
    for (a, b), n in bg.items():
        if n >= min_count and a != b:
            G.add_edge(a, b, weight=n)
    return G


def network_metrics(G):
    giant = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    try:
        comms = nx.community.greedy_modularity_communities(G, weight="weight")
        modularity = nx.community.modularity(G, comms, weight="weight")
        n_comm = len(comms)
    except Exception:
        modularity, n_comm = float("nan"), 0
    deg_c = nx.degree_centrality(G)
    btw = nx.betweenness_centrality(G)
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_degree": sum(d for _, d in G.degree()) / G.number_of_nodes(),
        "avg_clustering": nx.average_clustering(G),
        "connected_components": nx.number_connected_components(G),
        "giant_component_share": giant.number_of_nodes() / G.number_of_nodes(),
        "avg_path_length_giant": nx.average_shortest_path_length(giant) if giant.number_of_nodes() > 1 else 0,
        "diameter_giant": nx.diameter(giant) if giant.number_of_nodes() > 1 else 0,
        "communities": n_comm,
        "modularity": modularity,
        "top_degree": sorted(deg_c, key=deg_c.get, reverse=True)[:8],
        "top_betweenness": sorted(btw, key=btw.get, reverse=True)[:8],
    }


def draw_network(G, title, path, color):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42, k=0.6, weight="weight")
    deg = dict(G.degree())
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=[0.5 + G[u][v]["weight"] * 0.4 for u, v in G.edges()])
    nx.draw_networkx_nodes(G, pos, node_size=[80 + 60 * deg[n] for n in G], node_color=color, alpha=0.85)
    top = sorted(deg, key=deg.get, reverse=True)[:25]
    nx.draw_networkx_labels(G, pos, labels={n: n for n in top}, font_size=9)
    plt.title(title); plt.axis("off"); plt.tight_layout()
    plt.savefig(path, dpi=200); plt.close()


def main():
    analyzer = SentimentIntensityAnalyzer()
    results, data = {}, {}
    for label, path, cmap in [("pro", PRO, "Greens"), ("con", CON, "Reds")]:
        raw = open(path, encoding="utf-8").read()
        sents, toks = clean(raw)
        open(f"{OUT}/{label}_clean.txt", "w", encoding="utf-8").write(" ".join(toks))

        # VADER is run on the raw text (it relies on punctuation/capitalisation) and on the cleaned text
        vs_raw = analyzer.polarity_scores(raw)
        vs_clean = analyzer.polarity_scores(" ".join(toks))
        sent_scores = [analyzer.polarity_scores(s)["compound"]
                       for s in re.split(r"(?<=[.!?])\s+", raw) if len(s.split()) > 3]

        bg = bigrams(sents)
        pd.DataFrame([(" ".join(k), v) for k, v in bg.most_common(30)],
                     columns=["bigram", "count"]).to_csv(f"{OUT}/{label}_bigrams.csv", index=False)

        wc = WordCloud(width=1200, height=700, background_color="white", colormap=cmap,
                       max_words=100, random_state=1).generate(" ".join(toks))
        wc.to_file(f"{OUT}/{label}_wordcloud.png")

        G = build_network(bg)
        draw_network(G, f"{label.upper()} bigram network", f"{OUT}/{label}_network.png",
                     "#2e8b57" if label == "pro" else "#c0392b")

        freq = Counter(toks)
        results[label] = {
            "raw_words": len(raw.split()),
            "clean_tokens": len(toks),
            "unique_words": len(set(toks)),
            "lexical_diversity": lexical_diversity(toks),
            "vader_raw": vs_raw,
            "vader_clean": vs_clean,
            "sentence_mean_compound": sum(sent_scores) / len(sent_scores) if sent_scores else 0,
            "sentences_pos_neg_neu": [
                sum(s >= 0.05 for s in sent_scores),
                sum(s <= -0.05 for s in sent_scores),
                sum(-0.05 < s < 0.05 for s in sent_scores)],
            "top_words": freq.most_common(15),
            "top_bigrams": [(" ".join(k), v) for k, v in bg.most_common(15)],
            "network": network_metrics(G),
        }
        data[label] = freq

    # Words that are distinctive to each side
    p, c = data["pro"], data["con"]
    pt, ct = sum(p.values()), sum(c.values())
    diff = {w: (p[w] / pt - c[w] / ct) for w in set(p) | set(c) if p[w] + c[w] >= 4}
    results["distinctive_pro"] = sorted(diff, key=diff.get, reverse=True)[:12]
    results["distinctive_con"] = sorted(diff, key=diff.get)[:12]

    # Comparison chart: top words
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, label, col in zip(axes, ["pro", "con"], ["#2e8b57", "#c0392b"]):
        w, n = zip(*results[label]["top_words"][:12])
        ax.barh(w[::-1], n[::-1], color=col); ax.set_title(f"{label.upper()}: most frequent words")
    plt.tight_layout(); plt.savefig(f"{OUT}/top_words.png", dpi=200); plt.close()

    # Comparison chart: sentiment + lexical diversity
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, ttl in [(axes[0], "vader", "VADER (raw text)"), (axes[1], "ld", "Lexical diversity")]:
        if key == "vader":
            cats = ["pos", "neu", "neg"]
            x = range(3)
            for i, (label, col) in enumerate([("pro", "#2e8b57"), ("con", "#c0392b")]):
                ax.bar([j + i * 0.38 for j in x], [results[label]["vader_raw"][c] for c in cats], 0.38,
                       label=label, color=col)
            ax.set_xticks([j + 0.19 for j in x]); ax.set_xticklabels(cats); ax.legend()
        else:
            ax.bar(["pro", "con"], [results[l]["lexical_diversity"] for l in ("pro", "con")],
                   color=["#2e8b57", "#c0392b"])
        ax.set_title(ttl)
    plt.tight_layout(); plt.savefig(f"{OUT}/sentiment_diversity.png", dpi=200); plt.close()

    with open(f"{OUT}/metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()