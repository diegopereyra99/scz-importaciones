#!/usr/bin/env python3
import argparse, json, os, hashlib, re
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

# --- Vertex (GenAI SDK) ---
from google import genai

def normalize_text(md: str) -> str:
    md = md.strip().replace("\r\n", "\n")
    md = re.sub(r"[ \t]+", " ", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def jsonl_write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_email_summaries(processed_mails_dir: Path):
    items = []
    for mail_dir in sorted([p for p in processed_mails_dir.iterdir() if p.is_dir()]):
        md_path = mail_dir / "mail_summary.md"
        if not md_path.exists():
            continue
        text = normalize_text(md_path.read_text(encoding="utf-8"))
        items.append({
            "email_id": mail_dir.name,   # para POC ok; si querés, leés mail.json y ponés mail_id real
            "mail_dir": str(mail_dir),
            "summary_path": str(md_path),
            "text": text,
            "text_hash": sha1(text),
        })
    return items

def vertex_embed_texts(
    texts,
    model="gemini-embedding-001",
    output_dimensionality=None,
    batch_size=200
):
    """
    Embeddings con Vertex AI usando google-genai (API actual).
    """
    client = genai.Client(
        vertexai=True,
        project=os.environ["GOOGLE_CLOUD_PROJECT"],
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
    )

    vectors = []

    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]

        kwargs = {
            "model": model,
            "contents": chunk,
        }
        if output_dimensionality is not None:
            kwargs["output_dimensionality"] = int(output_dimensionality)

        resp = client.models.embed_content(**kwargs)

        # resp.embeddings es una lista
        for emb in resp.embeddings:
            vectors.append(emb.values)

    V = np.array(vectors, dtype=np.float32)

    # normalizar para cosine similarity
    norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    V = V / norms

    return V

def cluster_agglomerative_cosine(vecs: np.ndarray, distance_threshold: float):
    # cosine distance = 1 - cosine similarity
    try:
        cl = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=distance_threshold
        )
    except TypeError:
        cl = AgglomerativeClustering(
            n_clusters=None,
            affinity="cosine",
            linkage="average",
            distance_threshold=distance_threshold
        )
    labels = cl.fit_predict(vecs)
    return labels

def plot_cluster_scatter(vecs: np.ndarray, labels: np.ndarray, items, path: Path):
    if vecs.shape[0] < 2:
        return None
    coords = PCA(n_components=2).fit_transform(vecs)
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=labels,
        cmap="tab20",
        s=40,
        alpha=0.85,
        edgecolors="none"
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Emails clustered (PCA 2D)")
    legend = ax.legend(*scatter.legend_elements(num=None), title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.add_artist(legend)
    if len(items) <= 40:
        for idx, it in enumerate(items):
            ax.text(coords[idx, 0], coords[idx, 1], it["email_id"], fontsize=7, alpha=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

def plot_cosine_similarity_heatmap(vecs: np.ndarray, items, path: Path, max_items: int = 150):
    if vecs.size == 0:
        return None
    count = vecs.shape[0]
    idxs = np.arange(count)
    if count > max_items:
        idxs = np.linspace(0, count - 1, num=max_items, dtype=int)
    sub_vecs = vecs[idxs]
    sim = np.clip(np.dot(sub_vecs, sub_vecs.T), 0.0, 1.0)
    labels = [items[i]["email_id"] for i in idxs]
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sim, cmap="viridis", vmin=0, vmax=1)
    ax.set_title("Cosine similarity matrix")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine similarity")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-mails-dir", required=True)
    ap.add_argument("--db-dir", default="db")
    ap.add_argument("--model", default="gemini-embedding-001")
    ap.add_argument("--output-dim", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--cluster-distance", type=float, default=0.30, help="menor=>más clusters; típico 0.20-0.40")
    args = ap.parse_args()

    processed = Path(args.processed_mails_dir).resolve()
    db = Path(args.db_dir).resolve()
    db.mkdir(parents=True, exist_ok=True)
    cluster_plot_path = db / "email_clusters.png"
    sim_heatmap_path = db / "email_cosine_similarity.png"

    items = load_email_summaries(processed)
    if not items:
        raise SystemExit("No encontré mail_summary.md en processed_mails/*/")

    texts = [it["text"] for it in items]
    vecs = vertex_embed_texts(
        texts,
        model=args.model,
        output_dimensionality=args.output_dim,
        batch_size=min(args.batch_size, 250),  # doc dice 250 max :contentReference[oaicite:5]{index=5}
    )

    labels = cluster_agglomerative_cosine(vecs, args.cluster_distance)
    scatter_path = plot_cluster_scatter(vecs, labels, items, cluster_plot_path)
    heatmap_path = plot_cosine_similarity_heatmap(vecs, items, sim_heatmap_path)

    created_at = datetime.utcnow().isoformat()

    emb_rows = []
    clus_rows = []
    for it, v, lab in zip(items, vecs, labels):
        emb_rows.append({
            "entity_type": "email",
            "entity_id": it["email_id"],
            "cluster_id": int(lab),
            "model": args.model,
            "output_dimensionality": args.output_dim,
            "vector": v.tolist(),
            "text_hash": it["text_hash"],
            "text_path": it["summary_path"],
            "created_at": created_at
        })
        clus_rows.append({
            "email_id": it["email_id"],
            "cluster_id": int(lab),
            "mail_dir": it["mail_dir"],
            "summary_path": it["summary_path"],
            "created_at": created_at
        })

    emb_path = db / "email_embeddings.jsonl"
    clus_path = db / "email_clusters.jsonl"
    jsonl_write(emb_path, emb_rows)
    jsonl_write(clus_path, clus_rows)

    # Reporte simple
    clusters = {}
    for it, lab in zip(items, labels):
        clusters.setdefault(int(lab), []).append(it)

    ordered = sorted(clusters.items(), key=lambda kv: len(kv[1]), reverse=True)
    report_lines = []
    report_lines.append("# Clusters Report\n")
    report_lines.append(f"- Embedding model: `{args.model}`")
    report_lines.append(f"- Output dim: `{args.output_dim}`")
    report_lines.append(f"- Agglomerative cosine distance_threshold: `{args.cluster_distance}`")
    report_lines.append(f"- Emails: {len(items)}\n")

    if scatter_path or heatmap_path:
        report_lines.append("## Visualizaciones\n")
        if scatter_path:
            report_lines.append(f"- Gráfico clusters (PCA 2D): `{scatter_path}`")
        if heatmap_path:
            report_lines.append(f"- Matriz de similitud coseno: `{heatmap_path}`")
        report_lines.append("")

    for cid, its in ordered:
        report_lines.append(f"## Cluster {cid} ({len(its)} emails)\n")
        for it in its[:25]:
            first = it["text"].splitlines()[0] if it["text"] else ""
            report_lines.append(f"- `{it['email_id']}` — {Path(it['mail_dir']).name} — {first}")
        if len(its) > 25:
            report_lines.append(f"- ... (+{len(its)-25} más)")
        report_lines.append("")

    report_path = db / "clusters_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("✅ Listo")
    print(f"- {emb_path}")
    print(f"- {clus_path}")
    print(f"- {report_path}")
    if scatter_path:
        print(f"- {scatter_path}")
    if heatmap_path:
        print(f"- {heatmap_path}")

if __name__ == "__main__":
    main()
