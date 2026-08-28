#!/usr/bin/env python3
"""生成全集台词索引 dataset/transcripts.csv.

每行一句 whisper 转写台词 (OP/ED 已剔除), 附带:
  chi_prob   句内所有 chunk 的小叽分类概率最大值 (可粗略判断是谁说的)
  in_dataset 该句是否有片段入选最终数据集 (dataset/wavs/)

用途: 全文检索"是否说过某句话", 并定位到集数和时间.
  例: grep 'ちい' dataset/transcripts.csv

用法: .venv/bin/python pipeline/build_transcript_index.py
"""
import csv
import json
import os
import pickle

import numpy as np
import soundfile as sf
from speechbrain.inference.speaker import EncoderClassifier

import batch
from common import get_chunks, get_embeddings

MIN_DUR, MAX_DUR = 1.0, 10.0


def main():
    with open("annotations/chi_lr.pkl", "rb") as f:
        clf = pickle.load(f)["model"]
    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cpu"})

    # 已收录片段区间, 用于 in_dataset 标记
    exported = {}
    with open(os.path.join(batch.OUT_DIR, "metadata_full.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            exported.setdefault(r["ep"], []).append((float(r["start"]), float(r["end"])))

    rows = []
    for label, _ in batch.find_episodes():
        if "." in label:
            continue
        ep_dir = os.path.join(batch.BUILD_DIR, batch.ep_dir_name(label))
        kp = os.path.join(ep_dir, "kept.json")
        if not os.path.exists(kp):
            continue
        with open(kp, encoding="utf-8") as f:
            kept = json.load(f)
        audio16k, sr16 = sf.read(os.path.join(ep_dir, "vocals_16k.wav"))
        chunks = get_chunks(ep_dir, kept, audio16k, sr16, MIN_DUR, MAX_DUR)
        idx, X = get_embeddings(encoder, ep_dir, chunks, audio16k, sr16, MIN_DUR, MAX_DUR)
        probs = clf.predict_proba(X)[:, 1] if len(X) else np.array([])

        for seg in kept:
            s, e = seg["start"], seg["end"]
            p = max((pr for i, pr in zip(idx, probs)
                     if chunks[i]["start"] < e and chunks[i]["end"] > s), default=None)
            in_ds = any(s < xe and e > xs
                        for xs, xe in exported.get(batch.ep_dir_name(label), []))
            rows.append({"ep": batch.ep_dir_name(label), "start": round(s, 2), "end": round(e, 2),
                         "dur": round(e - s, 2), "text": seg["text"],
                         "chi_prob": round(float(p), 3) if p is not None else "",
                         "in_dataset": int(in_ds)})
        print(f"  第{label}话 {len(kept)} 句", flush=True)

    out = os.path.join(batch.OUT_DIR, "transcripts.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ep", "start", "end", "dur", "text",
                                          "chi_prob", "in_dataset"])
        w.writeheader()
        w.writerows(rows)
    n_in = sum(r["in_dataset"] for r in rows)
    print(f"完成: {len(rows)} 句 -> {out}; 其中 {n_in} 句有片段入选数据集")


if __name__ == "__main__":
    main()
