#!/usr/bin/env python3
"""公共函数: 带缓存的 chunk 生成(长句词级拆分)与 chunk embedding.

缓存文件名包含 min/max 时长参数, 参数变化时自动重建, 不会误用旧缓存:
  build/epXX/chunks_{min}_{max}.json
  build/epXX/chunk_emb_{min}_{max}.npy / chunk_idx_{min}_{max}.json
"""
import json
import os

import numpy as np
import torch
from audio_utils import is_quiet, split_long


def get_chunks(ep_dir, kept, audio16k, sr, min_dur, max_dur):
    cache = os.path.join(ep_dir, f"chunks_{min_dur}_{max_dur}.json")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            return json.load(f)
    chunks = []
    for seg in kept:
        dur = seg["end"] - seg["start"]
        if dur < min_dur:
            continue
        if dur <= max_dur:
            chunks.append(seg)
        else:
            chunks.extend(split_long(seg, audio16k, sr))
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=1)
    return chunks


def get_embeddings(encoder, ep_dir, chunks, audio16k, sr, min_dur, max_dur):
    """返回 (有效 chunk 下标列表, embedding 矩阵); 静音 chunk 不算不缓存."""
    tag = f"{min_dur}_{max_dur}"
    emb_path = os.path.join(ep_dir, f"chunk_emb_{tag}.npy")
    idx_path = os.path.join(ep_dir, f"chunk_idx_{tag}.json")
    if os.path.exists(emb_path) and os.path.exists(idx_path):
        with open(idx_path, encoding="utf-8") as f:
            return json.load(f), np.load(emb_path)
    idx, embs = [], []
    for i, seg in enumerate(chunks):
        s, e = int(seg["start"] * sr), int(seg["end"] * sr)
        clip = audio16k[s:e]
        if is_quiet(clip):
            continue
        with torch.no_grad():
            emb = encoder.encode_batch(
                torch.tensor(clip, dtype=torch.float32).unsqueeze(0)).squeeze().cpu().numpy()
        idx.append(i)
        embs.append(emb / (np.linalg.norm(emb) + 1e-8))
    X = np.stack(embs) if embs else np.zeros((0, 192))
    np.save(emb_path, X)
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(idx, f)
    return idx, X
