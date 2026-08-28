#!/usr/bin/env python3
"""批量提取全部剧集的小叽语音.

流程(每集): 抽音轨 -> demucs 人声分离 -> mlx-whisper 转写(裁掉 OP/ED)
-> ECAPA embedding -> 与小叽参考向量的 cosine 相似度 -> 阈值导出.

输出:
  dataset/wavs/000001.wav ...   22050Hz 单声道 16-bit PCM
  dataset/metadata.csv          文件名|文本
  dataset/metadata_full.csv     含集数/时间/相似度的完整表
  dataset/review/               相似度落在 [REVIEW_LO, THRESHOLD) 的待人工复核片段

用法: .venv/bin/python pipeline/batch.py
"""
import csv
import json
import os
import re
import subprocess
import sys

import mlx_whisper
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from speechbrain.inference.speaker import EncoderClassifier

MOVIE_DIR = "Chobits_Movie"
BUILD_DIR = "build"
OUT_DIR = "dataset"
REF_PATH = "annotations/legacy/chi_reference.npy"
THRESHOLD = 0.60
REVIEW_LO = 0.50
MIN_DUR, MAX_DUR = 1.0, 15.0
CHI_PATTERN = re.compile(r"^[ちチ][ぃいー]{1,2}[!！?？]?$")
N_REF_SAMPLES = 8  # chi_reference.npy 由 8 段平均而成

FFMPEG = subprocess.run(
    [".venv/bin/python", "-c", "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"],
    capture_output=True, text=True, check=True).stdout.strip()


def find_episodes():
    eps = []
    for name in os.listdir(MOVIE_DIR):
        m = re.match(r"第([\d.]+)话 .+\.mp4$", name)
        if m:
            label = m.group(1)
            eps.append((label, os.path.join(MOVIE_DIR, name)))
    return sorted(eps, key=lambda x: float(x[0]))


def ep_dir_name(label):
    """集数标签 -> 目录名: '1' -> ep01, '8.5' -> ep08_5"""
    if "." in label:
        head, tail = label.split(".")
        return f"ep{int(head):02d}_{tail}"
    return f"ep{int(label):02d}"


def op_ed_ranges(ep_no):
    """从该集元数据 JSON 中读 B 站官方 OP/ED 跳过区间(秒)."""
    metas = [f for f in os.listdir(MOVIE_DIR)
             if f.startswith(f"第{ep_no}话 ") and f.endswith("-元数据.json")]
    if not metas:
        return []
    with open(os.path.join(MOVIE_DIR, metas[0]), encoding="utf-8") as f:
        data = json.load(f)
    for ep in data.get("episodes", []):
        if ep.get("title") == str(ep_no):
            skip = ep.get("skip") or {}
            return [(skip[k]["start"], skip[k]["end"]) for k in ("op", "ed") if k in skip]
    return []


def prepare_episode(ep_no, mp4_path, ep_dir):
    """抽音轨/分离/转16k, 幂等, 返回 vocals_16k 路径."""
    os.makedirs(ep_dir, exist_ok=True)
    audio = os.path.join(ep_dir, "audio.wav")
    vocals = os.path.join(ep_dir, "separated", "htdemucs", "audio", "vocals.wav")
    vocals16k = os.path.join(ep_dir, "vocals_16k.wav")
    if not os.path.exists(audio):
        subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                        "-i", mp4_path, "-vn", "-ac", "2", "-ar", "44100", audio], check=True)
    if not os.path.exists(vocals):
        subprocess.run([".venv/bin/python", "-m", "demucs", "--two-stems=vocals",
                        "-n", "htdemucs", "--device", "mps",
                        "--out", os.path.join(ep_dir, "separated"), audio], check=True)
    if not os.path.exists(vocals16k):
        subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                        "-i", vocals, "-ac", "1", "-ar", "16000", vocals16k], check=True)
    return vocals, vocals16k


def transcribe(vocals16k, seg_path, skip_ranges):
    if os.path.exists(seg_path):
        with open(seg_path, encoding="utf-8") as f:
            return json.load(f)
    audio, sr = sf.read(vocals16k)
    result = mlx_whisper.transcribe(
        audio.astype("float32"),
        path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
        language="ja",
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
    )
    segments = []
    for s in result["segments"]:
        text = s["text"].strip()
        if not text or s.get("no_speech_prob", 0) >= 0.8:
            continue
        mid = (s["start"] + s["end"]) / 2
        if any(a <= mid <= b for a, b in skip_ranges):
            continue
        segments.append({"start": round(s["start"], 2), "end": round(s["end"], 2), "text": text})
    with open(seg_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=1)
    return segments


def main():
    os.makedirs(os.path.join(OUT_DIR, "wavs"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "review"), exist_ok=True)

    ref_global = np.load(REF_PATH)
    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cpu"})

    rows, full_rows = [], []
    clip_no = 0
    for ep_no, mp4_path in find_episodes():
        ep_dir = os.path.join(BUILD_DIR, ep_dir_name(ep_no))
        print(f"===== 第{ep_no}话 =====", flush=True)
        vocals, vocals16k = prepare_episode(ep_no, mp4_path, ep_dir)
        segments = transcribe(vocals16k, os.path.join(ep_dir, "segments.json"), op_ed_ranges(ep_no))

        audio16k, sr16k = sf.read(vocals16k)
        kept, embs = [], []
        for seg in segments:
            s, e = int(seg["start"] * sr16k), int(seg["end"] * sr16k)
            if (e - s) / sr16k < 0.5:
                continue
            wav = torch.tensor(audio16k[s:e], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                emb = encoder.encode_batch(wav).squeeze().cpu().numpy()
            embs.append(emb / (np.linalg.norm(emb) + 1e-8))
            kept.append(seg)
        if not embs:
            continue
        X = np.stack(embs)

        # 参考向量 = 全局确认样本 + 本集纯"ちい"独白段 的加权平均
        local = [i for i, s in enumerate(kept) if CHI_PATTERN.match(s["text"])]
        ref = ref_global * N_REF_SAMPLES
        if local:
            ref = ref + X[local].sum(axis=0)
        ref /= (N_REF_SAMPLES + len(local))
        ref /= np.linalg.norm(ref)

        sims = X @ ref
        # 从 44.1k 人声轨导出高质量切片
        vocals_audio, sr_v = sf.read(vocals, dtype="float32")
        n_chi = 0
        for seg, sim in zip(kept, sims):
            dur = seg["end"] - seg["start"]
            if not (MIN_DUR <= dur <= MAX_DUR) or sim < REVIEW_LO:
                continue
            s, e = int(seg["start"] * sr_v), int(seg["end"] * sr_v)
            clip = vocals_audio[s:e].mean(axis=1)  # 立体声混单声道
            clip = resample_poly(clip, 1, 2)         # 44100 -> 22050
            full_rows.append({"ep": ep_no, "start": seg["start"], "end": seg["end"],
                              "sim": round(float(sim), 3), "text": seg["text"]})
            if sim < THRESHOLD:
                sf.write(os.path.join(OUT_DIR, "review",
                                      f"{ep_dir_name(ep_no)}_sim{sim:.3f}_{seg['start']:.0f}s.wav"),
                         clip, 22050, subtype="PCM_16")
                continue
            clip_no += 1
            name = f"{clip_no:06d}"
            sf.write(os.path.join(OUT_DIR, "wavs", f"{name}.wav"), clip, 22050, subtype="PCM_16")
            rows.append((name, seg["text"]))
            n_chi += 1
        print(f"  segments={len(kept)} local_chi={len(local)} exported={n_chi}", flush=True)

    with open(os.path.join(OUT_DIR, "metadata.csv"), "w", encoding="utf-8") as f:
        for name, text in rows:
            f.write(f"{name}|{text}\n")
    with open(os.path.join(OUT_DIR, "metadata_full.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ep", "start", "end", "sim", "text"])
        w.writeheader()
        w.writerows(full_rows)
    print(f"完成: {clip_no} 个小叽片段 -> {OUT_DIR}/wavs, 待复核见 {OUT_DIR}/review")


if __name__ == "__main__":
    main()
