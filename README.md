---
license: cc-by-nc-sa-4.0
language:
- ja
- zh
tags:
- audio
- speech
- tts
- voice-cloning
- anime
task_categories:
- text-to-speech
pretty_name: Chobits-Chi
---

# Chobits-Chi

《人形电脑天使心》(Chobits) 中 **小叽 (Chi / ちぃ)** 角色的语音数据集,可用于语音合成 (TTS)、声音克隆等任务的训练与微调。

## 数据集简介

本数据集从《Chobits》动画音轨中提取角色**小叽**(CV: 田中理惠)的语音片段,经过切分、降噪与筛选后整理而成,并配有对应的文本标注。

> ⚠️ 注意:原始动画音频的版权归其权利方所有。本数据集仅供学习与研究使用,请勿用于商业用途。

## 数据集结构

<!-- 请根据实际目录结构修改 -->

```
Chobits-Chi/
├── wavs/               # 音频片段
│   ├── 000001.wav
│   ├── 000002.wav
│   └── ...
└── metadata.csv        # 标注文件: 文件名|文本
```

- **音频格式**:WAV,22050 Hz,单声道(16-bit PCM)<!-- 按实际情况修改 -->
- **标注格式**:`metadata.csv`,每行为 `文件名|文本`

## 数据统计

<!-- 请填写实际统计信息 -->

| 项目 | 数值 |
| --- | --- |
| 片段数量 | - |
| 总时长 | - |
| 采样率 | 22050 Hz |
| 语言 | 日语 |

## 使用方法

### 加载数据集

```python
from datasets import load_dataset

ds = load_dataset("chenxin199305/Chobits-Chi")
```

### 配合 TTS 训练

本数据集的目录结构兼容常见 TTS 框架(如 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)、[VITS](https://github.com/jaywalnut310/vits) 等),可直接作为训练输入或经简单转换后使用。

## 数据处理流程

1. **音轨提取**:从动画视频中提取音频流
2. **人声分离**:分离人声与背景音/BGM
3. **语音切分**:按句子切分为独立音频片段
4. **角色筛选**:保留小叽的语音(可借助说话人识别)
5. **文本标注**:生成/校对每段语音对应的文本
6. **质量过滤**:剔除过短、过长或含噪声的片段

## 许可协议

本数据集以 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh-hans)(署名-非商业性使用-相同方式共享)协议发布:

- **署名 (BY)**:使用时须注明数据集来源。
- **非商业性使用 (NC)**:不得将本数据集用于商业目的。
- **相同方式共享 (SA)**:基于本数据集的衍生作品须以相同协议发布。

## 免责声明

- 本数据集仅用于学术研究与个人学习,不构成对原作品版权的任何主张。
- 使用本数据集训练的模型,其生成内容不得用于侵犯原作品及相关声优权益的用途。
- 若权利方提出要求,本数据集将被下架。

## 引用

如果您在研究中使用了本数据集,请引用:

```bibtex
@misc{chobits-chi,
  author = {chenxin199305},
  title = {Chobits-Chi: Voice Dataset of Chi from Chobits},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/chenxin199305/Chobits-Chi}}
}
```
