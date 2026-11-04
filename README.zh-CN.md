<div align="center">
<h2>SparseDriveV2：端到端自动驾驶中，评分就是关键</h2>

[Wenchao Sun](https://scholar.google.com/citations?user=yd-sMoQAAAAJ&hl=en&oi=sra)<sup>1,2</sup>、[Xuewu Lin](https://scholar.google.com/citations?user=pfXQwcQAAAAJ&hl=en&oi=sra)<sup>3</sup>、[Keyu Chen](https://scholar.google.com/citations?hl=en&user=m_bC1VAAAAAJ)<sup>1</sup>、[Zixiang Pei](https://scholar.google.com/citations?user=Lq4s0IsAAAAJ&hl=en&oi=ao)<sup>2</sup>、\
[Xiang Li](https://openreview.net/profile?id=~Xiang_LI187)<sup>2</sup>、[Yining Shi](https://scholar.google.com/citations?user=2zOwf-QAAAAJ&hl=en&oi=sra)<sup>1</sup>、[Sifa Zheng](https://www.svm.tsinghua.edu.cn/essay/80/1835.html)<sup>1</sup>

<sup>1</sup> 清华大学　<sup>2</sup> Horizon Continental Technology　<sup>3</sup> Horizon
</div>

## 最新动态

* **2026 年 6 月 26 日：** SparseDriveV2 被 ECCV 2026 接收。
* **2026 年 5 月 7 日：** 发布 [Bench2Drive 分支](https://github.com/swc-17/SparseDriveV2/tree/bench2drive)。
* **2026 年 4 月 1 日：** 发布 [论文](https://arxiv.org/abs/2603.29163)、代码和模型权重。

## 目录

- [简介](#简介)
- [结果](#结果)
- [开始使用](#开始使用)
- [致谢](#致谢)
- [引用](#引用)

## 简介

端到端多模态规划常通过给候选轨迹打分并选出最优轨迹，来表示驾驶行为的不确定性。现有方法大致分为两类：对一个较大的静态轨迹词表进行评分，或对少量动态生成的候选轨迹进行评分。静态词表可能因为动作空间离散得较粗而难以覆盖足够多的行为；动态生成的候选轨迹精度更高，在已有基准上也表现出较强的性能。但一个问题仍未明确：要获得良好效果，是否必须动态生成轨迹？如果静态词表足够密集，能否达到相近的性能？

本文首先系统研究了具有代表性的评分式方法 Hydra-MDP。实验显示，随着轨迹锚点变得更密集，性能持续提升；在计算资源达到限制之前，性能尚未出现饱和。受此结果启发，SparseDriveV2 通过两项设计扩展评分式规划的能力：

1. **可扩展的因子化词表表示。** 将轨迹分解为几何路径和速度曲线，使二者可以组合，覆盖大量驾驶动作。
2. **可扩展的分阶段评分策略。** 先分别对路径和速度曲线进行粗筛，再对少量组合轨迹进行精细评分。

结合这两项设计，SparseDriveV2 的轨迹词表规模达到此前方法的 32 倍，同时仍能高效地为密集候选集评分。模型使用轻量级 ResNet-34 作为骨干网络，在 NAVSIM 上取得 92.0 PDMS 和 90.1 EPDMS；在 Bench2Drive 上取得 89.15 Driving Score 和 70.00 Success Rate。

<div align="center">
<b>SparseDriveV2 总体架构</b>
<img src="assets/overview.png"/>
</div>

## 结果

<div align="center">
  <img src="assets/navsimv1.png"/>
  <img src="assets/navsimv2.png"/>
  <img src="assets/bench2drive1.png"/>
  <img src="assets/bench2drive2.png"/>
</div>

## 开始使用

- [NAVSIM 环境准备指南](https://github.com/autonomousvision/navsim?tab=readme-ov-file#getting-started-)
- [数据缓存、训练与评测说明](docs/train_eval.md)

### 模型权重

**NAVSIM v1**

| 方法 | 模型大小 | 骨干网络 | PDMS | 权重下载 |
| :---: | :---: | :---: | :---: | :---: |
| SparseDriveV2 | 50.4M | [ResNet-34](https://huggingface.co/timm/resnet34.a1_in1k) | 92.22 | [Hugging Face](https://huggingface.co/wenchaosun/SparseDriveV2/resolve/main/sparsedrive_navsimv1_92p2.ckpt) |

**NAVSIM v2**

| 方法 | 模型大小 | 骨干网络 | EPDMS* | EPDMS | 权重下载 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| SparseDriveV2 | 50.9M | [ResNet-34](https://huggingface.co/timm/resnet34.a1_in1k) | 87.04 | 90.38 | [Hugging Face](https://huggingface.co/wenchaosun/SparseDriveV2/resolve/main/sparsedrive_navsimv2_90p3.ckpt) |

## 致谢

SparseDriveV2 的设计受到以下优秀开源项目的启发：[NAVSIM](https://github.com/autonomousvision/navsim)、[DiffusionDriveV2](https://github.com/hustvl/DiffusionDriveV2) 和 [GTRS](https://github.com/NVlabs/GTRS)。

## 引用

如果 SparseDriveV2 对你的研究或应用有所帮助，欢迎为项目点亮 Star，并引用本文：

```bibtex
@article{sun2026sparsedrivev2,
  title={SparseDriveV2: Scoring is All You Need for End-to-End Autonomous Driving},
  author={Sun, Wenchao and Lin, Xuewu and Chen, Keyu and Pei, Zixiang and Li, Xiang and Shi, Yining and Zheng, Sifa},
  journal={arXiv preprint arXiv:2603.29163},
  year={2026}
}
```
