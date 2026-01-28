# EEG-FM-Benchmark

## :speech_balloon: Annoucement
- [2026.01.27] 🚩 **News**  The manuscript of MIRepNet can be found in [EEG Foundation Models: Progresses, Benchmarking, and Open Problems](https://arxiv.org/abs/2601.17883).

- [2025.01.25] We propose a **fair and comprehensive benchmarking for open source EEG foundation models**.

## 📌 Abstract
Electroencephalography (EEG) foundation models have recently emerged as a promising paradigm for brain-computer interfaces (BCIs), aiming to learn transferable neural representations from large-scale heterogeneous recordings. Despite rapid progresses, there lacks fair and comprehensive comparisons of existing EEG foundation models, due to inconsistent pre-training objectives, preprocessing choices, and downstream evaluation protocols. This paper fills this gap. We first review 50 representative models and organize their design choices into a unified taxonomic framework including data standardization, model architectures, and self-supervised pre-training strategies. We then evaluate 12 open-source foundation models and competitive specialist baselines across 13 EEG datasets spanning nine BCI paradigms. Emphasizing real-world deployments, we consider both cross-subject generalization under a leave-one-subject-out protocol and rapid calibration under a within-subject few-shot setting. We further compare full-parameter fine-tuning with linear probing to assess the transferability of pre-trained representations, and examine the relationship between model scale and downstream performance. Our results indicate that: 1) linear probing is frequently insufficient; 2) specialist models trained from scratch remain competitive across many tasks; and, 3) larger foundation models do not necessarily yield better generalization performance under current data regimes and training practices.

![Benchmarking](rank.png)

## 🚀  Contributions

**A comprehensive overview of existing BCI foundation models.**
- 🧩 We survey 50 BCI foundation models, constituting the most comprehensive collection to date.
- 🛠️ We provide a detailed and structured comparison of their technical designs, encompassing basic information, pre-training data scale, preprocessing pipelines, pre-training strategies, and architectural choices.
- 🎯 We propose a unified taxonomic framework for EEG foundation models that organizes existing work into a coherent design space.
- 
**Fair and comprehensive benchmarking for open source EEG foundation models.**
- 🧩 We systematically compare "full parameter fine-tuning" with "classification head fine-tuning" across various models and tasks to assess whether pre-trained encoders provide broadly transferable EEG representations. Beyond the commonly used leave one subject out (LOSO) scenario, we introduce a within-subject few-shot adaptation scenario in which the fine-tuning data volume is approximately 1/20 ~ 1/100 of that typically used in LOSO protocols.
- 🛠️ We comprehensively compare traditional machine learning methods, CNN-based models, and Transformer-based models trained from scratch against fine-tuned EEG foundation models to evaluate whether conventional approaches remain competitive.
- 🎯 We evaluate EEG foundation models of varying parameter sizes pre-trained on diverse datasets to investigate whether a larger model necessarily leads to better generalization performance.


