# CS4248 Team 14

# 🔬🌐 IWSLT Dataset: Resolving Polysemous Words in Neural Machine Translation

## Abstract

The goal of this project is to improve the performance of neural machine translation (NMT) systems by addressing the challenge of polysemy in source languages. Polysemous words can have multiple meanings depending on their context, which can lead to incorrect translations. We propose a novel approach that leverages contextual information from surrounding sentences to disambiguate polysemous words and enhance translation accuracy.

We will evaluate our methods on the 2017 IWSLT dataset, which contains parallel corpora for various language pairs. For the purposes of our experiment, we will focus on the English-to-Chinese translation tasks. Our approach involves creating pseudo documents by concatenating multiple sentences via a sliding window and training a transformer-based NMT model, namely OpenNMT, to learn the contextual relationships between words. Moreover, we implement a word sense disambiguation (WSD) module that utilizes contextual embeddings from BERT to identify the correct sense of polysemous words in the source language. This module will be integrated into the NMT model in efforts of improving translation quality. We will then compare the performance of our model with a baseline transformer model to assess the effectiveness of our approach.

## Folder Structure

```bash
├── backbone/
│   ├── en-zh-opennmt/
|   ├── MT-preperation/
│   ├── run/
│   ├── 1_preprocessing.ipynb
│   ├── 2_training.ipynb
│   ├── 3_analysis.ipynb
│   ├── 4_data_preparation.ipynb
│   ...
├── pos_tagging/
├── ...
├── submission/
│   ├── analysis/
├── ├── ├── high_level_analysis.ipynb
│   ├── diagrams/
├── wsd/
```

## Setup

### Install dependencies

### Running the translation model

## Acknowledgments
