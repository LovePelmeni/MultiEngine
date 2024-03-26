# Multimodal Recommendation Engine for finding similar posts
Author: Klimushin Kirill Alexandrovich

<img>

# Motivation
Modern society is almost entirely encompassed by different social media and news platforms, which play an essential role in our day to day lifes. As the audience constantly grows and navigating content becomes more challenging, researchers propose new groundbreaking ideas for improving user experience.

# Introduction
In this document, we propose prototype of the social media post recommendation engine, which leverages information from multiple modalities to drive more correct predictions: image and text. 

# Scope
This document contains a low level overview of the system, including internal components and technical approaches, used during development, alongside with technology stack. 

# Architecture

<p align="center">
  <a><img src="https://github.com/LovePelmeni/DeepfakeStream/blob/main/docs/imgs/arch_diagram.png" style="width: 100%; height: 100%"></a>
</p>

# Data management and discovery
This section outlines key data modalities, fusion process and internal representation for intermediate flow inside the network. 

## Image Modality
Image modality represent either image or video file, used as a preview for post. 

## Text Modality
Main content of the post, including title, description, tags and references.

## Data Fusion
Fusion is one of the most crucial steps, as it fuses data in the way that harness power of both modalities, which leads to a more accurate predictions. We selected [Attention-based Fusion](""), which assigns unique weights to each modality vector and applies weighted sum to make a single fused embedding.

<attention mechanism image>

## Internal flow reprensentation
As an intermediate phase before fusion, we typically project multimodal data into K-dimensional embedding, after eliciting valuable data relationships them through [feature extraction encoders]("").

# Components
This section explain demarcation principles and responsibilies of individual system components.

# Encoding phase 
## Image Encoder
ResNet-101 network, pretrained on a custom image dataset for classifying N kinds of post categories. Last layer is replaced with the dense projection layer, which is used for dimensionality reduction to create a [standardized representation embedding]().

## Text Encoder
Module is responsible for analyzing semantic character of the post by its title and description. DistilBERT network, pretrained on a custom text dataset was used in conjunction with dense projection layer for projecting complex output into lower dimensional space to make it compatible for further fusion.

# Fusion phase
## Attention-based fuser

# Recommendation phase

## Recommender
Module is responsible for finding similar items by matching target embedding vector with ones, generated per item. Module uses cutting-edge Facebook's [FAISS]() library to provide an efficient and fast similarity search, while compromising accuracy.

# Train and Evaluation
This section encompasses training, validation and interpretation stages of the overall recommendation system. 

## Interpretability
The high level of complexity inhibit us to troubleshoot the entire system at ones. However project supports interpretation of individual components. Under `src/interpretation` folder you can find functionality for validating behaviour of [Image Embedding Generation Network](), [Text Embedding Generation Network](), [Fusion Layer]().

## Validation strategies
<description of the validation strategies, including sliced evaluation of individual components>

# Security and Privacy
<explain potential security concerns, that may arise>

## References

- [Towards Visualizing Multimodal networks by Paul Pu Liang, Yiwei Lyu, Gunjan Chhablani]("https://arxiv.org/pdf/2207.00056.pdf")
