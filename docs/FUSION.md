# Multimodal data fusion in MultiEngine

Fusion is undoubtedly the most critical component, requiring thorough
and well elaborated design. It should accurately reflect data and combine 
modalities with the aim to improve and uplift current model capabilities by considering
data, which came from manifold independent sources. 

Here I want to list critical requirements we kept in mind
when embarked on design of fusion strategy.

1. Accuracy blow up (fusion provides slight or significant blow up in accuracy, which is essential).
2. Robust to missing modalities (in case some articles may lack images, we still need to handle request with grace.)
3. Interpretability (fusion layer can be explained, analyzed and understood).
4. Light weight (consumes tenable amount of resources).

# Solution: 
One of the prevalent and publicly approved
approahes, that conform our expectations is called (Stacked Latent Attention-based Fusion)[" https://openaccess.thecvf.com/content_cvpr_2018/papers/Fan_Stacked_Latent_Attention_CVPR_2018_paper.pdf"]. It assigns
weights for each modality, which enforces them to contribute to the final output
in different proportions, so the network can distinguish between different modalities better. Moreover, authors proposed a new way of factoring multiple modalities in the way, that improves spatial reasoning as described in the paper itself.

<p align="center">
  <a><img src="https://github.com/LovePelmeni/MultiEngine/blob/main/docs/imgs/fusion_layer.png" style="width: 90%; height: 90%"></a>
</p>

# Implementation
The code for multimodal attention-based fusion algorithm is available under 'src/multimodal/fusions/attention_fusion.py' and has implementation in PyTorch.

- [Attention-based Fusion for multimodal video description]("https://arxiv.org/abs/1701.03126")
- [Attention Based Feature Fusion For Multi-Agent Collaborative Perception]("https://arxiv.org/abs/2305.02061")
- [Attention-based Fusion for Outfit Recommendation by Katrien Laenen, Marie-Francine Moens]("https://arxiv.org/abs/1908.10585")
