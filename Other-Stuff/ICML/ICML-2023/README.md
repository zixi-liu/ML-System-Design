
## Tutorials

### Tutorial on Multimodal Machine Learning: Principles, Challenges, and Open Questions

**Multimodal**

Connected Modalities
- Shared information that related modalities

Interacting Modalities
- process affecting each modality
- interactions happen during inference

Technical Challenges
- Representation
  - local representation
    - fusion
    - coordination
    - fission
- Alignment
  - synchronize modality
    - discrete alignment
- Reasoning
- Generation
- Transference
- Quantification

**Representation**

Basic Fusion
- Additive Interactions
- Multiplicative Interactions
- Gated Fusion

Modality-Shifting Fusion

Mixture of Fusions

Nonlinear Fusion

Fusion with Heterogeneous Modalities

Image Represenion Learning: Masked Autoencoder (MAE)

Dynamic Early Fusion 

Heterogeneity-aware Fusion

**Coordination**

Coordination with Contrastive Learning
- Visual-Semantic Embeddings
- CLIP (Contrastive Language-Image Pre-training)

**Alignment**

Discrete Alignment

Continuous Alignment

Contextualized Representation
- join undirected alignment
  - early fusion in sequence dimension (transformer self-attention)
- cross-modal directed alignment
  - Attention for language-vision similarities -> new visually contextualized representation no language
  - high-modality multimodal transformers
    - transfer across partially observable modalities
    - some implicit assumptions
      - all modalities can be representated as sequences without losing information
      - dimensions of heterogeneity can be perfectly captured by modality-specific embeddings
      - cross-modal connections & interactions are shared across modalities and tasks
- alignment with unimodal models
- structured alignment
