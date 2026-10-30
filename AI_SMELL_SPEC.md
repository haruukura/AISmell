# Teaching an AI to Smell: System Specification & Architecture Guide

Based on the research paper *"A principal odor map unifies diverse tasks in olfactory perception"* (Google Research & Monell Chemical Senses Center, published in *Science*, 2023) and the OpenPOM open-source implementation.

---

## 1. High-Level Overview & Biological Basis

- **Biological Olfaction**: The human olfactory system utilizes ~400 types of odorant receptors. Molecules bind to combinatorial subsets of these receptors, which neural circuitry in the olfactory bulb and brain interprets as complex olfactory perceptions (e.g., "fruity", "minty", "smoky", "fishy").
- **AI Approach (OpenPOM)**: Bypasses biological receptor simulation entirely. Directly maps chemical molecular structure (graph representation) to perceptual odor descriptors (multi-label semantic tokens).
- **Principal Odor Map (POM)**: A continuous 256-dimensional latent embedding space where chemical distance correlates with perceptual smell similarity.

---

## 2. Molecular Representation & Feature Pipeline

### 2.1 Input Format: SMILES
- Molecules are represented as **SMILES** (Simplified Molecular Input Line Entry System) strings (e.g., Ethanol = `CCO`).
- Hydrogens are implicit.

### 2.2 Graph Formulation
- **Nodes (Atoms)**:
  - Element type (Carbon, Nitrogen, Oxygen, Sulfur, Halogens, etc. as one-hot or indexed embeddings).
  - Degree (number of bonded atoms).
  - Formal charge.
  - Aromaticity / ring membership.
  - Hybridization state (sp, sp2, sp3).
- **Edges (Bonds)**:
  - Bond type (single, double, triple, aromatic).
  - Conjugation and ring membership.
  - Stereo configuration.

---

## 3. Ground Truth Labels & Dataset Characteristics

- **Dataset**: Curated dataset of **4,983 molecules** (`curated_datasets/curated_gs_lf_4983.csv`).
- **Target Descriptors**: **138 olfactory semantic labels** (multi-label classification).
  - Each sample is represented as a multi-hot binary vector $\mathbf{y} \in \{0, 1\}^{138}$.
  - Examples: `fishy`, `rose`, `sweet`, `cinnamon`, `fruity`, `smoky`.
- **Label Noise**: Fragrance and flavor catalog data contains subjective perceptual noise.
- **Extreme Class Imbalance**:
  - Frequent labels: e.g., `fruity` (~1,902 positive molecules).
  - Rare labels: e.g., `smoky` (~72 positive molecules).
  - Mitigation: **Iterative Stratified Split** (80% train, 10% val, 10% test) and **Class-weighted Binary Cross-Entropy Loss**.

---

## 4. Model Architecture: Message Passing Neural Network (MPNN)

1. **Atom & Bond Embeddings**: Initial linear projections of node and edge feature vectors into hidden dimension $d_h$ (typically 128 or 256).
2. **Message Passing Layers (5 Rounds)**:
   - In each round $t \in \{1, \dots, 5\}$:
     $$m_v^{(t)} = \sum_{u \in \mathcal{N}(v)} M_t(h_v^{(t-1)}, h_u^{(t-1)}, e_{uv})$$
     $$h_v^{(t)} = U_t(h_v^{(t-1)}, m_v^{(t)})$$
   - Propagation allows atoms up to 5 hops away to exchange contextual chemical state (e.g., on ethanol, $C_1$ learns about terminal $O$ by hop 2).
3. **Graph Readout / Pooling**:
   - Aggregates node representations across the entire molecule (e.g., sum pooling, attention pooling, or set2set readout) into a fixed-length molecular vector.
4. **Principal Odor Map (POM) Latent Dense Layer**:
   - Dense linear layer with **256 units**, followed by activation and **dropout (0.12)**.
   - Molecules with perceptually similar odor profiles cluster together in this 256-D metric space.
5. **Multi-Label Classification Head**:
   - Linear layer mapping $256 \to 138$ logits.
   - Element-wise Sigmoid $\sigma(z_i)$ for 138 independent binary probabilities.

---

## 5. Training, Optimization & Evaluation

- **Loss Function**: Weighted Binary Cross-Entropy (BCEWithLogitsLoss) across 138 tasks:
  $$\mathcal{L} = - \frac{1}{C} \sum_{c=1}^{C} w_c \left[ y_c \log \sigma(z_c) + (1 - y_c) \log (1 - \sigma(z_c)) \right]$$
  where $w_c$ compensates for label rarity.
- **Regularization**: Dropout rate = 0.12.
- **Learning Rate**: Decaying learning rate schedule (warmup + cosine / step decay).
- **Epochs**:
  - Smoke test / demo: 5 epochs.
  - Production benchmark: 150 epochs.
- **Ensemble Strategy**:
  - Average probabilities from an ensemble of 10 independently trained models with different seed initializations to achieve steady state performance.
- **Primary Evaluation Metric**:
  - **Macro ROC-AUC**: Area Under the Receiver Operating Characteristic curve computed per label and averaged over all 138 labels.
  - Target: ~0.89 ROC-AUC with 10-model ensemble.

---

## 6. Hardware Sensor Branch (Optional Physical Nose)

- **Hardware**: Bosch BME688 gas sensor breakout board (e.g., Adafruit $19.95) connected via I2C to Raspberry Pi.
- **Mechanism**: Reads electrical resistance at 10 distinct heater temperature steps, yielding a 10-dimensional gas signature per sampling instance.
- **Software**: Bosch BME AI-Studio (Windows).
- **Training**: Direct supervised classifier on sensor patterns (independent of OpenPOM GNN).
- **Critical Pitfalls**:
  - Baseline sensor drift and humidity fluctuations: always record across different sessions/days.
  - Sensor cannot identify unknown novel smells; it will classify inputs into the nearest trained cluster.

---

## 7. Common Implementation Traps

1. **No Pretrained Model Weights in OpenPOM Repo**: The weights are not distributed as pre-baked artifacts; models must be trained from the curated CSV dataset.
2. **Environment & Architecture Requirements**:
   - DeepChem, DGL, and OpenPOM dependencies have native C++/CUDA bindings requiring Linux x86_64 / Colab GPU runtime.
3. **Class Imbalance in Data Splitting**: Random splitting will starve validation and test sets of rare descriptors (e.g., smoky). Iterative stratification is mandatory.
