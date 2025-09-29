# Summary of the State of the Art Explainable AI methods and categorization

## 1. Data explainability

Explainable AI (XAI) methods can be used to explain the data used in a DL development/deployment project.
We refer to these methods as “data explainers” within this document.

### 1.1 Taxonomy of data explainers

| Category | What it tells you |
|----------|-------------------|
| **Dataset‑level insights** | Global statistics, distributions, clustering | 
| **Data‑point insights** | Feature importance for individual samples | 
| **Proximity / similarity** | How close a sample or a dataset is to a known dataset | 
| **Summarisation / representation** | Compact statistical views of the whole dataset  | 
| **Data readiness & provenance** | Readiness, documentation, and quality metadata | 

---

### 1.2 Data explainer details

#### 1.2.1 Dataset‑level insights

| Method | Purpose |Examples |
|------------------|---|-- |
| **EDA**  | Basic stats, missing‑value counts | [seaborn](https://seaborn.pydata.org/index.html) |
| **Profiling**  | Auto generated data reports | [yData](https://github.com/ydataai/ydata-profiling), [SweetViz](https://github.com/fbdesignpro/sweetviz), [Google Facets](https://pair-code.github.io/facets/)  |
| **Dimensionality reduction**  | Visualise clusters in 2D or 3D |[t‑SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html), [UMAP](https://github.com/lmcinnes/umap) |


---

#### 1.2.2 Data‑point insights

| Method | Purpose | Examples |
|--------|----------|--|
| **Domain‑specific descriptors** | Use domain expertise and EDA insights | [HOG](https://scikit-image.org/docs/0.25.x/api/skimage.feature.html#skimage.feature.hog), [SIFT](https://scikit-image.org/docs/0.25.x/api/skimage.feature.html#skimage.feature.SIFT), [LBP](https://scikit-image.org/docs/0.25.x/api/skimage.feature.html#skimage.feature.local_binary_pattern) |
| **Model‑based feature engineering**   | Use mathematical models to analyse the inherent structure of a dataset | [Dictionary learning](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.DictionaryLearning.html) / [clustering](https://scikit-learn.org/stable/modules/clustering.html)|


---

#### 1.2.3 Proximity & Similarity

| Method | Purpose | Examples |
|----------|------------------|----------------|
| **Statistical-tests** | Distribution differences | [Kolmogorov‑Smirnov](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html), [Chi‑squared](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html) |
| **Distance / similarity**  | Point‑to‑dataset or dataset‑to‑dataset | [Mahalanobis](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.mahalanobis.html), Bhattacharyya, MMD |
| **Kernel density estimation** | Non‑parametric density | Gaussian [KDE](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html#sklearn.neighbors.KernelDensity) |


---

#### 1.2.4 Summarisation & Representation

| Method | Purpose |Examples |
|-----------|----------------|---|
| **Variational Autoencoders (VAEs)** | Learn a low‑dimensional latent space | VAE |
| **Disentangled VAEs**  | Separate latent factors | [DIPVAE](https://aix360.readthedocs.io/en/latest/die.html) |
| **Deterministic reductions**   | Reveal key axes of variation |[PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html), [LDA](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html), [ICA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html) |


---

#### 1.2.5 Data mining and profiling

| Method | Purpose | Examples |
|------------|------|--|
| **Data Readiness Levels (DRL)**  | Structured assessment of data quality | [Kirk et al., 2020](https://www.nasa.gov/mission_pages/ames/people/engineering/Data_Readiness_Levels.html) |
| **Datasheets for Datasets**   | Structured documentation | [Gebru et al., 2018](https://arxiv.org/abs/1908.07832) |
| **Dataset Nutrition Labels**  | Transparency on bias & missingness |[Kleiner et al., 2020](https://arxiv.org/abs/2002.02879) |
| **Data Declarations for NLP**  | Record dataset properties | [Zhang et al., 2021](https://arxiv.org/abs/2106.07623) |
| **Data Prototype and Criticism** | Unsupervised search for prototypes and criticisms  | [ProtoDash](https://ai-explainability-360.org/), [MMD-critic](https://github.com/BeenKim/MMD-critic) |


---

## 2. Model Explainers

### 2.1 Taxonomy of Model Explainers

| Category | What it tells you | 
|----------|-------------------|
| **Feature Attribution** | Importance of each input feature for a single prediction | 
| **Structural** | Flow of relevance through the network layers or graph | 
| **Concept‑based** | High‑level semantic concepts that drive decisions | 
| **Counterfactual & Contrastive** | “What‑if” scenarios that would change the prediction | 
| **Representation & Prototype** | Self‑explanatory architectures that expose reasoning | 
| **Uncertainty & Reliability** | Confidence and robustness of predictions | 
| **Hybrid & Multi‑modal** | Integration of multiple explainer families or modalities | 


---

### 2.2 Model Explainer Details

#### 2.2.1 Feature & Attribution Methods

| Method | Purpose | Examples |
|--------|------------------------|---------------|
| **Activation/Gradient‑based** |Utilize gradients of the model output with respect to input data to assign importance scores and propagate them back to individual features| [Saliency](https://arxiv.org/abs/1312.6034), [Integrated Gradients](https://dl.acm.org/doi/10.5555/3305890.3306024), [Grad‑CAM](https://ieeexplore.ieee.org/document/8237336), [DeepLIFT](https://github.com/kundajelab/deeplift), [SmoothGrad](https://arxiv.org/abs/1706.03825) | 
| **Perturbation‑based** |Assess feature importance by perturbing the input data and analysing the resulting changes in model predictions| [LIME](https://github.com/marcotcr/lime), [SHAP](https://github.com/shap/shap), [Anchors](https://doi.org/10.1609/aaai.v32i1.11491), [RISE](https://github.com/eclique/RISE), [Counterfactuals](https://arxiv.org/abs/1711.00399) | 
| **Representation‑based** |Provide insights into the model's internal workings by learning interpretable and disentangled representations| TCAV, beta‑VAE, ProtoPNet | 
| **Graph & Visualization** |Visualize relationships between features and model outputs using graphs| PDP, ALE, t‑SNE, UMAP, Inceptionism | 

---

#### 2.2.2 Model‑Structure Interpretation

| Method | Purpose | Examples |
|--------|------------------------|---------------|
| **Influence tracing** |Aim to understand the model's structure and decision-making process via tracing the influence scores through the network| LRP, Deep Taylor, Spectral Relevance | 
| **Concept & Feature Analysis** |Focus on analysing and understanding the features and concepts learned by the model| Concept Relevance Propagation, Decision‑boundary visualisation | 
| **Disentanglement** |Aim to disentangle the extracted feature representations in a lower-dimensional feature space| PCA, Isomap, beta‑VAE, NMF | 

---

#### 2.2.3 Semantic Understanding

| Method | Purpose | Examples |
|--------|------------------------|---------------|
| **Example‑based** |Leveraging specific instances from the dataset as examples to illustrate how model derives its prediction, enhancing simulatability of the explanations| ProtoDash, ProtoPNet, Counterfactuals | 
| **Concept‑based** |Mapping the important features to specific concepts| Concept Relevance Propagation, And‑Or Graphs | 

---

#### 2.2.4 Uncertainty & Reliability Analysis

| Method | Purpose | Examples |
|--------|------------------------|---------------|
| **Bayesian NNs** |Evaluate various uncertainties by replacing traditional network weights with probabilistic variables| Probabilistic weights, Dropout as Bayesian approximation, Ensemble variance | 
| **Robustness tests** |Evaluate a model's resilience to noisy inputs | Adversarial, perturbation tolerance | 

---

#### 2.2.5 Enhanced Representation

| Method | Purpose | Examples |
|--------|------------------------|---------------|
| **Attention mechanisms** |Learn to assign different weights to different elements, highlighting those that contribute most to the task| GALA, Residual Attention, DomainNet | 
| **Autoencoders-based representation** | The latent space can be analysed to understand which features are most important for encoding the input.| Explainer models, XCNN | 

---

