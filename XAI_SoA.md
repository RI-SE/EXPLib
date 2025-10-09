# Summary of the State of the Art Explainable AI methods and categorization

## 1. Data explainability

Explainable AI (XAI) methods can be used to explain the data used in a DL development/deployment project.
We refer to these methods as “data explainers” within this document.

### 1.1 Taxonomy of data explainers

| Category | What it tells you |
|----------|-------------------|
| **Dataset-level insights** | Provide insights about the dataset | 
| **Data-point insights** | Provide insights about a data point within a dataset | 
| **Proximity / similarity** | How close a data point or a dataset is to another known dataset | 
| **Summarisation / representation** | Compact descriptions of the whole dataset  | 
| **Data mining and profiling** | Explore data insights and summarize dataset | 

---

### 1.2 Data explainer details

#### 1.2.1 Dataset-level insights

| Method | Purpose |Examples |
|------------------|---|-- |
| **EDA**  | Assess data quality of a dataset| [seaborn](https://seaborn.pydata.org/index.html) |
| **Profiling**  | Auto generated data reports for a dataset | [yData](https://github.com/ydataai/ydata-profiling), [SweetViz](https://github.com/fbdesignpro/sweetviz), [Google Facets](https://pair-code.github.io/facets/)  |
| **Dimensionality reduction**  | Visualise datasets in 2D or 3D |[t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html), [UMAP](https://github.com/lmcinnes/umap) |


---

#### 1.2.2 Data-point insights

| Method | Purpose | Examples |
|--------|----------|--|
| **Domain-specific descriptors** | Use domain knowledge to extract relevant features representing a dataset | [HOG](https://scikit-image.org/docs/0.25.x/api/skimage.feature.html#skimage.feature.hog), [SIFT](https://scikit-image.org/docs/0.25.x/api/skimage.feature.html#skimage.feature.SIFT), [LBP](https://scikit-image.org/docs/0.25.x/api/skimage.feature.html#skimage.feature.local_binary_pattern) |
| **Model-based feature engineering**   | Use mathematical models to analyse the inherent structure of a dataset | [Dictionary learning](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.DictionaryLearning.html) / [clustering](https://scikit-learn.org/stable/modules/clustering.html)|


---

#### 1.2.3 Proximity / Similarity

| Method | Purpose | Examples |
|----------|------------------|----------------|
| **Statistical-tests** | Find statistical differences between datasets | [Kolmogorov-Smirnov](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html), [Chi-squared](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html) |
| **Distance / similarity**  | Point-to-dataset or dataset-to-dataset | [Mahalanobis](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.mahalanobis.html), [Bhattacharyya](https://www.jstor.org/stable/25047882), [MMD](https://dl.acm.org/doi/10.5555/2188385.2188410) |
| **Kernel density estimation** | Non-parametric methods to estimate kernel density functions without assuming distribution  | [Gaussian KDE](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html#sklearn.neighbors.KernelDensity) |


---

#### 1.2.4 Summarisation / Representation

| Method | Purpose |Examples |
|-----------|----------------|---|
| **Variational Autoencoders (VAEs)** | Learn a low-dimensional latent space | [VAE](https://doi.org/10.48550/arXiv.1312.6114) |
| **Disentangled VAEs**  | Separate latent factors | [DIPVAE](https://doi.org/10.48550/arXiv.1711.00848) |
| **Deterministic reductions**   | Reveal key axes of variation |[PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html), [LDA](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html), [ICA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html) |


---

#### 1.2.5 Data mining and profiling

| Method | Purpose | Examples |
|------------|------|--|
| **Data Readiness Levels (DRL)**  | Structured assessment of data quality | [Data Readiness Levels](https://doi.org/10.48550/arXiv.1705.02245) |
| **Datasheets for Datasets**   | Structured documentation | [Datasheets for Datasets](https://doi.org/10.48550/arXiv.1803.09010) |
| **Dataset Nutrition Labels**  | Transparency on bias & missingness | [The Dataset Nutrition Label: A Framework To Drive Higher Data Quality Standards](https://doi.org/10.48550/arXiv.1805.03677) |
| **Data Declarations for NLP**  | Record dataset properties | [Data Statements for Natural Language Processing: Toward Mitigating System Bias and Enabling Better Science](https://doi.org/10.1162/tacl_a_00041) |
| **Data Prototype and Criticism** | Unsupervised search for prototypes and criticisms  | [ProtoDash](https://ai-explainability-360.org/), [MMD-critic](https://github.com/BeenKim/MMD-critic) |


---

## 2. Model Explainers

### 2.1 Taxonomy of Model Explainers

| Category | What it tells you | 
|----------|-------------------|
| **Feature and attribution methods** | Pinpoints the contribution of each input feature for a single prediction|
| **Model structure interpretation** | Traces the flow of relevance or influence through the network’s layers/graph to reveal how the model combines evidence. |
| **Semantic understanding** | Maps the model’s internal representations or decisions to high-level *concepts* or *prototypes* that are human understandable. | 
| **Interpretable model design** | Embeds interpretability into the model *architecture*  | 
| **Rule extraction** | Extracts a set of symbolic rules or decision paths that approximate or explain the model. | 
| **Uncertainty and reliability analysis** | Quantifies the model’s robustness to noise or adversarial perturbations | 
| **Enhanced representation** | Projects datasets into lower-dimensional feature spaces that capture underlying structure and highlight feature importance | 




---

### 2.2 Model explainer details

#### 2.2.1 Feature and attribution methods

| Method | Purpose | Examples |
|--------|------------------------|---------------|
| **Activation/Gradient-based** |Utilize gradients/activations of the model response with respect to input data to assign importance scores and propagate them back to individual features| [Saliency map](https://arxiv.org/abs/1312.6034), [Integrated Gradients](https://dl.acm.org/doi/10.5555/3305890.3306024), [Grad-CAM](https://ieeexplore.ieee.org/document/8237336), [DeepLIFT](https://github.com/kundajelab/deeplift), [SmoothGrad](https://arxiv.org/abs/1706.03825),[ Gradient based feature importance](https://doi.org/10.48550/arXiv.1605.01713),  [Guided backpropagation](https://doi.org/10.48550/arXiv.1412.6806), [EigenCAM](https://doi.org/10.1109/IJCNN48605.2020.9206626) | 
| **Perturbation-based** |Assess feature importance by perturbing the input data and analysing the resulting changes in model predictions| [LIME](https://github.com/marcotcr/lime), [SHAP](https://github.com/shap/shap), [Anchors](https://doi.org/10.1609/aaai.v32i1.11491), [Feature occlusion](https://doi.org/10.1007/978-3-319-10590-1_53), [RISE](https://github.com/eclique/RISE), [Counterfactuals](https://arxiv.org/abs/1711.00399) | 
| **Representation-based** |Provide insights into the model's internal workings by learning interpretable and disentangled representations| [TCAV](https://github.com/soumyadip1995/TCAV), [beta-VAE](https://openreview.net/forum?id=Sy2fzU9gl), [ProtoPNet](https://dl.acm.org/doi/10.5555/3454287.3455088), [Neural-symbolic learning](https://doi.org/10.1007/978-3-540-73246-4_4), [infoGAN](https://dl.acm.org/doi/10.5555/3157096.3157340)  | 
| **Graph & Visualization** |Visualize relationships between features and model outputs using graphs| [PDP](https://scikit-learn.org/stable/modules/partial_dependence.html), [ALE](https://doi.org/10.1111/rssb.12377),[Visualizing multi-dimensional decision boundaries in 2D](https://doi.org/10.1007/s10618-013-0342-x)   | 
|**Feature space visualization and dimensionality reductions**| Reduce high-dimensional feature spaces to 2D/3D for exploratory analysis while preserving structural relationships| [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html), [UMAP](https://github.com/lmcinnes/umap),  [Higher-Layer Feature Visualization](https://api.semanticscholar.org/CorpusID:15127402),[Inceptionism](https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html)|

---

#### 2.2.2 Model structure interpretation

| Method | Purpose | Examples |
|--------|------------------------|---------------|
| **Influence tracing** |Aim to understand the model's structure and decision-making process via tracing the influence scores through the network| [LRP](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140), [Deep Taylor](https://doi.org/10.1016/j.patcog.2016.11.008), [Spectral Relevance](https://doi.org/10.1038/s41467-019-08987-4) | 
| **Concept & Feature Analysis** |Focus on analysing and understanding the features and concepts learned by the model| [Concept Relevance Propagation](https://doi.org/10.1038/s42256-023-00711-8), [Decision-boundary visualisation](https://doi.org/10.48550/arXiv.1904.08939) | 
| **Representation and disentanglement** |Aim to disentangle the extracted feature representations in a lower-dimensional feature space| [PCA](https://doi.org/10.1080/14786440109462720), [Isomap](https://doi.org/10.1126/science.290.5500.2319), [beta-VAE](https://openreview.net/forum?id=Sy2fzU9gl), [NMF](https://dl.acm.org/doi/10.5555/248979), [SVD](https://dl.acm.org/doi/10.5555/248979), [LLE](https://doi.org/10.1126/science.290.5500.2323) | 

---

#### 2.2.3 Semantic understanding

| Method | Purpose | Examples |
|--------|------------------------|---------------|
| **Example-based** |Leveraging specific instances from the dataset as examples to illustrate how model derives its prediction, enhancing simulatability of the explanations| [ProtoDash](https://doi.org/10.48550/arXiv.1707.01212), [ProtoPNet](https://doi.org/10.48550/arXiv.1806.10574), [Counterfactuals](https://doi.org/10.48550/arXiv.1711.00399) | 
| **Concept-based** |Mapping the important features to specific concepts| [Concept Relevance Propagation](https://doi.org/10.1038/s42256-023-00711-8), [And-Or Graphs](https://doi.org/10.48550/arXiv.1611.04246) | 

---
#### 2.2.4 Interpretable model design

| Method | Purpose | Examples |
|--------|------------------------|---------------|
| **Example-based** |Leveraging specific instances from the dataset as examples to illustrate how model derives its prediction, enhancing simulatability of the explanations| [ProtoDash](https://doi.org/10.48550/arXiv.1707.01212), [ProtoPNet](https://doi.org/10.48550/arXiv.1806.10574), [Counterfactuals](https://doi.org/10.48550/arXiv.1711.00399) | 
| **Concept-based** |Mapping the important features to specific concepts| [Concept Relevance Propagation](https://doi.org/10.1038/s42256-023-00711-8), [And-Or Graphs](https://doi.org/10.48550/arXiv.1611.04246) | 

---

#### 2.2.5 Rule extraction

| Method | Purpose | Examples |
|----------|---------|----------|
| **Model decomposition / analysis** | Reverse engineer the trained network to express its logic directly | [Anchors](https://doi.org/10.1609/aaai.v32i1.11491), [RxTEN](https://doi.org/10.1007/s11063-011-9207-8),[DEXiRE](https://doi.org/10.3390/electronics11244171), [ECLAIRE](https://doi.org/10.48550/arXiv.2111.12628) |
| **Surrogate models** | Train a simpler, interpretable model to approximate the behaviour of a complex neural network | [DeepRED](https://doi.org/10.1007/978-3-319-46307-0_29),[TrePAN](https://dl.acm.org/doi/10.5555/2998828.2998832), [RuleFit](https://doi.org/10.1214/07-AOAS148), [Two-step CNN rule extractor](https://doi.org/10.3390/electronics9060990) |
| **Neuro-symbolic approaches** | Integrate neural networks with symbolic reasoning techniques, enabling learning and inference under logical constraints | [Neural-symbolic learning](https://doi.org/10.1007/978-3-540-73246-4_4), [DL2](https://proceedings.mlr.press/v97/fischer19a.html), [DeepProbLog](https://doi.org/10.1016/j.artint.2021.103504), [Learning with logical constraints](https://doi.org/10.24963/ijcai.2022/767) |
| **Concept Bottleneck Models (CBM)** | Separate processing into predicting concepts from the input, then predicting task labels from those concepts | [Concept bottleneck](https://dl.acm.org/doi/10.5555/3524938.3525433), [Concept whitening](https://doi.org/10.1038/s42256-020-00265-z)  |
| **Post-hoc Concept Extraction (CME)** | Extract concepts from a pre-trained model’s hidden representations | [Now You See Me](https://doi.org/10.48550/arXiv.2010.13233), [ConceptSHAP](https://dl.acm.org/doi/abs/10.5555/3495724.3497450) |
| **Disentanglement Learning** | Encourage the deep learning model to learn disentangled latent factors that correspond to human-perceived concepts | [Unsupervised β-VAE](https://openreview.net/forum?id=Sy2fzU9gl), [Weakly-supervised](https://doi.org/10.48550/arXiv.2002.02886) |
| **Visual Attention & Captioning** | Leverage attention to generate natural-language explanations (e.g., image captions) that highlight relevant visual features | [Show, attend and tell: neural image caption generation with visual attention](https://dl.acm.org/doi/10.5555/3045118.3045336) |

---


#### 2.2.6 Uncertainty and reliability analysis

| Method | Purpose | Examples |
|--------|------------------------|---------------|
| **Bayesian Neural Network (BNN)** |Model uncertainty by placing priors over its weights and making predictions by integrating over the posterior| [Laplace approximation](https://dl.acm.org/doi/10.5555/2986766.2986882), [Variational Inference](https://dl.acm.org/doi/10.1145/168304.168306), [Markov Chain Monte Carlo](https://dl.acm.org/doi/10.5555/525544),  | 
| **Robustness tests** |Evaluate a model's resilience to noisy inputs | [Explaining and Harnessing Adversarial Examples](https://doi.org/10.48550/arXiv.1412.6572), [Certified Adversarial Robustness via Randomized Smoothing](https://doi.org/10.48550/arXiv.1902.02918) | 

---

#### 2.2.7 Enhanced representation

| Method | Purpose | Examples |
|--------|------------------------|---------------|
| **Attention mechanisms** |Learn to assign different weights to different elements, highlighting those that contribute most to the task| [GALA](https://doi.org/10.48550/arXiv.1805.08819), [DomainNet](https://doi.ieeecomputersociety.org/10.1109/CVPR.2015.7298685), [Residual Attention](https://doi.org/10.1109/CVPR.2017.683), [Loss-based attention](https://doi.org/10.1109/TIP.2020.3046875), [D-Attn](https://doi.org/10.1145/3109859.3109890) | 
| **Autoencoders-based representation** | The latent space can be analysed to understand which features are most important for encoding the input.| [Explainer models](https://ui.adsabs.harvard.edu/link_gateway/2018arXiv180507468Z/doi:10.48550/arXiv.1805.07468), [XCNN](https://doi.org/10.48550/arXiv.2007.06712) | 
|**Others**|Enhance representations| [Adaptive DeConv](https://doi.org/10.1109/ICCV.2011.6126474), [Deep Fuzzy Classifier](https://doi.org/10.1109/TFUZZ.2019.2946520), [Network In Network]([NIN](https://doi.org/10.48550/arXiv.1312.4400)) |

---

