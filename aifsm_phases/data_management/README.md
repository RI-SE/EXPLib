# XAI usages in PhDM-Data Management phase

*Effective data management in the AI-FSM lifecycle is critical for reducing domain uncertainties and defining the boundary between known and unknown data distributions. Explainable AI (XAI) techniques play a key role in ensuring data traceability, quality, and reliability, helping to establish a well-defined Operational Design Domain (ODD). This structured approach enables Out-of-Distribution (OOD) detection, ensuring that AI models remain robust against unseen or anomalous data.*

---

## Data requirement specification step
To ensure that the dataset meets the Deep Learning (DL) system’s functional and safety requirements, it must be traceable, reproducible, and free from data integrity issues. AI-FSM leverages XAI-based data profiling, validation, and integrity assessment tools to ensure that datasets align with the AI system’s intended use.

| Activity | Data explainer supports |
|--------|-------------------------|
| **Data traceability** | Establishing a direct link between raw data and DL model requirements ensures transparency in dataset origins. Version control mechanisms track dataset modifications, preventing unintentional data shifts or inconsistencies during retraining. XAI-enhanced dataset lineage tracking enables explainability by maintaining a clear audit trail of data transformations. |
| **Data integrity** | XAI-based data validation rules ensure data quality and consistency by measuring the percentage of missing values and their impact on model learning, continuity of time-series data to identify unexpected gaps in sequences, and statistical consistency by verifying that feature distributions remain within expected ranges. Malicious data modifications, such as data poisoning, are detected using anomaly detection models and distribution consistency checks. |
| **Annotations** | Bounding box consistency checks verify that object annotations are correctly labelled, avoiding misalignment issues that could impact model learning. Data balance analysis ensures that critical classes and features are well-represented in the dataset, preventing biases in model predictions. |
| **Data gaps & compliance tools** | Dataset completeness verification identifies missing data segments that may lead to performance degradation in critical scenarios. Compliant tools include explainability-driven dataset metrics and integrity checks, such as outlier detection for spotting inconsistencies and statistical summaries of dataset properties to detect imbalances and biases. |

---

## Data collection step
The data collection phase establishes a baseline dataset by applying XAI-based analysis and reporting techniques to ensure that collected data is valid, representative, and free from systematic biases.

| Activity | Data explainer supports |
|--------|-------------------------|
|**Baseline & data consistency**|Ensuring dataset consistency involves verifying file format standardization to prevent errors in data loading, dataset volume and balance analysis through histograms and heatmaps to detect imbalances in class distributions, and dataset comparisons to track changes across different collection stages.|
|**Profiling & reporting**| XAI-enhanced data profiling tools, such as [DataGradients](https://github.com/Deci-AI/data-gradients) or SHAP explainers for the entire dataset as distribution, generate statistical graphs that describe feature distributions, label consistency, and dataset shifts over time.|
|**Multi-dimensional data description**|Variational Autoencoders (VAEs) extract latent feature representations, providing a high-dimensional understanding of dataset structure. Clustering techniques, such as k-means and hierarchical clustering, reveal underlying data patterns, improving explainability.|
|**Synthetic data assessment**| Synthetic data augmentation is validated using Fréchet Inception Distance (FID) to measure similarity between synthetic and real data distributions, Kernel Inception Distance (KID) to assess feature-level alignment, and Structural Similarity Index (SSI) to evaluate perceptual consistency.|


---

## Data preparation step
In the data preparation phase, XAI techniques ensure data cleanliness, accurate labelling, balanced distributions, and robust preprocessing.
| Activity | Data explainer supports |
|------|------------------------|
| **Annotation & accuracy** | Statistical plots reveal annotation inconsistencies, such as bounding box overlaps that could lead to misclassifications in object detection models. Class distribution analysis ensures sufficient representation across different ODD parameters, such as weather and time of day. |
| **Augmentation & balance** | Augmentation techniques, including scaling, rotation, and flipping, improve generalization and robustness. XAI-based metrics measure class balance before and after augmentation to ensure that minority classes are sufficiently represented and analyse distribution stability after transformations to prevent data distortions that could affect model performance. |
| **Cleaning & labelling** | Noise characterization and removal use autoencoders to denoise corrupt images and restore missing information. Manual re-labelling recommendations help address datasets where automated correction is insufficient. |
| **Pre-processing & dimension reduction** | Normalization, scaling, and feature selection improve dataset quality while maintaining explainability. XAI-based dimension reduction techniques, such as PCA, UMAP, and t-SNE, help measure distributions across different feature dimensions before and after preprocessing and identify the most important features while maintaining explainability. |

---

## Data verification step
The data verification phase ensures that datasets align with the requirements for deep learning models, reducing domain uncertainties and improving explainability. By leveraging XAI tools, this phase validates whether collected data covers expected distributions, identifies representative samples, and establishes clear boundaries between known and unknown data. 

---
### Data profiling
Data profiling involves evaluating data distributions and structure to confirm that the dataset meets predefined quality and completeness criteria. This step generates reports on dataset characteristics, ensuring that no significant biases or inconsistencies affect model training and inference.

| Technique | Usages |
|---------------------|------------------|
| **Prototype extraction via clustering** | Identifies core representations of objects and concepts, ensuring each category is well‑defined and improving interpretability. |
| **Prototype patching** | Tests whether smaller image regions retain key characteristics required for classification, confirming that models rely on meaningful features rather than background artifacts. |
| **Illustration** | Figure 8 shows extracted prototypes (typical small, squared patches) from the MVP dataset using the MMD‑criticisms algorithm. |

---
### Data prototyping
Data prototyping helps extract representative examples from the dataset, ensuring that key patterns, objects, or concepts are well captured and understandable. This process enables explainability by linking AI decision-making to specific dataset elements.

| Technique | Usages |
|--------|-------------|
| **Class distributions** | Checks that every object category is represented in sufficient quantity to avoid bias in predictions. |
| **Bounding‑box (Bbox) distribution analysis** | Validates the sizes and locations of detected objects to guarantee consistency in training data for object‑detection tasks. |
| **Image statistics** | Assesses image size, resolution, and colour profiles to confirm that all images share a standard format, reducing pre‑processing variability. |
| **ODD & scenario‑related parameter distributions** | Examines contextual data such as time of day, illumination, weather, target distances, viewing angles, and scene variations to ensure a well‑distributed set of operating conditions. |
| **Timestamp‑based data consistency** | Ensures chronological ordering of time‑dependent data, preventing temporal inconsistencies that could damage models relying on sequential inputs. |
| **Illustration** | Figure 7 shows class and bounding‑box distribution for the Railway UC. |

---

### Data descriptors
Data descriptors provide a structured way to summarize dataset characteristics, helping define the boundaries between known and unknown data distributions. These descriptors support the identification of OOD data, ensuring that AI models operate within reliable and well-understood domains.
| Technique | Usages |
|-----------------|--------------|
| **VAE‑based descriptors** | Uses Variational Autoencoders to compress dataset features into a lower‑dimensional latent space, capturing meaningful variations while filtering noise. |
| **Distribution descriptors** | Logs feature distributions across different dataset dimensions, including ODD parameters, annotations, and representations extracted via VAEs or clustering. |
| **Latent‑space analysis** | Applies t‑SNE, UMAP, or PCA to visualize dataset structure, revealing whether categories are well‑separated or overlap (indicating potential misclassifications or biases).|
| **Boundary reporting** | Defines limits between in‑distribution and out‑of‑distribution data, mapping areas where the model may struggle to generalize and ensuring reliable, interpretable decision‑making. |

---
