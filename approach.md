# Uncertainty Management for Safety‑Critical Deep Learning

EXPLib implements a structured, SOTIF-aligned uncertainty management workflow for autonomous systems that use deep learning.
The goal is to keep the system safely inside the **Known / Not Hazardous** space while continuously monitoring and mitigating all relevant uncertainties.

## Table of Contents
- [Uncertainty management strategy](#uncertainty-management-strategy)
- [Uncertainty management recommendations](#uncertainty-management-recommendations)
  - [Reduction of uncertainties within the development lifecycle](#reduction-of-uncertainties-within-the-development-lifecycle)
    - [Reduction of domain uncertainty](#reduction-of-domain-uncertainty)
    - [Reduction of model epistemic uncertainty](#reduction-of-model-epistemic-uncertainty)
    - [Logging of aleatoric uncertainty](#logging-of-aleatoric-uncertainty)
  - [Management of residual uncertainty within Operation and Monitoring stage](#management-of-residual-uncertainty-within-operation-and-monitoring-stage)
    - [Management of data/model epistemic uncertainty](#management-of-data-model-epistemic-uncertainty)
    - [Management of aleatoric uncertainty](#management-of-aleatoric-uncertainty)

---
# Uncertainty management strategy
A challenge in integrating DL into safety-critical autonomous systems is addressing the inherent uncertainty associated with its predictions. It is essential to adopt a structured and systematic approach that manages and mitigates this uncertainty to the greatest extent possible. The Safety of the Intended Functionality (SOTIF) provides a guideline for achieving this goal. SOTIF defines scenario categories based on two dimensions: Known/Unknown and Hazardous/Not hazardous. This framework highlights the importance of understanding and managing uncertainty to ensure safe and reliable system operation.

Effective management of uncertainty during both development and deployment phases can be achieved by formally identifying and separating the distinct sources of  uncertainty. In the following, the disentangled sources of uncertainty are classified into three primary categories: domain uncertainty, model epistemic uncertainty, and aleatoric uncertainty.

These uncertainties are addressed throughout the lifecycle as follows:
- **Uncertainty reduction in the Development Lifecycle (AI‑FSM phases)**:  During development lifecycle the focus is on identifying and minimizing reducible uncertainties. Methods and approaches are used to estimate irreducible uncertainty and the residual components of reducible uncertainty. A clear definition of the boundaries between the *Known* and *Unknown* areas is established, providing a baseline for the anomaly detectors employed in the OM stage. This ensures that the system operates within its safe boundaries, with all known uncertainties properly managed and mitigated.

- **Uncertainty management in the Operation and Monitoring (OM) Stage**:  In this stage the architecture incorporates safety components that guarantee operation strictly within its known scope (safe boundaries). The mitigation strategy for known uncertainty is continuously verified for effectiveness. Residual uncertainty is continually estimated, and the risk of entering hazardous situations is assessed. If the calculated risk exceeds an acceptable threshold, the decision-making module is alerted to execute mitigation actions, thereby ensuring the system’s safe operation and preventing potential hazards.

# Uncertainty management practical approaches

## Reduction of uncertainties within the AI-FSM compliant development lifecycle
### Reduction of domain uncertainty
AI‑FSM Data Management (PhDM) Phase is specifically designed to address mitigation of domain uncertainty. XAI methods belonging to the data explainer category can be used to support different steps within the AI‑FSM PhDM phase as follows:

- Identification of analytical and statistical gaps between the collected dataset and the real‑world
This will be done with the supports of data explainers in the PhDM phase.

  - **Data value completeness analysis**: Identify feature importance in missing values to determine which features are most critical to the problem.
  - **Feature‑wise completeness analysis**: Extract important features from the data and evaluate missing features that are crucial to supporting the problem.
  - **Reconstruction of missing data**: Use data descriptors to describe the dataset and synthesize missing data samplings to close the data distribution gap.
  - **Data balance analysis**:
    - Visualize feature distribution and data‑point distribution across selected data dimensions to identify potential biases or imbalances.
    - Measure distribution distances between datasets and design approaches to close gaps, including strategies to mix different datasets (e.g., synthetic and collected datasets) to improve overall data quality in terms of distributional representativeness.
  - **Data relevance analysis**: Extract prototypes and/or criticisms to identify data points or features most representative of the dataset and problem space.
  - **Data accuracy analysis**:
    - Analyse annotation distributions and identify potential mislabelled instances (e.g., anomalous labels).
    - Use feature‑importance methods to explain how different data features influence annotations.

- Data collection planning

  - **Uncertainty analysis**: Use uncertainty quantification techniques (e.g., Bayesian neural networks) to identify high‑uncertainty areas that may require more data to be collected.
  - **Diversity analysis**: Apply clustering methods to guide data balance across unsupervised clusters, ensuring that the collected data is diverse and representative.

- Data preparation

  - **Data synthetic**: Use GAN‑based synthetic data to augment real‑world data points in under‑represented areas.
  - **Feature importance guided**: Identify important features where data augmentation should be focused.
  - **Counterfactual guided augmentation**: Use counterfactual examples to guide data augmentation and improve the robustness of the dataset.
  - **Noise modelling and augmentation**: Identify realistic data noises and adversarial noises and use them to augment the dataset and address potential vulnerabilities.

---

### Reduction of model epistemic uncertainty

The reduction of model epistemic uncertainty is done in the AI‑FSM PhLM phase, provided the assumption that data domain uncertainty has been already mitigated in the PhDM phase. Various model explainers can be employed, targeting AI developer audience, including the below:

- **Improving model architectures**: Where applicable, design models with decomposition of concepts to ensure that all sub‑concepts are accurately modelled and can be thoroughly tested. This approach enables the identification and mitigation of potential flaws in the model architecture.
- **Enhancing model explainability**: Utilize XAI techniques to increase the transparency of the model (intrinsic or post‑hoc designs).
- **Design uncertainty‑aware models**: Leverage uncertainty quantification techniques to modify models, enabling them to provide not only predictions but also estimates of epistemic uncertainty.
- **Analysing model global/local behaviour**: Employ various model explainers to verify the learned feature space and assess the model’s robustness to input changes, including both expected and unexpected variations.

---

### Logging of aleatoric uncertainty

Aleatoric uncertainty, being irreducible, can be quantified and estimated using XAI methods within the AI‑FSM PhLM phase. To achieve this, aleatoric‑uncertainty aware models will be designed and trained on the same datasets used for the primary model. This approach enables the estimation of uncertainty levels associated with the data quality and label quality.

Known uncertainties corresponding to the verified datasets are used as the baseline to support the OM stage.

---

## Management of residual uncertainty within Operation and Monitoring stage
### Management of data/model epistemic uncertainty

Residual epistemic uncertainty from both data and model will be mitigated by XAI‑enabled supervision components to ensure the trustworthiness of the system, even in the presence of uncertainty. The key XAI components include:

- **Out-of‑Distribution (OOD) detectors**: These detectors monitor the input data, model activation patterns, and output predictions to ensure they fall within established boundaries. These boundaries are defined based on the knowledge gained during the AI‑FSM phases, as documented in corresponding artifacts.
- **Certified surrogate model**: This safe alternative model provides a fallback option when epistemic uncertainty levels exceed predefined acceptable limits. The certified surrogate model guarantees prediction confidence and handles situations where the primary model’s uncertainty is too high.


### Management of aleatoric uncertainty

The aleatoric uncertainty‑aware model will be integrated into the OM safety architecture, providing insights to inform decision‑making and ensure reliable system performance. The employment of this model enables various approaches, including:

- **Trade‑off mechanisms**: When estimated uncertainty exceeds an acceptable threshold, the system can adapt by:
  - Running inference on multiple consecutive frames, increasing processing delay but reducing uncertainty to an acceptable level.
  - Dynamically adjusting the trade‑off between prediction accuracy and latency, ensuring that the system operates within defined safety bounds.
- **Model fusion**: The decision function, based on an ensemble model, will utilize estimated uncertainty as input to:
  - Produce final consolidated predictions with higher certainty, considering the estimated uncertainty of individual model outputs.
  - Assign weights to each model’s output based on its corresponding uncertainty estimate.
- **Defining safe limits for system operation**: The aleatoric uncertainty‑aware model will help establish safe limits for the system, where:
  - Prediction variations within predefined boundaries can be tolerated and used by the decision‑making component.
  - The system’s operating range is defined, ensuring that predictions are reliable and trustworthy, even in the presence of uncertainty.