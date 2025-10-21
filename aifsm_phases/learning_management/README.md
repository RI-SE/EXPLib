# XAI Usages in Learning Management

> **Learning management in the AI‑FSM lifecycle focuses on structuring models to mitigate model epistemic uncertainty in a way that enhances interpretability, modular transparency, and verification of decision‑making logic.**

The usage of XAI is organised around the four main stages of the **Phased Learning Management (PhLM)** workflow:

| Stage | Focus |
|-------|-------|
| **Model Design** | Choose an architecture that limits epistemic uncertainty and decomposes the model into transparent modules. |
| **Model Training** | Use XAI tools to monitor convergence, prevent over‑fitting, and align sub‑objectives. |
| **Model Evaluation** | Quantify trade‑offs between speed, accuracy, and explainability across hyper‑parameter settings. |
| **Model Verification** | Test robustness, adversarial resilience, and explainability on out‑of‑distribution data. |


---

## Model Design 

### Selecting an Architecture
The selection of an appropriate design architecture for a deep learning (DL) model is a crucial first step in reducing epistemic uncertainty related to a specific problem. The chosen model structure serves as the foundation for defining the entire model space, where each trained model is represented by its unique set of parameters, effectively mapping to coordinates within this space. A well-designed model space should ideally encompass the optimal solution and possess a structure that enables efficient convergence towards this ideal solution during the training process.

**Questions to be addressed**:
  1. Does the input data provide sufficient insights to enable the model of such architecture to solve the problem effectively?
  2. Are the dimensions of the model parameters in the model space comprehensive enough to capture and account for all important aspects of the problem adequately?
  3. Can the model space be decomposed into subspaces, corresponding to sub-problems, given the domain knowledge to ensure that the complex problem can be solve correctly via solving its subproblems?

### Modular and transparent architecture

The modular design of deep learning models plays a key role in improving transparency and explainability. By decomposing models into functional blocks, it becomes easier to analyse and validate each component
independently, ensuring that the system’s decision-making logic remains interpretable and traceable. 

**Transparency of internal working of selected model can be provided by some architectural modifications**:
  - **Feature extractor** - Specify key layer(s) of the model’s backbone and add functions to extract activation patterns and/or gradients (requiring backward pass).
  - **Attention module** - add additional block accounting for attentions such as [CBAM](https://doi.org/10.48550/arXiv.1807.06521)
  - **Uncertainty estimator** - add additional block accounting for uncertainty estimates and modify the loss function accordingly.
  - **Interpretable surrogate model** - train an interpretable surrogate model on the same dataset to predict the model predictions. This surrogate model can then be used to explain the main model.
  - **Feature importance:** - Design the model so that it best leverages the extracted dominant prototypes or data features as techniques provided in PhDM
  - **Disentanglement of backbone feature space** - Using architecture design and training techniques to introduce concept representation into the feature space (e.g. specific dimensions are connected to specific concepts).
---

## Model Training

During the Model training step in PhLM, XAI tools are leveraged to minimize model epistemic uncertainty from multiple angles.

| Aspect | XAI enabled practice | 
|--------|-----------------------|
| **Optimizer and scheduler selection** | ensuring that the chosen training optimizer and scheduler promote model robustness and resilience to varying training conditions. | 
| **Convergence monitoring** | verifying that the training process converges to a global optimum, rather than getting stuck in local minima. | 
| **Sub‑objective alignment** |confirming that each subobjective (represented by individual loss components) is properly supported and balanced to achieve overall model objectives | 
| **Over‑fitting prevention** | preventing model overfitting by carefully selecting the batch size and learning rate, and monitoring convergence stability through visualization of iteration logs to minimize fluctuations caused by incoming batches of data | 
| **Reproducibility** | ensuring that the training process is reproducible, allowing for consistent results and facilitating model reliability and trustworthiness. | 

### XAI Techniques During Training

**Global explainers** - Global model explainers can be employed to validate the overall behaviour of the DL model, ensuring it aligns with domain expertise and expectations. Domain experts verify that the model behaves as expected, providing confidence in its performance.

**Intermediate explanation monitoring** - Ensure the model learns correctly:
- *Track progress*: evaluate the model's improvement over time and identify potential issues. 
- *Refine the model*: adjust the training process, hyperparameters, or architecture to optimize convergence and improve overall performance. 
- *Analysing model inner activation*: and feature representation (if model uses or ignores important features).  


**Distribution‑shift analysis** - Compute distribution distance such as Mean Maximum Discrepancy (MMD) between activation patterns on two datasets. The low distance low as data propagates through deeper layers indicates robust internal representations.

---

## Model Evaluation

To determine which model architecture is best suited for a particular problem, it's essential to evaluate different architectural settings and measure their impact on model performance:

| Aspect | XAI enabled practice | 
|------------------|--------------------|
| **Layer configuration** | Assessing the effects of varying layer configurations, such as: Number of layers, Convolutional kernel size, Bottleneck size. | 
| **Hyperparameter tuning** | Optimizing hyperparameters to achieve optimal model performance and balance between speed, accuracy, and explainability. | 


XAI algorithms can also be used to gain a deeper understanding of the model's internal workings:

| Aspect | XAI enabled practice | 
|------------------|--------------------|
| **Feature importance** | Measuring the contribution of different features or components (e.g., heads, skip connections) to the final prediction. |
| **Activation & gradient analysis** | Visualizing activations and gradients at key layers such as last backbone layer, blocks of Feature Pyramid Network, different estimation heads. |


---

## Model Verification
In this step, the goal is to ensure that the model performs well and remains robust across the dataset space, considering variations in input parameters.

| Aspect | XAI enabled practice |
|------------|--------------------------|
| **Trade‑off evaluation** | Analyse the balance between: timing, performance (accuracy), uncertainty and explainability|
| **Performance degradation measurement** | Quantify how the model’s accuracy, robustness, and reliability deteriorate when exposed to generalized datasets (e.g., distribution shifts, adversarial noise). |
| **Correlation analysis** | Identify statistical relationships between model performance metrics and dataset input dimensions. |
| **Domain specific diagnostics** | Verify that the model’s behaviour aligns with established domain knowledge. |


### Report the expected behaviours 

Expected model behaviours shall be logged, and used as known safe boundaries. This is helpful in building OM supervisors that can validate if the model is processing its input data with known behaviours.

| XAI method | What it does | 
|--------|------------------|
| **Performance expectation** | Estimating the model's performance for specific input parameters. | 
| **Uncertainty levels** | Predicting the level of uncertainty associated with different input parameters. | 
| **In-distribution activation patterns** | Analysing the model's internal workings and identifying typical activation patterns for in-distribution data. | 
---

← [Back to **Example usages in AI-FSM**](../README.md)