# XAI Usages in Inference Management

> **Inference management in AI-FSM ensures that deployed models operate within acceptable safe boundaries while maintaining consistency with their original training behaviour. Explainable AI (XAI) techniques play a key role in validating inference stability, monitoring deviations, and detecting Out-of-Distribution (OOD) inputs. By continuously assessing model predictions against expected behaviours, inference management ensures that decisions remain interpretable, reliable, and aligned with safety requirements.**


---

## Verify that the model performs as intended in real-world scenarios
A key requirement for inference management is to verify that the model performs as intended in real-world scenarios. XAI techniques help compare inference-time decisions with the original model's expected behaviour by:

- **Tracking feature importance shifts** between training and inference phases to ensure that the model relies on the same key features for decision-making.
- **Analysing activation patterns** in neural layers to detect any unexpected deviations that could indicate model drift or changes in learned representations.
- **Ensuring stability in explainability** metrics, such as attention maps and saliency visualizations, to confirm that inference-time reasoning remains consistent with training logic. 
- **Comparing performance distributions** across different ODD conditions (e.g., different lighting conditions, object scales, viewing 
angles) to ensure inference robustness across varying environments.

## Integrating and testing safety components 

Inference models must operate within predefined safe boundaries, ensuring that predictions remain
trustworthy and interpretable. To achieve this, AI-FSM incorporates supervisory monitors, which act as a
safety layer to cross-check predictions, detecting potential misclassifications or model failures before they
impact decision-making:

- **Uncertainty estimation**, where models quantify their confidence levels in predictions, flagging high-uncertainty outputs for human review or fallback mechanisms.
- **Anomaly detection** for OOD inputs, which identifies when inference data significantly deviates from training data distributions. Techniques such as distance metrics in latent space, probabilistic modelling, and clustering-based detection help recognize unexpected scenarios.

## Trustworthiness and explainability of optimized models

To verify that the inference process remains transparent and trustworthy, various explainability techniques can be applied:
- **Feature attribution methods**, such as SHAP or LIME, analyse whether the model is using the correct input features for its predictions.
- **Prototype-based verification**, ensuring that the model’s inferences align with known, representative samples rather than relying on spurious correlations.
- **Saliency maps**, which highlight the most important regions in input data that contribute to the model’s decision-making. By visualizing which areas of an image, text, or structured data impact predictions, saliency maps provide insights into the reasoning process and help detect cases where models rely on unintended or misleading features.

← [Back to **Example usages in AI-FSM**](../README.md)