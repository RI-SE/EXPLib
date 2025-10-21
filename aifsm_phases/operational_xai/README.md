# XAI usages in Operation and Monitoring stage

> **Within the Operational and Monitoring (OM) stage, the residual uncertainties (irreducible uncertainties and residual reducible uncertainties) are managed by different safety components within the safety architecture. The components aim to identify and assess uncertainties, making sure that the system is operating safely as it has been designed for (upon the steps taken in AI-FSM lifecycle).**

Different uncertainty types can be monitored by XAI enabled supervisors as follows:


| Uncertainties | Description | Supervisory monitors|
|-------|-------|-------|
| **Residual domain uncertainties** | Out of Distribution: input data in runtime has not been seen (not likely belong to the datasets generated from AI-FSM PhDM). |Anomaly detectors (input data, extracted feature, output prediction)|
| **Residual model epistemic uncertainty** | Low confidence of model predictions (or high uncertainty) due to non-representative training dataset or non-ideal model architecture. | Uncertainty-aware model, surrogate model  logical ruled constraints |
| **Aleatoric uncertainty** | Sensor noises, annotation uncertainties, occlusion, adversarial attacks. |Aleatoric uncertainty aware model, adversarial detector, distribution shift detector|

---


← [Back to **Example usages in AI-FSM**](../README.md)
