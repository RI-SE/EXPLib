# Uncertainty Management for Safety‑Critical Deep Learning

EXPLib implements a structured, SOTIF-aligned uncertainty management workflow for autonomous systems that use deep learning.
The goal is to keep the system safely inside the **Known / Not Hazardous** space while continuously monitoring and mitigating all relevant uncertainties.

---

## Table of Contents

- [What problem does it solve?](#what-problem-does-it-solve)
- [Core ideas](#core-ideas)
- [Development lifecycle](#development-lifecycle)
- [Operation & monitoring](#operation--monitoring)


---

## What problem does it solve?

Deep‑learning (DL) models for autonomous driving, robotics, etc. bring *unknown* behaviour in *unknown* environments.
SOTIF (ISO 21448) defines a safety requirement: *“A system shall remain safe even when the environment is not fully understood.”*
EXPLib gives a **structured, algorithmic approach** to:

- **Identify** the three uncertainty sources
  - *Domain uncertainty* - dataset / environment mismatch
  - *Model epistemic uncertainty* - lack of knowledge in the DL model
  - *Aleatoric uncertainty* - inherent noise in observations/annotations
- **Reduce** the *reducible* part of uncertainties during development
- **Monitor** the residual uncertainty at runtime 
- **Trigger** mitigation actions when estimated risk resulted from the uncertainties exceeds a threshold

---

## Core ideas

| Step | Focus | Goal |
|------|-------|------|
| **AI‑FSM (Development)** | Uncertainty **reduction** | Engineer a model that works safely inside the **Known** space; estimate irreducible uncertainty |
| **OM (Operation & Monitoring)** | Runtime safety | Keep the system in safe boundaries; continuously evaluate residual risk; activate safe mitigation when necessary |



---

## Development lifecycle (AI‑FSM)

1. **Data collection** – label a *Known* dataset.
2. **Domain shift detection** – use Domain Uncertainty estimation methods to flag outliers.
3. **Model learning and Inference** – minimise model epistemic uncertainty.
4. **Validation** – evaluate residual uncertainties of both types.
5. **Boundary definition** – set statistical thresholds that separate Known/Unknown.
6. **Anomaly detector design** – trained on the residual uncertainty profile.

---

## Operation & monitoring (OM)

- **Risk threshold** – if exceeded inform the Decision Control.
- **Continuous learning** – store samples that cross the boundary (rejected samples) for subsequent development iterations.
- **Safety envelope** – guarantees the system not get into the *UnKnown/Hazardous* region.

---
