# EXPLib

EXPLib – A library for explainable and dependable DL components

![Logo](./assets/EXPLib_logo.png) <a name="logo"></a>

- Domain: Agnostic 
- Keywords: deep learning, explainable AI, computer vision, functional safety
- Responsible unit at RISE: Digital Systems/Mobility and Systems/Human Centered AI
- Contributors: [RISE](https://www.ri.se/en) (main contributor), [Ikerlan](https://www.ikerlan.es/en), [AIKO](https://aikospace.com/)
- Related research projects: [SAFEXPLAIN](https://safexplain.eu/)
- License: [GPL3-0](https://github.com/RI-SE/EXPLib/blob/main/LICENSE)

## Description

EXPLib is an open-source research library built to accelerate the deployment of *explainable and dependable* deep-learning (DL) components.  At its core, EXPLib bundles a curated selection of state-of-the-art XAI techniques, wrapped in a lightweight Python library. 

Beyond simply providing algorithms, EXPLib provides how to use explainability artifacts to support relevant activities in **AI‑FSM (Artificial Intelligence‑Functional Safety Management) lifecycle**.  This ensures that DL related artifact (model, data) is transparent, traceable, and compliant with AI-FSM process, and in turns to other safety and regulatory standards (IEC61508, ISO 26262, ISO 21448,...). 

## What is inside

[Approach](approach.md)

[Example usages with AI-FSM](Place_holder)

[Explainable AI methods and library](XAI_SoA.md)

## Disclaimer: The use of open-source software in EXPLib

EXPLib itself is released under the GNU license, however the library **does not include its own full stack of dependencies**.  Instead, it pulls in a diverse ecosystem of open-source projects. When you build, extend, or deploy EXPLib, you are automatically adopting those projects and, consequently, their respective licenses, stability guarantees, and security postures.

## Getting Started

```
$ git clone https://github.com/RI-SE/EXPLib.git
$ cd EXPLib
```

## How to cite this work

## Branching Model

The EXPLib development follows the popular [git-flow](https://nvie.com/posts/a-successful-git-branching-model/) branching model. The model uses two *infinite* branches (`master` and `develop`) and two types of supporting branches (`feature` and `hotfix` branches). Supporting branches shall be *ephemeral*, i.e., they should only last as long as the feature or hotfix itself is in development. Once completed, they shall be merged back into one of the infinite branches and/or discarded.

In the following examples, `feature-x` or `hotfix-x` shall be replaced with a short phrase describing the feature or hotfix.

- `master` - the main branch where the source code of HEAD always reflects a production-ready state.
- `develop` - where the main development is reflected. Merges into `master`.
- `feature-x` - used to develop new features for the upcoming release. Merges into `develop`.
-	`hotfix-x` - used when it is necessary to act immediately upon an undesired state of a live production version. Merges into `master`.

The repository administrators are responsible for deleting the remote copies of ephemeral branches and updating the version tag for the `master` branch.

External pull requests are welcome, but must be reviewed before they can be merged into the master branch. Reviewers may ask questions or make suggestions for edits and improvements before your feature can be merged. If your feature branch pull request is not accepted, make the necessary adjustments or fixes as indicated by the repository administrators and redo the pull request.

For a longer description of the branching model, please refer to our [examples](https://github.com/RI-SE/EXPLib/blob/main/branching.md).

## License

This library is licensed under the GNU License – see the [GPL3-0](https://github.com/RI-SE/EXPLib/blob/main/LICENSE) file for details.