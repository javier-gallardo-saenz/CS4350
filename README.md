# Hub Laplacian Operators for Directional GNNs

This is the GitHub repository for our CS4350 project, "Hub Laplacian Operators for Directional GNNs."

## 📄 Full Report

The complete project report, `Report_GNN_project.pdf`, can be found [here](Report_GNN_project.pdf).

## 🚀 Running the Models

This repository contains implementations of two models: our custom Graph Convolutional Neural Network (GCNN) and a modified version of Graph Attention Diffusion (GAD).

### 1. Graph Convolutional Neural Network (GCNN)

To run the implemented GCNN:

1. Navigate to the `GCNN/` directory.
2. Open `main.py` and modify the script to choose your desired hyperparameters.
3. Execute the script.

### 2. Modified Graph Attention Diffusion (GAD)

To run the modified GAD model:

1. Go to `GAD/experiments/QM9/`.
2. Refer to the `README.md` file within this directory for detailed instructions and ready-to-use sample commands.

### 3. Dataset spectral analysis + F matrices cosine similarity
To run the spectral analysis of the dataset (or a subset of it) and obtain the cosine similarity between the different F matrices corresponding to each Hub Operator:
1. Go to `GAD/experiments/QM9/`.
2. Modify the hyperparameters in `dataset/evaluation.py` and run. 

---

## 📚 Appendix: Additional Ablation Study

Due to space constraints in the main report, we performed an additional ablation study whose results could not be included. These results can be found in the following file:

`GAD/experiments/QM9/saved_models/prop0_ablation_study/results.txt`
