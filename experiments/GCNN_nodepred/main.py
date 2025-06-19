#main.py
from torch.nn import Tanh, ReLU
from operators import hub_laplacian, adv_diff
from gcnn_train import run_experiment
from evaluate_similarity import eval_cos_sim_per_layer, plot_avg_similarity, plot_similarity_heatmap

if __name__ == "__main__":
    PARAMS = {
        "lr":           1e-3,
        "weight_decay": 1e-5,
        "num_epochs":   200,
        "dims":         [34, 4, 4],
        "out_dim":      4,                      # number of classes in Karate Club dataset                             
        "degrees":      [1]* 2,                 # must be = to number of hidden layer
        "act_fns":      [Tanh()]* 2,            # must be = to number of hidden layer
        "alpha":        0.5,                    # alpha to initialize GSO
        "readout_dims": [4, 4],                 # first dimension must match last of the hidden layer
        "apply_readout": True,
        "gso_generator": hub_laplacian,
        "eval_embeddings": True,  # set to True if you want to return embeddings
    }
    
    best_val_ce, test_acc, val_ce_history, embeddings = run_experiment(PARAMS)
    print("\nExperiment Summary:")
    print(f"Best Validation Cross-Entropy: {best_val_ce:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    #print(f"Validation Cross-Entropy History: {val_ce_history}")

    # evaluate cosine similarity of embeddings per layer
    sim_matrices, avg_cos_sims = eval_cos_sim_per_layer(embeddings, mask=None)

    #for i, sim_matrix in enumerate(sim_matrices):
    #    plot_similarity_heatmap(sim_matrix, title=f"Layer {i + 1} Cosine Similarity")

    plot_avg_similarity(avg_cos_sims)

