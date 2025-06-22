#main.py
from torch.nn import Tanh, ReLU
from operators import hub_laplacian, adv_diff
from gcnn_train import run_experiment
from evaluate_similarity import eval_cos_sim_per_layer, plot_avg_similarity, plot_similarity_heatmap
from data import get_karateclub_data, get_karateclub_data_custom

if __name__ == "__main__":
    PARAMS = {
        "lr":           1e-4,
        "weight_decay": 1e-5,
        "num_epochs":   200,
        "dims":         [34, 16],
        "out_dim":      4,                      # number of classes in Karate Club dataset                             
        "degrees":      [1]* 2,                 # must be = to number of hidden layer
        "act_fns":      [ReLU()]* 2,            # must be = to number of hidden layer
        "alpha":        0,                      # alpha to initialize GSO
        "readout_dims": None,                   # first dimension must match last of the hidden layer
        "apply_readout": True,
        "gso_generator": hub_laplacian,
        "eval_embeddings": True,  # set to True if you want to return embeddings
    }
    
    best_val_ce, test_acc, val_ce_history, embeddings, pred = run_experiment(PARAMS) # pred only returned if eval_embeddings=True

    data, _, _, _ = get_karateclub_data_custom()  # Load the dataset to get the true labels
    true_labels = data.y  # Assuming data.y contains the labels

    # get true labels from the dataset

    print("\nExperiment Summary:")
    print(f"Best Validation Cross-Entropy: {best_val_ce:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    #print(f"Validation Cross-Entropy History: {val_ce_history}")

    # evaluate cosine similarity of embeddings per layer
    
    sim_matrices, avg_cos_sims, avg_intra_class_sims, avg_inter_class_sims = eval_cos_sim_per_layer(embeddings, labels=true_labels)

    for i, sim_matrix in enumerate(sim_matrices):
        plot_similarity_heatmap(sim_matrix, title=f"Layer {i + 1} Cosine Similarity")

    plot_avg_similarity(avg_cos_sims)
    plot_avg_similarity(avg_intra_class_sims, title="Average Intra-Class Cosine Similarity per Layer")
    plot_avg_similarity(avg_inter_class_sims, title="Average Inter-Class Cosine Similarity per Layer")
    