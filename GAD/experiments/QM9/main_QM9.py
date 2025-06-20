import torch
import torch.optim as opt
import torch.nn as nn

from torch_geometric.datasets import QM9
from torch_geometric.data import DataLoader
import os
import sys

from tqdm import tqdm

import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), "../")) 
sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/")) 

from utils.preprocessing import preprocessing_dataset, average_node_degree
from utils.operator_preparation import get_operator_and_params, get_diff_operator_and_diff_type
from train_eval_QM9 import train_epoch, evaluate_network
from train_QM9 import train_QM9
from GAD_QM9.gad import GAD


def main():

    parser = argparse.ArgumentParser()
    

    parser.add_argument('--n_layers', help="Enter the number of GAD layers", type=int)
    parser.add_argument('--hid_dim', help="Enter the hidden dimensions", type=int)
    parser.add_argument('--atomic_emb', help="Enter the embedding dimensions of the atomic number", type=int)
    
    parser.add_argument('--dropout', help="Enter the value of the dropout", type=int, default=0)
    parser.add_argument('--readout', help="Enter the readout agggregator", type=str, default='mean')

    
    parser.add_argument('--use_diffusion', help="Enter the use_diffusion", type=bool)
    parser.add_argument('--diffusion_method', help="Enter the diffusion layer solving scheme ", type=str)
    parser.add_argument('--diffusion_operator', help="Enter the operator used to perform the initial "
                                                     "diffusion in each layer", type=str, default='Laplacian')
    parser.add_argument('--learn_diff', help="Enter whether to learn the parameter alpha of the diffusion operator",
                        type=bool, default=False)
    parser.add_argument('--diff_alpha', help="Enter the initial value for the parameter alpha of the diffusion operator",
                        type=float, default=0.0)
    parser.add_argument('--diff_gamma_adv', help="Enter the initial value for the advection weight of the diffusion"
                                            " advection-diffusion operator", type=float, default=0.0)
    parser.add_argument('--diff_gamma_diff', help="Enter the initial value for the diffusion weight of the "
                                                  "diffusion advection-diffusion operator", default=0.0, type=float)
    parser.add_argument('--k', help="Enter the num of eigenvector for spectral scheme", type=int)

    parser.add_argument('--aggregators', nargs='+', help="Enter the aggregators (space-separated list)", type=str)
    parser.add_argument('--scalers', nargs='+', help="Enter the scalers (space-separated list)", type=str)

    parser.add_argument('--use_edge_fts', help="Enter true if you want to use the edge_fts", type=bool)
    parser.add_argument('--use_graph_norm', help="Enter true if you want to use graph_norm", type=bool, default=True)
    parser.add_argument('--use_batch_norm', help="Enter true if you want to use batch_norm", type=bool, default=True)
    parser.add_argument('--use_residual', help="Enter true if you want to use residual connection", type=bool)

    parser.add_argument('--type_net', help="Enter the type_net for DGN layer", type=str, default='simple')
    parser.add_argument('--towers', help="Enter the num of towers for DGN_tower", type=int, default=3)

    parser.add_argument('--prop_idx', help="Enter the QM9 property index", type=int)
    parser.add_argument('--factor', help="Enter the factor 1000 to convert the QM9 property with Unit eV"
                                         " to meV. Enter 1 for the others properties", type=int)

    
    parser.add_argument('--num_epochs', help="Enter the num of epochs", type=int)
    parser.add_argument('--batch_size', help="Enter the batch size", type=int)
    parser.add_argument('--lr', help="Enter the learning rate", type=float)
    parser.add_argument('--weight_decay', help="Enter the weight_decay", type=float)
    parser.add_argument('--min_lr', help="Enter the minimum lr", type=float)

    parser.add_argument('--operator', help="Enter the desired operator whose eigenvectors will define "
                                           "the gradient maps", type=str, default='Laplacian')
    parser.add_argument('--alpha', help="Value of the hub operator parameter alpha", default=0, type=float)
    parser.add_argument('--gamma_adv', help="Enter advection weight for advection-diffusion operator",
                        default=0, type=float)
    parser.add_argument('--gamma_diff', help="Enter diffusion weight for advection-diffusion operator",
                        default=0, type=float)


    
    args = parser.parse_args()

    DATA_DIR = './data/QM9'

    # --- Create the directory if it doesn't exist ---
    os.makedirs(DATA_DIR, exist_ok=True)

    # check if QM9 already exists locally to avoid redownloading
    processed_dir = os.path.join(DATA_DIR, 'processed')
    if os.path.exists(processed_dir) and len(os.listdir(processed_dir)) > 0:
        print(f"Loading QM9 dataset from local storage: {DATA_DIR}")
        # force_reload=False ensures it doesn't redownload/reprocess if already present
        dataset = QM9(root=DATA_DIR, force_reload=False)
    else:
        print(f"QM9 dataset not found locally in {DATA_DIR}. Downloading and processing...")
        # This will trigger download and initial processing
        dataset = QM9(root=DATA_DIR, force_reload=True)  # force_reload=True is default for first time
        print("Download and initial processing complete.")

    dataset = dataset.shuffle()  # This shuffles the entire dataset in-place

    # Define the split points
    size = 1000
    test_size = int(0.1*size)
    val_size = int(0.1*size)
    train_size = int(0.8*size)

    dataset_test = dataset[:test_size]
    dataset_val = dataset[test_size:val_size+test_size]
    dataset_train = dataset[val_size+test_size:train_size+val_size+test_size]  # All remaining samples

    print("Dataset loading and splitting complete.")
    print(f"dataset_train contains {len(dataset_train)} samples")
    print(f"dataset_val contains {len(dataset_val)} samples")
    print(f"dataset_test contains {len(dataset_test)} samples")

    print("data preprocessing: calculate and store the vector field F, etc.")

    operator, params = get_operator_and_params(args.operator, args.alpha, args.gamma_adv, args.gamma_diff)

    D, avg_d = average_node_degree(dataset_train)
    dataset_train = preprocessing_dataset(dataset_train, args.k, operator, **params)
    dataset_val = preprocessing_dataset(dataset_val, args.k, operator, **params)
    dataset_test = preprocessing_dataset(dataset_test, args.k, operator, **params)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    train_loader = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=os.cpu_count() // 2, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(dataset=dataset_val, batch_size=args.batch_size, shuffle=False,
                            num_workers=os.cpu_count() // 2, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False,
                             num_workers=os.cpu_count() // 2, pin_memory=True, persistent_workers=True)


    diff_operator, diff_type, diff_parameters = get_diff_operator_and_diff_type(args.diffusion_operator, args.learn_diff,
                                                                                args.diff_alpha,
                                                                                args.diff_gamma_adv,
                                                                                args.diff_gamma_diff)

    print("create GAD model")
    
    model = GAD(num_of_node_fts=11, num_of_edge_fts=4, hid_dim=args.hid_dim, atomic_emb=args.atomic_emb,
                graph_norm=args.use_graph_norm, batch_norm=args.use_batch_norm, dropout=args.dropout,
                readout=args.readout, aggregators=args.aggregators, scalers=args.scalers, edge_fts=args.use_edge_fts,
                avg_d=avg_d, D=D, device=device, towers=args.towers, type_net=args.type_net, residual=args.use_residual,
                use_diffusion=args.use_diffusion, diffusion_method=args.diffusion_method,
                diffusion_type=diff_type,
                diffusion_param=diff_parameters,
                k=args.k, n_layers=args.n_layers)
    

    model = model.to(device)
    
    optimizer = opt.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    train_QM9(model, optimizer, train_loader, val_loader, prop_idx=args.prop_idx, factor=args.factor, device=device,
              num_epochs=args.num_epochs, min_lr=args.min_lr, diffusion_operator=diff_operator)
    
    print("Uploading the best model")

    model_ = torch.load('model.pth')

    test_mae = evaluate_network(model_, test_loader, args.prop_idx, args.factor, device)
    val_mae = evaluate_network(model_, val_loader, args.prop_idx, args.factor, device)
    train_mae = evaluate_network(model_, train_loader, args.prop_idx, args.factor, device)

    print("")
    print("Best Train MAE: {:.4f}".format(train_mae))
    print("Best Val MAE: {:.4f}".format(val_mae))
    print("Best Test MAE: {:.4f}".format(test_mae))


if __name__ == '__main__':
    main()

   
