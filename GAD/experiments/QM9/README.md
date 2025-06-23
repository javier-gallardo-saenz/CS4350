## Description 
Here we include the scripts to generate our results on QM9 dataset, including uploading the data, preprocessing steps, etc. 

***To control the operator used to create the eigenmaps***:

-> Laplacian: 
    ```
    --operator=Laplacian
    ```

-> Hubs Laplacian:
    ````
    --operator=Hub_Laplacian  --alpha=0.5 
    ````

-> Hubs Advection-Diffusion Operator:
    ````
    --operator=Hub_Advection_Diffusion --alpha=0.5 --gamma_adv=0.5 --gamma_diff=0.5 (example)
    ````


***To control the diffusion step***:

-> Diffusion Operators:
    ````
    --diff_operator=Laplacian 
    ````
    available: Laplacian, Hub_Laplacian, Hub_Advection_Diffusion

-> Diffusion methods:
    ````
    --diffusion_method=implicit 
    ````
    available:implicit, spectral 
    Note that if one uses Hub_Laplacian or Hub_Advection_Diffusion this value will automatically be overwritten to implicit

-> Diffusion parameter learning:
    ```
    --learn_dif=True 
    ```
    available:True/False
    Note that if this is set to True the diffusion method is automatically overwritten to implicit

-> Alpha parameter for diffusion:
    ````
    --diff_alpha=0.5
    ````

-> Advection gamma parameter for diffusion:
    ````
    --diff_gamma_adv=0.5
    ````

-> Diffusion gamma parameter for diffusion:
    ```
    --diff_gamma_diff=0.5
    ```

-> Number of eigenvectors to be used in spectral scheme:  
    ````
    --k=10
    ````
    We do not use the spectral scheme in our experiments to reduce as much as possible the eigenvalue computation.



***Example Run Command for single parameter optimization***:
```
python -m main_QM9 --n_layers=6 --hid_dim=50 --atomic_emb=25 --dropout=0 --readout=mean --use_diffusion=True --diffusion_method=implicit --k=25 --aggregators mean sum max dir_der --scalers identity amplification attenuation --use_edge_fts=True --use_graph_norm=True --use_batch_norm=True --use_residual=True --type_net=simple --prop_idx 0 --factor 1 --num_epochs=200 --batch_size=48 --lr=1e-3 --weight_decay=3e-6 --min_lr=1e-5 --patience=25 --operator=Hub_Laplacian  --alpha=0.5 --gamma_diff=0 --gamma_adv=0 --diffusion_operator=Laplacian --learn_diff=False --diff_alpha=0 --diff_gamma_adv=0 --diff_gamma_diff=0 --model_name=saved_models\model
```


***Example Run Command for multiparameter optimization***:

```
python -m main_QM9 --n_layers=6 --hid_dim=50 --atomic_emb=25 --dropout=0 --readout=mean --use_diffusion=True --diffusion_method=implicit --k=25 --aggregators mean sum max dir_der --scalers identity amplification attenuation --use_edge_fts=True --use_graph_norm=True --use_batch_norm=True --use_residual=True --type_net=simple --prop_idx 0 1 2 --factor 1 1 10 --num_epochs=200 --batch_size=48 --lr=1e-3 --weight_decay=3e-6 --min_lr=1e-5 --patience=25 --operator=Hub_Laplacian  --alpha=0.5 --gamma_diff=0 --gamma_adv=0 --diffusion_operator=Laplacian --learn_diff=False --diff_alpha=0 --diff_gamma_adv=0 --diff_gamma_diff=0 --model_name=saved_models\model
```




***ORIGINAL GAD MODEL INITIALIZATION***

***To run GAD model on QM9 properties***:

$\mu$ (Dipole moment). Unit: $\textrm{D}$

```
python -m main_QM9 --n_layers=8 --hid_dim=100 --atomic_emb=50 --dropout=0 --readout='mean' --use_diffusion=True --diffusion_method='spectral' --k=25 --aggregators mean sum max dir_der --scalers identity amplification attenuation --use_edge_fts=True --use_graph_norm=True --use_batch_norm=True --use_residual=True --type_net='tower' --towers=5 --prop_idx=0 --factor=1 --num_epochs=300 --batch_size=256 --lr=1e-3 --weight_decay=3e-6 --min_lr=1e-5
```

$\alpha$ (Isotropic polarizability). Unit: ${a_0}^3$
```
python -m main_QM9 --n_layers=8 --hid_dim=100 --atomic_emb=50 --dropout=0 --readout='mean' --use_diffusion=True --diffusion_method='spectral' --k=25 --aggregators mean sum max dir_der --scalers identity amplification attenuation --use_edge_fts=True --use_graph_norm=True --use_batch_norm=True --use_residual=True --type_net='tower' --towers=5 --prop_idx=1 --factor=1 --num_epochs=300 --batch_size=256 --lr=1e-3 --weight_decay=3e-6 --min_lr=1e-5
```

$\epsilon_{\textrm{HOMO}}$ (Highest occupied molecular orbital energy). Unit: $\textrm{meV}$
```
python -m main_QM9 --n_layers=8 --hid_dim=100 --atomic_emb=50 --dropout=0 --readout='mean' --use_diffusion=True --diffusion_method='spectral' --k=25 --aggregators mean sum max dir_der --scalers identity amplification attenuation --use_edge_fts=True --use_graph_norm=True --use_batch_norm=True --use_residual=True --type_net='tower' --towers=5 --prop_idx=2 --factor=1000 --num_epochs=300 --batch_size=256 --lr=1e-3 --weight_decay=3e-6 --min_lr=1e-5
```

$\epsilon_{\textrm{LOMO}}$  (Lowest occupied molecular orbital energy). Unit: $\textrm{meV}$
```
python -m main_QM9 --n_layers=8 --hid_dim=100 --atomic_emb=50 --dropout=0 --readout='mean' --use_diffusion=True --diffusion_method='spectral' --k=25 --aggregators mean sum max dir_der --scalers identity amplification attenuation --use_edge_fts=True --use_graph_norm=True --use_batch_norm=True --use_residual=True --type_net='tower' --towers=5 --prop_idx=3 --factor=1000 --num_epochs=300 --batch_size=256 --lr=1e-3 --weight_decay=3e-6 --min_lr=1e-5
```

$\Delta \epsilon$ (Gap between $\epsilon_{\textrm{HOMO}}$ and $\epsilon_{\textrm{LOMO}}$). Unit: $\textrm{meV}$
```
python -m main_QM9 --n_layers=8 --hid_dim=100 --atomic_emb=50 --dropout=0 --readout='mean' --use_diffusion=True --diffusion_method='spectral' --k=25 --aggregators mean sum max dir_der --scalers identity amplification attenuation --use_edge_fts=True --use_graph_norm=True --use_batch_norm=True --use_residual=True --type_net='tower' --towers=5 --prop_idx=4 --factor=1000 --num_epochs=300 --batch_size=256 --lr=1e-3 --weight_decay=3e-6 --min_lr=1e-5
```

$\langle R^2 \rangle$ (Electronic spatial extent). Unit: ${a_0}^2$
```
python -m main_QM9 --n_layers=8 --hid_dim=100 --atomic_emb=50 --dropout=0 --readout='mean' --use_diffusion=True --diffusion_method='spectral' --k=25 --aggregators mean sum max dir_der --scalers identity amplification attenuation --use_edge_fts=True --use_graph_norm=True --use_batch_norm=True --use_residual=True --type_net='tower' --towers=5 --prop_idx=5 --factor=1 --num_epochs=300 --batch_size=256 --lr=1e-3 --weight_decay=3e-6 --min_lr=1e-5
```

$\textrm{ZPVE}$ (Zero point vibrational energy). Unit: $\textrm{meV}$
```
python -m main_QM9 --n_layers=8 --hid_dim=100 --atomic_emb=50 --dropout=0 --readout='mean' --use_diffusion=True --diffusion_method='spectral' --k=25 --aggregators mean sum max dir_der --scalers identity amplification attenuation --use_edge_fts=True --use_graph_norm=True --use_batch_norm=True --use_residual=True --type_net='tower' --towers=5 --prop_idx=6 --factor=1000 --num_epochs=300 --batch_size=256 --lr=1e-3 --weight_decay=3e-6 --min_lr=1e-5
```
