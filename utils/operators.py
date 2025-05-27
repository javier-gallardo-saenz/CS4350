import numpy as np

def laplacian_standard(adj):
    degrees = np.sum(adj, axis=1)
    D = np.diag(degrees)
    return D - adj

def laplacian_hubs_repelling(adj):
    degrees = np.sum(adj, axis=1)
    safe_degrees = np.where(degrees == 0, 1, degrees)
    
    scaling = (degrees[:, None] / safe_degrees[None, :]) * (adj > 0)
    diag_entries = np.sum(scaling, axis=1)
    
    Eps_A = np.diag(diag_entries)
    D = np.diag(degrees)
    D_inv = np.diag(1 / safe_degrees)
    
    return Eps_A - D @ adj @ D_inv

def laplacian_hubs_attracting(adj):
    degrees = np.sum(adj, axis=1)
    safe_degrees = np.where(degrees == 0, 1, degrees)

    scaling = (safe_degrees[None, :] / degrees[:, None]) * (adj > 0)
    diag_entries = np.sum(scaling, axis=1)

    Eps_R = np.diag(diag_entries)
    D = np.diag(degrees)
    D_inv = np.diag(1 / safe_degrees)

    return Eps_R - D_inv @ adj @ D

def compute_laplacian(adj, alpha=0):
    """
    Dispatch to one of the Laplacian variants based on alpha.
    """
    if alpha == 0:
        return laplacian_standard(adj)
    elif alpha == 1:
        return laplacian_hubs_repelling(adj)
    elif alpha == -1:
        return laplacian_hubs_attracting(adj)
    else:
        raise ValueError("alpha must be -1, 0, or 1")

#adj = np.array([[0, 1, 0, 0],[1, 0, 1, 0],[0, 1, 0, 1],[0, 0, 1, 0]])
adj = np.array([[0, 1, 1, 1],
                [1, 0, 1, 0],
                [1, 1, 0, 0],
                [1, 0, 0, 0]])              

L_R = compute_laplacian(adj,1)
L_A = compute_laplacian(adj,-1)
L = compute_laplacian(adj,0)

print("Hubs-attracting Laplacian matrix L_A:")
print(L_A)

print("Laplacian matrix L")
print(L)

print("Hubs-repelling Laplacian matrix L_R:")
print(L_R)