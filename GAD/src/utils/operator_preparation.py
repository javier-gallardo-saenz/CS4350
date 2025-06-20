from hub_utils.hub_operators import (hub_laplacian_dense_pytorch, hub_advection_diffusion_laplacian_dense_pytorch,
                                     get_hub_laplacian_dense_pytorch, get_hub_advection_diffusion_laplacian_dense_pytorch)


def get_operator_and_params(operator_name: str, alpha: float = 0, gamma_adv: float = 0, gamma_diff: float = 0):
    if operator_name == 'Laplacian':
        # alpha is always 0 for Laplacian, so it's a fixed kwarg
        operator = hub_laplacian_dense_pytorch
        kwargs = {'alpha': 0}
        return operator, kwargs

    elif operator_name == 'Hub_Laplacian':
        operator = hub_laplacian_dense_pytorch
        # alpha is passed directly from the input for Hub_Laplacian
        kwargs = {'alpha': alpha}
        return operator, kwargs

    elif operator_name == 'Hub_Advection_Diffusion':
        operator = hub_advection_diffusion_laplacian_dense_pytorch
        # Here, we create a dictionary with named keyword arguments
        kwargs = {
            'alpha': alpha,
            'gamma_adv': gamma_adv,
            'gamma_diff': gamma_diff
        }
        return operator, kwargs

    else:
        raise ValueError('Unknown operator_name')


def get_diff_operator_and_diff_type(diffusion_operator_name, learn_diff=False, alpha=0, gamma_diff=0, gamma_adv=0):
    diff_parameters = {'alpha': alpha, 'gamma_adv': gamma_adv, 'gamma_diff': gamma_diff}

    if diffusion_operator_name == 'Laplacian':
        diffusion_operator = get_hub_laplacian_dense_pytorch
        diff_parameters['alpha'] = 0.0
        if learn_diff:
            diffusion_type = "standard"
        else:
            diffusion_type = "learnable_degree_operators"

    elif diffusion_operator_name == 'Hub_Laplacian':
        diffusion_operator = get_hub_laplacian_dense_pytorch
        if learn_diff:
            diffusion_type = "learnable_degree_operators"
        else:
            diffusion_type = "degree_operators"

    elif diffusion_operator_name == 'Hub_Advection_Diffusion':
        diffusion_operator = get_hub_advection_diffusion_laplacian_dense_pytorch
        if learn_diff:
            diffusion_type = "learnable_degree_operators"
        else:
            diffusion_type = "degree_operators"

    else:
        raise ValueError('Unknown diffusion_operator_name')

    return diffusion_operator, diffusion_type, diff_parameters





