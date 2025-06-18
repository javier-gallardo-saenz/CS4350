from hub_utils.hub_operators import hub_laplacian_dense_pytorch, hub_advection_diffusion_laplacian_dense_pytorch


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



