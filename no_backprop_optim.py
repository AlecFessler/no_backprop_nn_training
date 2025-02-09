import copy
import torch

def generate_model_copies(model, N, device):
    models = []

    for _ in range(N):
        model_copy = copy.deepcopy(model)
        model_copy.to(device)
        models.append(model_copy)

    return models

def perturb_params(model, perturb_prob=0.5, perturb_range=0.01):
    with torch.no_grad():
        for param in model.parameters():
            mask = torch.rand_like(param) < perturb_prob
            perturbs = torch.randn_like(param) * perturb_range
            param.data += perturbs * mask

def linear_combination_scalars(losses):
    losses_tensor = -torch.tensor(losses)
    return torch.softmax(losses_tensor, dim=0)

def linear_combine_models(models, scalars, device):
    combined_model = copy.deepcopy(models[0])

    with torch.no_grad():
        for name, param in combined_model.named_parameters():
            param.data = torch.zeros_like(param.data)
            for model, scalar in zip(models, scalars):
                param.data += dict(model.named_parameters())[name].data * scalar

    combined_model.to(device)
    return combined_model
