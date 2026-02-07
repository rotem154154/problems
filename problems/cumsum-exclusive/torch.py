import torch

def solution(input, output, batch_size, length, dim):
    cumsum = torch.cumsum(
        input.narrow(dim=dim, start=0, length=input.size(dim) - 1),
        dim=dim
    )
    leading_zeros = torch.zeros_like(input.select(dim, 0).unsqueeze(dim))
    output[:] = torch.cat((leading_zeros, cumsum), dim=dim)
