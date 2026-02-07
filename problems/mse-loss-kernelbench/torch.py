import torch

def solution(predictions, targets, output, batch_size, length):
    output[:] = torch.mean((predictions - targets) ** 2)
