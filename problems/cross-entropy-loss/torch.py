import torch

def solution(predictions, targets, output, batch_size, num_classes):
    output[:] = torch.nn.functional.cross_entropy(predictions, targets) 