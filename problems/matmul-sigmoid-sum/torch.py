import torch

def solution(input, weight, bias, output, batch_size, input_size, hidden_size):
    out = torch.nn.functional.linear(input, weight, bias)
    out = torch.sigmoid(out)
    output[:] = torch.sum(out, dim=1, keepdim=True)
