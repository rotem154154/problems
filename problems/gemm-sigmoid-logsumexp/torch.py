import torch

def solution(input, weight1, bias1, weight2, bias2, output,
            batch_size, input_size, hidden_size, output_size):
    out = torch.nn.functional.linear(input, weight1, bias1)
    out = torch.sigmoid(out)
    out = torch.nn.functional.linear(out, weight2, bias2)
    output[:] = torch.logsumexp(out, dim=1)
