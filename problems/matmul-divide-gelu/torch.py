import torch

def solution(input, weight, bias, output, divisor, batch_size, input_size, output_size):
    out = torch.nn.functional.linear(input, weight, bias)
    out = out / divisor
    output[:] = torch.nn.functional.gelu(out)
