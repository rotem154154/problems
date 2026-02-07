import torch

def solution(input_a, input_b, output_c, batch, m, n, k):
    output_c[:] = torch.bmm(input_a, input_b)
