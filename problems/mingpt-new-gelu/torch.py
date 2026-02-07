import math
import torch

def solution(input, output, rows, cols):
    output[:] = 0.5 * input * (
        1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0)))
    )
