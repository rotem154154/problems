import torch

def solution(input, weight, bias, output, dropout_p, batch_size, in_features, out_features):
    out = torch.nn.functional.linear(input, weight, bias)
    out = torch.nn.functional.dropout(out, p=dropout_p, training=False)
    output[:] = torch.softmax(out, dim=1)
