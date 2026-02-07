import torch

def solution(input, kernel, conv_bias, gn_weight, gn_bias, output, num_groups,
            batch_size, in_channels, out_channels, height, width, kernel_h, kernel_w):
    conv_out = torch.nn.functional.conv2d(
        input,
        kernel,
        bias=conv_bias,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    )
    normed = torch.nn.functional.group_norm(conv_out, num_groups=num_groups, weight=gn_weight, bias=gn_bias)
    tanh_out = torch.tanh(normed)
    hardswish_out = torch.nn.functional.hardswish(tanh_out)
    residual = conv_out + hardswish_out
    output[:] = torch.logsumexp(residual, dim=1, keepdim=True)
