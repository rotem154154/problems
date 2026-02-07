import torch

def solution(input, kernel, bias, output, constant_value, scaling_factor,
            batch_size, in_channels, out_channels, height, width, kernel_h, kernel_w):
    out = torch.nn.functional.conv2d(
        input,
        kernel,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    )
    scalar = torch.tensor(constant_value, device=out.device, dtype=out.dtype)
    out = torch.min(out, scalar)
    out = out + bias
    output[:] = out * scaling_factor
