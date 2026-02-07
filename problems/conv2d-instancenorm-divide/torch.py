import torch

def solution(input, kernel, output, divide_by, batch_size, in_channels, out_channels,
            height, width, kernel_h, kernel_w):
    out = torch.nn.functional.conv2d(
        input,
        kernel,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    )
    out = torch.nn.functional.instance_norm(out, running_mean=None, running_var=None, use_input_stats=True)
    output[:] = out / divide_by
