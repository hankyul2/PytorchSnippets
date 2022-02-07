rt os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark
import torchvision.models as models
from itertools import product

    
def get_upsample_module(ch, groups=2, upsample=2, reduction=2):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(ch, ch // reduction, upsample, stride=upsample, dilation=1, groups=groups, bias=False),
        torch.nn.BatchNorm2d(ch // reduction)
    ).cuda()


def get_point_wise_module(ch, up, groups=1):
    return nn.Sequential(
            nn.Conv2d(ch, ch*up, kernel_size=1, stride=1, padding=0, groups=groups, bias=False),
            nn.BatchNorm2d(ch*up)
        ).cuda()

    
if __name__ == '__main__':
    use_benchmark = True
    amp = True
    channel_last = True

    batch_size = [256]

    # linear-imagenet
    title = 'linear'
    channel_size = [512, 1024, 2048]
    scale_factors = [2048]
    image_size = [1]
    mode = ['linear'] 
    scale_mean='out_channel'

    # upsample-cifar
    # channel_size = [256]
    # scale_factors = [2, 16, 128, 256]
    # image_size = [8, 16]
    # mode = ['deconv'] 
    # scale_mean='groups'

    # upsample-imagenet
    # channel_size = [1024, 2048]
    # scale_factors = [2, 16, 128]
    # image_size = [7, 14]
    # mode = ['deconv'] 
    # scale_mean='groups'

    # point wise-cifar
    # channel_size = [64, 128]
    # image_size = [16, 32]
    # mode = ['point_wise'] 
    # scale_factors = [4, 2]
    # scale_mean = 'channel expansion'

    # point wise-imagenet
    # channel_size = [256, 512]
    # image_size = [14, 28]
    # mode = ['point_wise'] 
    # scale_factors = [4, 8]
    # scale_mean = 'channel expansion'

    torch.backends.cudnn.benchmark = use_benchmark

    results = []

    for b, c, n in product(batch_size, channel_size, image_size):
        label = f'{title} (benchmark={use_benchmark}, amp={amp}, channel_last={channel_last})'
        sub_label = f'[{b}, {c}, {n}, {n}]'
        x = torch.rand((b, c, n, n)).cuda()

        for method, scale in product(mode, scale_factors):
            if method == 'deconv':
                model = get_upsample_module(c, scale)
            elif method =='point_wise':
                model = get_point_wise_module(c, scale)
            elif method =='linear':
                model = nn.Sequential(nn.Flatten(), nn.Linear(c, scale)).cuda()

            if channel_last:
                x = x.to(memory_format=torch.channels_last)
                model = model.to(memory_format=torch.channels_last)
            
            with torch.autocast('cuda', amp), torch.no_grad():
                results.append(benchmark.Timer(
                    stmt='model(x)',
                    setup='from __main__ import model',
                    globals={'x': x},
                    label=label,
                    sub_label=sub_label,
                    description=f"{method}({scale_mean}={scale})",
                    num_threads=4
                ).blocked_autorange(min_run_time=1))

compare = benchmark.Compare(results)
compare.colorize()
compare.print()

    
