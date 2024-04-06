import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def agg_default(x):
    if x.ndim == 4:
        return np.abs(x).sum((0, 1))

    elif x.ndim == 3:
        return np.abs(x).sum(0)


epsilon = 1e-10


def clip(x, top_clip=True):
    if x.ndim == 3:
        batch_size, height, width = x.shape
        x = x.reshape(batch_size, -1)
        if top_clip:
            vmax = np.percentile(x, 99, axis=1, keepdims=True)
        else:
            vmax = np.max(x, axis=1, keepdims=True)
        vmin = np.min(x, axis=1, keepdims=True)
        vdiff = vmax - vmin
        for i, v in enumerate(vdiff):
            v = max(0, np.abs(v))
            if np.abs(v) < epsilon:
                x[i] = np.zeros_like(x[i])
            else:
                x[i] = np.clip((x[i] - vmin[i]) / v, 0, 1)
        x = x.reshape(batch_size, height, width)
    elif x.ndim == 2:
        height, width = x.shape
        x = x.ravel()
        x = np.nan_to_num(x)
        vmax = np.percentile(x, 99) if top_clip else np.max(x)
        vmin = np.min(x)
        vdiff = max(0, np.abs(vmax - vmin))
        if np.abs(vdiff) < epsilon:
            x = np.zeros_like(x)
        else:
            x = np.clip((x - vmin) / (vmax - vmin), 0, 1)
        x = x.reshape(height, width)
    return x


def agg_clip(x, top_clip=True):
    return clip(agg_default(x), top_clip=top_clip)
def simplegrad(net, x, label):
    x_stack = Variable(x, requires_grad=True).cuda()
    pred = net(x_stack)[0, label]
    x_grad = torch.autograd.grad(pred, x_stack, create_graph=False)[0].detach().cpu()
    return x_grad