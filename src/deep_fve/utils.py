import cupy as cp
import torch as th

def asarray(var):
    if isinstance(var, th.Tensor):
        if var.device.type == "cpu":
            return var.detach().numpy()

        with cp.cuda.Device(var.device.index):
            return cp.asarray(var)

    return var

def ema(old, new, *, alpha: float, t: int):
    prev_correction = 1 - (alpha ** (t-1))
    correction = 1 - (alpha ** t)

    uncorrected_old = old * prev_correction
    res = alpha * uncorrected_old + (1 - alpha) * new

    return res / correction
