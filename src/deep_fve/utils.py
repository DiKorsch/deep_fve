import cupy as cp
import torch as th

def asarray(var):
    if isinstance(var, th.Tensor):
        if var.device.type == "cpu":
            return var.detach().numpy()

        with cp.cuda.Device(var.device.index):
            return cp.asarray(var.detach().cpu().float())

    return var

def ema(old, new, *, alpha: float, t: int):
    prev_correction = 1 - (alpha ** (t-1))
    correction = 1 - (alpha ** t)

    uncorrected_old = old * prev_correction
    res = alpha * uncorrected_old + (1 - alpha) * new

    return res / correction


if __name__ == '__main__':

    th_a = th.Tensor((1,2,3)) +1
    np_a = asarray(th_a)
    cp_a0 = asarray(th_a.to("cuda:0"))
    cp_a1 = asarray(th_a.to("cuda:1"))

    for a in [th_a, np_a, cp_a0, cp_a1]:
        print(a, getattr(a, "device", None), type(a))
