import abc
import numpy as np
import torch as th
import torch.nn as nn
import typing as T

from deep_fve import utils

class WeightedResult(T.NamedTuple):
    result: th.Tensor
    weight: T.Optional[th.Tensor] = None


class BaseEncodingLayer(nn.Module, abc.ABC):
    _LOG_2PI = np.log(2 * np.pi)

    def __init__(self, in_size, n_components, *,
        init_mu=None,
        init_sig=1,
        eps=1e-2,
        dtype=th.float32,
        **kwargs):
        super().__init__()


        self.n_components = n_components
        self.in_size = in_size
        self.eps = th.scalar_tensor(eps)

        self.add_params(dtype)
        self.init_params(init_mu, init_sig)


    def add_params(self, dtype):
        self.register_buffer("mu", th.zeros((self.in_size, self.n_components), dtype=dtype))
        self.register_buffer("sig", th.ones((self.in_size, self.n_components), dtype=dtype))
        self.register_buffer("w", th.ones((self.n_components), dtype=dtype) / self.n_components)

    def init_params(self, init_mu: float = 0.0, init_sig: float = 1.0):

        if init_mu is not None:
            if isinstance(init_mu, np.ndarray):
                self.mu[:] = th.as_tensor(init_mu)
            elif isinstance(init_mu, (int, float)):
                nn.init.constant_(self.mu, init_mu)
            else:
                raise ValueError(
                    "\"init_mu\" should be either omited, be an instance of " + \
                    f"a numpy array or a numeric value (type was: {type(init_mu)})!"
                )

        if init_sig is not None:
            if isinstance(init_sig, np.ndarray):
                self.sig[:] = th.as_tensor(init_sig)
            elif isinstance(init_sig, (int, float)):
                nn.init.constant_(self.sig, init_sig)
            else:
                raise ValueError(
                    "\"init_sig\" should be either omited, be an instance of " + \
                    f"a numpy array or a numeric value (type was: {type(init_sig)})!"
                )


    def _check_input(self, x: th.Tensor):
        assert x.ndim == 3, \
            "input should have following dimensions: (batch_size, n_features, feature_size)"
        n, t, in_size = x.shape
        assert in_size == self.in_size, \
            "feature size of the input does not match input size: ({} != {})! ".format(
                in_size, self.in_size)
        return n, t


    def _expand_params(self, x: th.Tensor):
        n, t = self._check_input(x)
        shape = (n, t, self.in_size, self.n_components)
        shape2 = (n, t, self.n_components)

        _x = th.broadcast_to(th.unsqueeze(x, -1), shape)

        _params = [(self.mu, shape), (self.sig, shape), (self.w, shape2)]
        _ps = []
        for p, s in _params:
            _p = th.unsqueeze(th.unsqueeze(p, 0), 0)
            # print(p.shape, _p.shape, s, sep=" -> ")
            _ps.append(th.broadcast_to(_p, s))
        _mu, _sig, _w = _ps
        return _x, _mu, _sig, _w

    def soft_assignment(self, x: th.Tensor) -> th.Tensor:
        log_soft_asgn = self.log_soft_assignment(x)
        return log_soft_asgn.exp()

    def log_soft_assignment(self, x: th.Tensor) -> th.Tensor:

        _log_proba, _w = self.log_proba(x, weighted=False)
        _log_wu = _log_proba + _w.log()

        _log_wu_sum = th.logsumexp(_log_wu, axis=-1)
        _log_wu_sum = _log_wu_sum.unsqueeze(dim=-1)
        _log_wu_sum = th.broadcast_to(_log_wu_sum, _log_wu.shape)

        return _log_wu - _log_wu_sum

    def _dist(self, x) -> WeightedResult:
        """
            computes squared Mahalanobis distance
            (in our case it is the standartized Euclidean distance)
        """
        _x, _mu, _sig, _w = self._expand_params(x)

        dist = th.sum((_x - _mu)**2 / _sig, dim=2)

        return WeightedResult(dist, _w)

    def mahalanobis_dist(self, x) -> th.Tensor:
        return self._dist(x).result.sqrt()

    def _log_proba_intern(self, x) -> WeightedResult:
        _dist, _w = self._dist(x)

        # normalize with (2*pi)^k and det(sig) = prod(diagonal_sig)
        log_det = th.sum(self.sig.log(), dim=0)
        _log_proba = -0.5 * (self.in_size * self._LOG_2PI + _dist + log_det)

        return WeightedResult(_log_proba, _w)

    def log_proba(self, x, *args, weighted=False, **kwargs) -> WeightedResult:

        _log_proba, _w = self._log_proba_intern(x, *args, **kwargs)

        if weighted:
            _log_wu = _log_proba + _w.log()
            _log_proba = th.logsumexp(_log_wu, dim=-1)

        return WeightedResult(_log_proba, _w)

    def proba(self, x, *args, **kwargs):
        _log_proba, _w = self.log_proba(x, *args, **kwargs)
        return _log_proba.exp(), _w

    def get_mask(self, x, use_mask, visibility_mask=None):
        if not use_mask:
            return Ellipsis
        _feats = x.detach()#utils.asarray(x)
        _feat_lens = th.sum(_feats**2, axis=2).sqrt()


        if visibility_mask is None:
            _mean_feat_lens = _feat_lens.mean(axis=1, keepdims=True)
            selected = _feat_lens >= _mean_feat_lens

        else:

            if 0 in visibility_mask.sum(axis=1):
                raise RuntimeError("Selection mask contains not selected samples!")

            _mean_feat_lens = (_feat_lens * visibility_mask).sum(axis=1, keepdims=True)
            _n_visible_feats = visibility_mask.sum(axis=1, keepdims=True)#.astype(chainer.config.dtype)
            _mean_feat_lens /= _n_visible_feats

            selected = _feat_lens >= _mean_feat_lens
            selected = th.logical_and(selected, visibility_mask)

        return th.where(selected)
