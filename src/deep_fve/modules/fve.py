import abc
import torch as th
import numpy as np

from deep_fve import utils
from deep_fve.modules.base import BaseEncodingLayer
from deep_fve.modules.gmm import GMMLayer
from deep_fve.modules.gmm import GMMMixin

def _mask(tensor: th.Tensor, eps: float, *, op = th.maximum) -> th.Tensor:
    return op(tensor, th.scalar_tensor(eps))

class BaseFVELayer(abc.ABC):

    def encode(self, x, use_mask=False, visibility_mask=None, eps=1e-6):
        gamma = self.soft_assignment(x)
        _x, *params = self._expand_params(x)

        _gamma = th.unsqueeze(gamma, axis=2)
        _mu, _sig, _w = [p for p in params]

        """
            If the GMM component is degenerate and has a null prior, then it
            must have null posterior as well. Hence it is safe to skip it.
            In practice, we skip over it even if the prior is very small;
            if by any chance a feature is assigned to such a mode, then
            its weight would be very high due to the division by
            priors[i_cl] below.

        """
        # mask out all gammas, that are < eps
        _gamma = _mask(_gamma, eps)

        ### Here the computations begin
        _std = _sig.sqrt()
        _x_mu_sig = (_x - _mu) / _std

        if use_mask:
            mask = self.get_mask(x, use_mask, visibility_mask)
            selected = th.zeros_like(_x_mu_sig)
            selected[mask] = 1
        else:
            selected = th.ones_like(_x_mu_sig)

        G_mu = _gamma * _x_mu_sig * selected
        G_sig = _gamma * (_x_mu_sig**2 - 1) * selected

        """
            Here we are not so sure about the normalization factor.
            In [1] the factor is 1 / sqrt(T) (Eqs. 10, 11, 13, 4),
            but in [2] the factor is 1 / T (Eqs. 7, 8).

            Actually, this has no effect on the final classification,
            but is still a mismatch in the literature.

            In this code, we stick to the version of [2], since it
            fits the results computed by cyvlfeat.

            ---------------------------------------------------------------------
            [1] - Fisher Kernels on Visual Vocabularies for Image Categorization
            (https://ieeexplore.ieee.org/document/4270291)
            [2] - Improving the Fisher Kernel for Large-Scale Image Classification
            (https://link.springer.com/chapter/10.1007/978-3-642-15561-1_11)
        """
        # Version 1:
        # G_mu = F.sum(G_mu, axis=1) / xp.sqrt(selected.sum(axis=1))
        # G_sig = F.sum(G_sig, axis=1) / xp.sqrt(selected.sum(axis=1))
        # Version 2:
        G_mu = G_mu.sum(dim=1) / selected.sum(axis=1)
        G_sig = G_sig.sum(dim=1) / selected.sum(axis=1)

        _w = th.broadcast_to(self.w, G_mu.shape)
        # mask out all weights, that are < eps
        _w = _mask(_w, eps)

        G_mu /= _w.sqrt()
        G_sig /= (2 * _w).sqrt()

        # 2 * (n, in_size, n_components) -> (n, 2, in_size, n_components)
        res = th.stack([G_mu, G_sig], axis=1)
        # (n, 2, in_size, n_components) -> (n, 2, n_components, in_size)
        res = res.permute(0, 1, 3, 2)
        # res = th.stack([G_mu], axis=-1)
        # (n, 2, n_components, in_size) -> (n, 2*in_size*n_components)
        res = th.reshape(res, (x.shape[0], -1))
        return res

class FVELayer(BaseFVELayer, GMMLayer):

    def forward(self, x, use_mask=False, visibility_mask=None):
        _ = super().forward(x, use_mask, visibility_mask)
        return self.encode(x, use_mask, visibility_mask)

class FVELayer_noEM(BaseFVELayer, GMMMixin, BaseEncodingLayer):

    def init_params(self, init_mu, init_sig):
        if init_mu is not None:
            if isinstance(init_mu, np.ndarray):
                assert init_mu.shape == self.mu.shape, \
                    f"shapes do not fit: {init_mu.shape} != {self.mu.shape}"
                self.mu = th.nn.Parameter(th.as_tensor(init_mu))

            elif isinstance(init_mu, (int, float)):
                th.nn.init.constant_(self.mu, init_mu)

            else:
                raise ValueError(
                    "\"init_mu\" should be either omited, be an instance of " + \
                    f"a numpy array or a numeric value (type was: {type(init_mu)})!"
                )

        if init_sig is not None:
            if isinstance(init_sig, np.ndarray):
                assert init_sig.shape == self._sig.shape, \
                    f"shapes do not fit: {init_sig.shape} != {self._sig.shape}"
                self._sig = th.nn.Parameter((th.as_tensor(init_sig) - self.eps).log())

            elif isinstance(init_sig, (int, float)):
                init_sig = np.log(init_sig - float(self.eps))
                th.nn.init.constant_(self._sig, init_sig)

            else:
                raise ValueError(
                    "\"init_sig\" should be either omited, be an instance of " + \
                    f"a numpy array or a numeric value (type was: {type(init_sig)})!"
                )


    def add_params(self, dtype):
        self.mu = th.nn.Parameter(
            th.zeros((self.in_size, self.n_components), dtype=dtype))
        self._sig = th.nn.Parameter(
            th.ones((self.in_size, self.n_components), dtype=dtype))
        self._w = th.nn.Parameter(
            th.ones(self.n_components, dtype=dtype) / self.n_components)

    @property
    def w(self):
        w_sigmoid = self._w.sigmoid()
        res =  w_sigmoid / w_sigmoid.sum()
        return res

    @w.setter
    def w(self, param):
        self._w = param

    @property
    def sig(self):
        return self._sig.exp() + self.eps

    def set_gmm_params(self, gmm):
        gmm.precisions_cholesky_ = utils.asarray(self.precisions_chol).T
        gmm.covariances_ = utils.asarray(self.sig).T
        gmm.means_= utils.asarray(self.mu).T
        gmm.weights_= utils.asarray(self.w)

    @property
    def precisions_chol(self):
        return 1. / self.sig.sqrt()

    def forward(self, x, use_mask=False, visibility_mask=None):
        return self.encode(x, use_mask, visibility_mask)

