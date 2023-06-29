import abc
import numpy as np
import torch as th

from dataclasses import dataclass
from dataclasses import asdict
from deep_fve import mixtures
from deep_fve import utils
from deep_fve.modules import base

@dataclass
class SKLearnArgs:
    max_iter: int = 1
    tol: float = np.inf
    reg_covar: float = 1e-2

class GMMMixin(abc.ABC):

    def __init__(self, *args, gmm_cls=mixtures.GMM, sk_learn_kwargs: SKLearnArgs = SKLearnArgs(), **kwargs):

        super().__init__(*args, **kwargs)

        self.gmm_cls = gmm_cls
        sk_learn_kwargs.reg_covar = getattr(sk_learn_kwargs, "reg_covar", self.eps)
        self.sk_learn_kwargs = sk_learn_kwargs
        self.sk_gmm = None

    @abc.abstractmethod
    def set_gmm_params(self, gmm):
        pass

    def new_gmm(self, gmm_cls=None, *args, **kwargs):
        return (gmm_cls or self.gmm_cls)(
            covariance_type="diag",
            n_components=self.n_components,
            **kwargs
        )

    def as_sklearn_gmm(self, gmm_cls=None, **gmm_kwargs):
        gmm = self.new_gmm(gmm_cls=gmm_cls, warm_start=True, **gmm_kwargs)
        self.set_gmm_params(gmm)
        return gmm

    def sample(self, n_samples):
        gmm = self.as_sklearn_gmm(**asdict(self.sk_learn_kwargs))
        return gmm.sample(n_samples)

    def precisions_chol(self) -> th.Tensor:
        """
            Compute the Cholesky decomposition of the precisions.
            Reference:
                https://github.com/scikit-learn/scikit-learn/blob/0.21.3/sklearn/mixture/gaussian_mixture.py#L288
        """
        return 1. / self.sig.sqrt()


class GMMLayer(GMMMixin, base.BaseEncodingLayer):

    def __init__(self, in_size, n_components, *,
        init_from_data=False,
        alpha=0.99,
        **kwargs):
        super().__init__(in_size, n_components, **kwargs)

        self.register_buffer("alpha", th.scalar_tensor(alpha))
        self.register_buffer("t", th.scalar_tensor(1))

        self._initialized = not init_from_data


    def init_from_data(self, x, gmm_cls=None):

        data = utils.asarray(x)
        gmm = self.new_gmm(gmm_cls=gmm_cls)
        self.set_gmm_params(gmm)

        gmm.fit(data.reshape(-1, data.shape[-1]))

        self.mu[:]  = th.as_tensor(gmm.means_.T)
        self.sig[:] = th.as_tensor(gmm.covariances_.T)
        self.w[:]   = th.as_tensor(gmm.weights_)

        self._initialized = True

    def set_gmm_params(self, gmm) -> None:
        means_, covariances_, prec_chol_, weights_ = \
            [self.mu.T, self.sig.T, self.precisions_chol().T, self.w]

        gmm.precisions_cholesky_ = utils.asarray(prec_chol_)
        gmm.covariances_ = utils.asarray(covariances_)
        gmm.means_= utils.asarray(means_)
        gmm.weights_= utils.asarray(weights_)

    def forward(self, x, use_mask=False, visibility_mask=None):
        if self.training:
            mask = self.get_mask(x,
                                 use_mask=use_mask,
                                 visibility_mask=visibility_mask)
            selected = x[mask]
            selected = selected.reshape(-1, x.shape[-1])
            self.update_parameter(selected)
        return x

    def _log_proba_intern(self, x, use_sk_learn=False):

        if not use_sk_learn:
            return super()._log_proba_intern(x)

        _x, _mu, _sig, _w = self._expand_params(x)
        _dist = self._sk_learn_dist(x)
        prec_chol_ = self.precisions_chol()
        # det(precision_chol) is half of det(precision)
        log_det_chol = th.sum(prec_chol_.log(), 0)
        _log_proba = -0.5 * (self.in_size * self._LOG_2PI + _dist) + log_det_chol

        return _log_proba, _w

    def get_new_params(self, x):
        if self.sk_gmm is None:
            # self.sk_gmm = self.as_sklearn_gmm(**self.sk_learn_kwargs)
            self.sk_gmm = self.new_gmm(**asdict(self.sk_learn_kwargs))

        self.sk_gmm.fit(x)

        new_mu, new_sig, new_w = map(th.as_tensor,
            [self.sk_gmm.means_.T, self.sk_gmm.covariances_.T, self.sk_gmm.weights_.T])

        return new_mu, new_sig, new_w

    def update_parameter(self, x):
        if not self._initialized:
            self.init_from_data(x)

        if self.alpha >= 1:
            return #pragma: no cover

        new_mu, new_sig, new_w = self.get_new_params(x)

        self.w[:] = utils.ema(self.w, new_w.to(self.w.device), alpha=self.alpha, t=self.t)
        self.mu[:]  = utils.ema(self.mu, new_mu.to(self.mu.device), alpha=self.alpha, t=self.t)
        self.sig[:] = utils.ema(self.sig, new_sig.to(self.sig.device), alpha=self.alpha, t=self.t)
        self.t += 1

        self.sig = th.maximum(self.sig, self.eps)

    def _sk_learn_dist(self, x):
        """
            Estimate the log Gaussian probability
            Reference:
                https://github.com/scikit-learn/scikit-learn/blob/0.21.3/sklearn/mixture/gaussian_mixture.py#L380
        """
        n, t, size = x.shape
        _x = x.reshape(-1, size)#.detach().numpy()
        _mu = self.mu.T
        _precs = 1 / self.sig.T

        res0 = th.sum((_mu ** 2 * _precs), 1)
        res1 = -2. * (_x @ (_mu * _precs).T)
        res2 = (_x ** 2 @ _precs.T)
        res = th.broadcast_to(res0, res1.shape) + res1 + res2
        return res.reshape(n, t, -1)
