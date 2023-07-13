import numpy as np

from sklearn.mixture import GaussianMixture

from deep_fve.mixtures.base import GPUMixin

class GMM(GPUMixin, GaussianMixture):


    def _m_step(self, X, log_resp, xp=np):
        """ M-Step
            Copied from sklearn/mixture/gaussian_mixture.py
        """

        nk, self.means_, self.covariances_ = \
            self._gaussian_params(X, log_resp, xp=xp)


        self.weights_ = nk / nk.sum() #X.shape[0]
        self.precisions_cholesky_ = 1 / xp.sqrt(self.covariances_)
