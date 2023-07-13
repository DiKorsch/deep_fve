import cupy as cp
import numpy as np
import time
import torch as th
import unittest

from functools import partial
from scipy.stats import multivariate_normal as mvn
from sklearn.mixture import GaussianMixture as GMM

from deep_fve import GMMLayer
from deep_fve import utils
from deep_fve.mixtures import GMM as gpuGMM
from deep_fve.mixtures.base import _basic_e_step
from deep_fve.mixtures.base import _kernel_e_step
from deep_fve.modules.gmm import SKLearnArgs

from tests import base

class GMMMixtureTest(unittest.TestCase):

    def setUp(self):
        T, N_COMP, SIZE = 32, 4, 256
        dtype, xp = cp.float32, cp

        self.atol = 1e-3
        self.rtol = 1e-2

        self.X = xp.random.randn(T, SIZE).astype(dtype)
        self.mu = xp.random.rand(N_COMP, SIZE).astype(dtype)
        self.sig = xp.random.rand(N_COMP, SIZE).astype(dtype) + 1
        self.ws = xp.ones(N_COMP).astype(dtype) / N_COMP
        self.xp = xp

    def assertClose(self, arr0, arr1, msg):
        self.assertTrue(np.allclose(utils.asarray(arr0), utils.asarray(arr1), rtol=self.rtol, atol=self.atol),
            f"{msg}:\nMSE: {np.mean((arr0-arr1)**2)}")

    def test_e_step(self):
        logL0, log_gammas0 = self.bench(_kernel_e_step, 10000, 200)
        logL1, log_gammas1 = self.bench(_basic_e_step, 10000, 200)

        self.assertClose(log_gammas0, log_gammas1, msg="Were not equal")
        self.assertClose(logL0, logL1, msg="Were not equal")

    def bench(self, func, n_iter: int, warm_up: int):

        for _ in range(warm_up):
            res = func(self.X, self.mu, self.sig, self.ws, xp=self.xp)

        t0 = time.time()

        for _ in range(n_iter):
            res = func(self.X, self.mu, self.sig, self.ws, xp=self.xp)

        t0 = time.time() - t0
        func_name = func.func.__name__ if isinstance(func, partial) else func.__name__
        print(f"{func_name} took {t0:.3f}s for {n_iter:,d} runs")
        return res

    def check_params(self, gmm0, gmm1, prefix=""):

        params0 = gmm0._get_parameters()
        params1 = gmm1._get_parameters()
        names = ["w", "mu", "sig", "prec"]

        for name, p0, p1 in zip(names, params0, params1):
            self.assertClose(p0, p1,
                f"{prefix}Parameters {name} did not match: {p0} != {p1}")

    def test_fit(self):
        n_components = self.mu.shape[0]
        seed = np.random.randint(2**32 - 1)

        kwargs = SKLearnArgs(max_iter=10, tol=1e-3).asdict()
        kwargs["warm_start"] = True
        x = self.X.get()

        rnd = np.random.RandomState(seed)
        gmm = GMM(n_components, covariance_type="diag", random_state=rnd, **kwargs)
        gmm._initialize_parameters(x, random_state=rnd)

        # we need this for warm start!
        gmm.converged_, gmm.lower_bound_ = False, -np.inf


        rnd = np.random.RandomState(seed)
        gpu_gmm = gpuGMM(n_components, covariance_type="diag", random_state=rnd, **kwargs)
        gpu_gmm._initialize_parameters(x, random_state=rnd)

        self.check_params(gmm, gpu_gmm, prefix="[Init] ")

        gmm.fit(x)
        gpu_gmm.fit(x)

        self.check_params(gmm, gpu_gmm, prefix="[Fit] ")




    def test_steps(self):
        n_components = self.mu.shape[0]
        kwargs = SKLearnArgs(max_iter=100, tol=1e-3).asdict()
        x = self.X.get()

        gmm = GMM(n_components, covariance_type="diag", **kwargs)
        gmm.fit(x)

        gpu_gmm = gpuGMM(n_components, covariance_type="diag", **kwargs)
        gpu_gmm._set_parameters(gmm._get_parameters())

        self.check_params(gmm, gpu_gmm)

        result0 = gmm._e_step(x)
        result1 = gpu_gmm._e_step(x, use_kernel=False)

        for i, (res0, res1) in enumerate(zip(result0, result1)):
            self.assertClose(res0, res1, f"E-Step Result #{i} was incorrect: {res0} != {res1}")

        log_prob_norm, log_resp = result0

        gmm._m_step(x, log_resp)
        gpu_gmm._m_step(x, log_resp)
        self.check_params(gmm, gpu_gmm, prefix="[M-step] ")

class GMMLayerTest(base.BaseFVEncodingTest):

    def _new_layer(self, *args, **kwargs):
        return super(GMMLayerTest, self)._new_layer(layer_cls=GMMLayer, *args, **kwargs)


    def test_initialization(self):
        layer = self._new_layer(init_mu=10)
        self.assertTrue(layer.mu.min() >= -10)
        self.assertTrue(layer.mu.max() <=  10)

        self.init_mu = None
        layer = self._new_layer()
        self.assertTrue(layer.mu.min() >= -1)
        self.assertTrue(layer.mu.max() <=  1)


        with self.assertRaises(ValueError):
            self.init_mu = "None"
            self._new_layer()

    def test_masks(self):

        layer = self._new_layer()
        layer(self.X, use_mask=True)


        layer = self._new_layer()
        vis_mask = th.ones(self.X.shape[:-1], dtype=bool)
        _, *size, _ = self.X.shape
        idxs = np.random.randint(np.multiply.reduce(size), size=(self.n,))
        idxs = np.unravel_index(idxs, size)

        if len(size) == 2:
            vis_mask[np.arange(self.n), idxs[0], idxs[1]] = 0

        elif len(size) == 1:
            vis_mask[np.arange(self.n), idxs] = 0

        layer(self.X, use_mask=True, visibility_mask=vis_mask)

        with self.assertRaises(RuntimeError):
            layer = self._new_layer()
            vis_mask = th.zeros(self.X.shape[:-1], dtype=bool)
            layer(self.X, use_mask=True, visibility_mask=vis_mask)

    def test_init_from_data(self):

        layer = self._new_layer(init_from_data=True)
        layer(self.X)

        layer = self._new_layer(init_from_data=True)
        layer(utils.asarray(self.X))

    def test_sklearn_dist(self):
        layer = self._new_layer()
        res0, _ = layer.log_proba(self.X, use_sk_learn=False)
        res1, _ = layer.log_proba(self.X, use_sk_learn=True)

        self.assertClose(res0, res1,
            "sklearn results in a different result!")

    def test_gpu(self):
        layer = self._new_layer()
        layer.to("cuda:0")
        self.X.to("cuda:0")

        layer(self.X)

    def test_output(self):
        layer = self._new_layer()

        res = layer(self.X)

        self.assertIs(res, self.X,
            "Output should be the input!")

    def test_update(self):
        layer = self._new_layer()
        names = ["mu", "sig", "w"]
        params = (layer.mu, layer.sig, layer.w)

        params0 = [np.copy(p) for p in params]
        layer.eval()
        layer(self.X)
        params1 = [np.copy(p) for p in params]

        for name, p0, p1 in zip(names, params0, params1):
            self.assertTrue(np.all(p0 == p1),
                f"Param {name} should not be updated when not training!")


        params0 = [np.copy(p) for p in params]

        layer.train()
        layer(self.X)

        params1 = [np.copy(p) for p in params]

        for name, p0, p1 in zip(names, params0, params1):
            self.assertTrue(np.all(p0 != p1),
                f"Param {name} should be updated when training!")

    def test_assignment_shape(self):
        layer = self._new_layer()

        gamma = layer.soft_assignment(self.X)
        correct_shape = (self.n, self.t, self.n_components)
        self.assertEqual(gamma.shape, correct_shape,
            "Shape of the soft assignment is not correct!")

    def test_assignment_sum(self):
        layer = self._new_layer()

        gamma = layer.soft_assignment(self.X)
        gamma_sum = gamma.sum(axis=-1)
        self.assertClose(gamma_sum, 1,
            "Sum of the soft assignment should be always equal to 1, but was")

    def test_assignment(self):
        layer = self._new_layer()
        correct_shape = (self.n, self.t, self.n_components)
        gamma = layer.soft_assignment(self.X)

        gmm = layer.as_sklearn_gmm()
        x = self.X.reshape(-1, self.in_size)
        ref_gamma = gmm.predict_proba(x.detach().numpy()).reshape(correct_shape)

        self.assertClose(gamma, ref_gamma,
            "Soft assignment is not similar to reference value from sklearn")


    def test_log_assignment(self):
        layer = self._new_layer()
        correct_shape = (self.n, self.t, self.n_components)
        log_gamma = layer.log_soft_assignment(self.X)

        gmm = layer.as_sklearn_gmm()
        x = self.X.reshape(-1, self.in_size)
        _, log_ref_gamma = gmm._estimate_log_prob_resp(x.detach().numpy())
        log_ref_gamma = log_ref_gamma.reshape(correct_shape)

        self.assertClose(log_gamma, log_ref_gamma,
            "Log soft assignment is not similar to reference value from sklearn")

    def test_log_proba(self):

        layer = self._new_layer()

        mean, var, _ = (layer.mu, layer.sig, layer.w)
        mean, var = mean.T, var.T

        log_ps2, _ = layer.log_proba(self.X, weighted=False)
        ps2, _ = layer.proba(self.X, weighted=False)


        for i, x in enumerate(self.X.reshape(-1, self.in_size)):
            n, t = np.unravel_index(i, (self.n, self.t))

            for component in range(self.n_components):
                log_p1 = mvn.logpdf(x, mean[component], var[component]).astype(self.dtype)
                log_p2 = log_ps2[n, t, component]

                self.assertClose(log_p1, log_p2,
                    f"{[n,t,component]}: Log-Likelihood was not the same")

                p1 = mvn.pdf(x, mean[component], var[component]).astype(self.dtype)
                p2 = ps2[n, t, component]

                self.assertClose(p1, p2,
                    f"{[n,t,component]}: Likelihood was not the same")
