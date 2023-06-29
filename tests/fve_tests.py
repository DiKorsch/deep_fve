import abc
import numpy as np
import torch as th

from cyvlfeat.fisher import fisher
from cyvlfeat.gmm import cygmm

from deep_fve import FVELayer
from deep_fve import FVELayer_noEM
from deep_fve import utils

from tests import base

class BaseFVELayerTest(base.BaseFVEncodingTest):

    @abc.abstractmethod
    def _new_layer(self, *args, **kwargs):
        return super(BaseFVELayerTest, self)._new_layer(*args, **kwargs)

    def test_output_shape(self):
        layer = self._new_layer(train=False)
        output = layer(self.X)

        output_shape = (self.n, 2 * self.n_components * self.in_size)
        self.assertEqual(output.shape, output_shape,
            "Output shape was not correct!")

    def test_output(self):
        layer = self._new_layer(train=False)

        mean, var, w = map(utils.asarray, [layer.mu, layer.sig, layer.w])

        output = utils.asarray(layer(self.X))

        x = utils.asarray(self.X).astype(np.float32)

        # we need to convert the array in Fortran order, but still remain the dimensions,
        # since the python API awaits <dimensions>x<components> arrays,
        # but they are indexed with "i_cl*dimension + dim" in the C-Code

        params = (
            mean.T.copy(),
            var.T.copy(),
            w.copy(),
        )
        ref = [fisher(_x, *params,
                    normalized=False,
                    square_root=False,
                    improved=False,
                    fast=False,
                    verbose=False,
            ) for _x in x]

        ref = np.stack(ref)

        self.assertEqual(output.shape, ref.shape,
            "Output shape was not equal to ref shape!")

        self.assertClose(output, ref,
            "Output was not similar to reference")


    def test_cygmm(self):
        layer = self._new_layer()
        mean, var, w = map(utils.asarray, [layer.mu, layer.sig, layer.w])

        gamma = layer.soft_assignment(self.X)
        log_proba, _ = layer.log_proba(self.X, weighted=True)
        log_proba = utils.asarray(log_proba)

        for i, _x in enumerate(self.X):
            cy_mean, cy_var, cy_w, LL, cy_gamma = cygmm.cy_gmm(
                utils.asarray(_x),
                self.n_components,
                0, # max_iterations
                "custom".encode("utf8"), # init_mode
                1, # n_repitions
                0, # verbose
                covariance_bound=None,

                init_means=mean.T.copy(),
                init_covars=var.T.copy(),
                init_priors=w.copy(),
            )

            self.assertClose(gamma[i], cy_gamma,
                f"[{i}] Soft assignment was not similar to reference (vlfeat)")

            self.assertClose(float(log_proba[i].sum()), LL,
                f"[{i}] Log-likelihood was not similar to reference (vlfeat)")


    def test_gap_init(self):
        self.n_components = 1
        layer = self._new_layer(init_mu=0, init_sig=1, train=False)
        _x = utils.asarray(self.X)
        gap_output = self.xp.mean(_x, axis=1)
        ref_sig_output = self.xp.mean(_x**2 - 1, axis=1) / self.xp.sqrt(2)

        output = layer(self.X)

        mu_output, sig_output = output[:, :self.in_size], output[:, self.in_size:]

        self.assertClose(mu_output, gap_output,
            "mu-Part of FVE should be equal to GAP!")

        self.assertClose(sig_output, ref_sig_output,
            "sig-Part of FVE should be equal to reference!")


class FVELayerTest(BaseFVELayerTest):

    def _new_layer(self, *args, **kwargs):
        return super(FVELayerTest, self)._new_layer(layer_cls=FVELayer, *args, **kwargs)

class FVELayer_noEMTest(BaseFVELayerTest):

    def _new_layer(self, *args, **kwargs):
        return super(FVELayer_noEMTest, self)._new_layer(layer_cls=FVELayer_noEM, *args, **kwargs)


    def test_gradients(self):

        layer = self._new_layer(train=True)

        for param in layer.parameters():
            self.assertIsNone(param.grad)


        output = layer(self.X)

        # from chainer.computational_graph import build_computational_graph as bcg
        # import graphviz
        # g = bcg([output])
        # graphviz.Source(g.dump()).render(view=True)

        output.sum().backward()

        for param in layer.parameters():
            self.assertIsNotNone(param.grad)
