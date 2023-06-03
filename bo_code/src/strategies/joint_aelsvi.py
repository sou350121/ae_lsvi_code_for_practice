"""
Improve optimization strategies for joint GPs.
"""
from argparse import Namespace
import numpy as np
from scipy.stats import norm as normal_distro

from strategies.joint_opt import JointOpt
from util.misc_util import sample_grid, build_gp_posterior

from dragonfly.opt.gpb_acquisitions import _get_ucb_beta_th, _get_gp_ucb_dim


class JointAELSVI(JointOpt):

    def decide_next_query(self):
        """Given a particular gp, decide best point to query.
        Returns: Function index and point to query next.
        """
        build_gp_posterior(self.gps[0])
        return self._find_best_query()

    @staticmethod
    def get_opt_method_name():
        """Get the name of the method."""
        return 'joint-aelsvi'

    def _find_best_query(self):
        """Draw a joint sample over the entire space, use this to select
        the context and the action.
        Returns: Index of best context/task and the best point.
        """
        gp = self.gps[0]
        max_acq_values = []
        max_acq_points = []
        for idx in range(len(self.fcns)):
            f_idx = idx
            grid_pts = sample_grid([self.f_locs[f_idx]], self.domains[0],
                                   self.options.max_opt_evals)
            mu, sigma = gp.eval(grid_pts, include_covar=True)
            sigma = np.sqrt(sigma.diagonal().ravel())
            adim = self.gp_options.act_dim
            beta = 0.2 * adim * np.log(2 * adim * self.t + 1)
            # beta = np.sqrt(0.2 * adim * np.log(2 * adim * self.t + 1))
            ucbs = mu + beta * sigma
            lcbs = mu - beta * sigma
            acq = np.max(ucbs) - np.max(lcbs)
            pt_idx = np.argmax(ucbs)
            acq_pt = grid_pts[pt_idx, len(self.f_locs[0]):]
            max_acq_values.append(acq)
            max_acq_points.append(acq_pt)
        query_idx = np.argmax(max_acq_values)
        best_pt = max_acq_points[query_idx]
        return query_idx, best_pt
