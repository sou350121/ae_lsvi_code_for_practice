"""
The AE LSVI strategy.
"""
import numpy as np

from strategies.multi_opt import MultiOpt

from util.misc_util import build_gp_posterior, ucb_acq, lcb_acq


class AELSVI(MultiOpt):

    def decide_next_query(self):
        """Make a query that decides which function and which point.
        Returns: Index of functions selected and point to query.
        """
        for gp in self.gps:
            build_gp_posterior(gp)
        query_ctx, query_pt = self._find_best_eval()
        return query_ctx, query_pt

    def _find_best_eval(self):
        """Given a particular gp, decide best point to query.
        Returns: Point index of best task/context and the point.
        """
        best_idx, best_pt = None, None
        best_score = float('-inf')
        for idx in range(len(self.fcns)):
            gp = self.gps[idx]
            domain = self.domains[idx]
            ucb_pt, ucb_val = ucb_acq(gp, domain, self.options.max_opt_evals, self.t)
            lcb_pt, lcb_val = ucb_acq(gp, domain, self.options.max_opt_evals, self.t)
            score = ucb_val - lcb_val
            if score > best_score:
                best_score = score
                best_idx = idx
                best_pt = ucb_pt
        return best_idx, best_pt

    @staticmethod
    def get_opt_method_name():
        return 'aelsvi'
