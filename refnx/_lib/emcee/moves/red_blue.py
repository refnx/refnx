# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

from .move import Move

__all__ = ["RedBlueMove"]


class RedBlueMove(Move):
    """
    An abstract red-blue ensemble move with parallelization as described in
    `Foreman-Mackey et al. (2013) <http://arxiv.org/abs/1202.3665>`_.

    Args:
        nsplits (Optional[int]): The number of sub-ensembles to use. Each
            sub-ensemble is updated in parallel using the other sets as the
            complementary ensemble. The default value is ``2`` and you
            probably won't need to change that.

        randomize_split (Optional[bool]): Randomly shuffle walkers between
            sub-ensembles. The same number of walkers will be assigned to
            each sub-ensemble on each iteration. By default, this is ``False``.

        live_dangerously (Optional[bool]): By default, an update will fail with
            a ``RuntimeError`` if the number of walkers is smaller than twice
            the dimension of the problem because the walkers would then be
            stuck on a low dimensional subspace. This can be avoided by
            switching between the stretch move and, for example, a
            Metropolis-Hastings step. If you want to do this and suppress the
            error, set ``live_dangerously = True``. Thanks goes (once again)
            to @dstndstn for this wonderful terminology.

    """
    def __init__(self,
                 nsplits=2,
                 randomize_split=True,
                 live_dangerously=False):
        self.nsplits = int(nsplits)
        self.live_dangerously = live_dangerously
        self.randomize_split = randomize_split

    def setup(self, coords):
        pass

    def get_proposal(self, sample, complement, random):
        raise NotImplementedError("The proposal must be implemented by "
                                  "subclasses")

    def propose(self, coords, log_probs, blobs, log_prob_fn, random):
        """Use the move to generate a proposal and compute the acceptance

        Args:
            coords: The initial coordinates of the walkers.
            log_probs: The initial log probabilities of the walkers.
            log_prob_fn: A function that computes the log probabilities for a
                subset of walkers.
            random: A numpy-compatible random number state.

        """
        # Check that the dimensions are compatible.
        nwalkers, ndim = coords.shape
        if nwalkers < 2 * ndim and not self.live_dangerously:
            raise RuntimeError("It is unadvisable to use a red-blue move "
                               "with fewer walkers than twice the number of "
                               "dimensions.")

        # Run any move-specific setup.
        self.setup(coords)

        # Split the ensemble in half and iterate over these two halves.
        accepted = np.zeros(nwalkers, dtype=bool)
        all_inds = np.arange(nwalkers)
        inds = all_inds % self.nsplits
        if self.randomize_split:
            random.shuffle(inds)
        for split in range(self.nsplits):
            S1 = inds == split

            # Get the two halves of the ensemble.
            sets = [coords[inds == j] for j in range(self.nsplits)]
            s = sets[split]
            c = sets[:split] + sets[split + 1:]

            # Get the move-specific proposal.
            q, factors = self.get_proposal(s, c, random)

            # Compute the lnprobs of the proposed position.
            new_log_probs, new_blobs = log_prob_fn(q)

            # Loop over the walkers and update them accordingly.
            for i, (j, f, nlp) in enumerate(zip(
                    all_inds[S1], factors, new_log_probs)):
                lnpdiff = f + nlp - log_probs[j]
                if lnpdiff > np.log(random.rand()):
                    accepted[j] = True

            coords, log_probs, blobs = self.update(
                coords, log_probs, blobs,
                q, new_log_probs, new_blobs,
                accepted, S1)

        return coords, log_probs, blobs, accepted
