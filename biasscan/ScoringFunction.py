import numpy as np


class ScoringFunction:
    def __init__(self, **kwargs):
        """
        This is an abstract class for Scoring Functions (or expectation-based scan statistics).

        [1] introduces a property of many commonly used log-likelihood ratio scan statistics called
        Additive linear-time subset scanning (ALTSS) that allows for exact of efficient maximization of these
        statistics over all subsets of the data, without requiring an exhaustive search over all subsets and
        allows penalty terms to be included.

        [1] Speakman, S., Somanchi, S., McFowland III, E., & Neill, D. B. (2016). Penalized fast subset scanning.
        Journal of Computational and Graphical Statistics, 25(2), 382-404.
        """
        self.kwargs = kwargs
        self.direction = kwargs.get('direction')

        directions = ['positive', 'negative']
        assert self.direction in directions, f"Expected one of {directions}, got {self.direction}"

    def score(
            self, observed_sum: float, expectations: np.array, penalty: float, q: float
    ):
        """
        Computes the score for the given q. (for the given records).

        The alternative hypothesis of MDSS assumes that there exists some constant multiplicative factor q > 1
        for the subset of records being scored by the scoring function.
        q is sometimes refered to as relative risk or severity.

        """
        raise NotImplementedError

    def dscore(self, observed_sum: float, expectations: np.array, q: float):
        """
        Computes the first derivative of the score function
        """
        raise NotImplementedError

    def q_dscore(self, observed_sum: float, expectations: np.array, q: float):
        """
        Computes the first derivative of the score function multiplied by the given q
        """
        raise NotImplementedError

    def qmle(self, observed_sum: float, expectations: np.array):
        """
        Computes the q which maximizes score (q_mle).
        """
        raise NotImplementedError

    def compute_qs(self, observed_sum: float, expectations: np.array, penalty: float):
        """
        Computes roots (qmin and qmax) of the score function (for the given records)
        """
        raise NotImplementedError


from aif360.detectors.mdss.ScoringFunctions.ScoringFunction import ScoringFunction
from aif360.detectors.mdss.ScoringFunctions import optim

import numpy as np


class Bernoulli(ScoringFunction):
    def __init__(self, **kwargs):
        """
        Bernoulli score function. May be appropriate to use when the outcome of
        interest is assumed to be Bernoulli distributed or Binary.

        kwargs must contain
        'direction (str)' - direction of the severity; could be higher than expected outcomes ('positive') or lower than expected ('negative')
        """

        super(Bernoulli, self).__init__(**kwargs)

    def score(self, observed_sum: float, expectations: np.array, penalty: float, q: float):
        """
        Computes bernoulli bias score for given q

        :param observed_sum: sum of observed binary outcomes for all i
        :param expectations: predicted outcomes for each data element i
        :param penalty: penalty term. Should be positive
        :param q: current value of q
        :return: bias score for the current value of q
        """

        assert q > 0, (
                "Warning: calling compute_score_given_q with "
                "observed_sum=%.2f, expectations of length=%d, penalty=%.2f, q=%.2f"
                % (observed_sum, len(expectations), penalty, q)
        )

        ans = observed_sum * np.log(q) - np.log(1 - expectations + q * expectations).sum() - penalty
        return ans

    def qmle(self, observed_sum: float, expectations: np.array):
        """
        Computes the q which maximizes score (q_mle).

        :param observed_sum: sum of observed binary outcomes for all i
        :param expectations: predicted outcomes for each data element i
        """
        direction = self.direction
        ans = optim.bisection_q_mle(self, observed_sum, expectations, direction=direction)
        return ans

    def compute_qs(self, observed_sum: float, expectations: np.array, penalty: float):
        """
        Computes roots (qmin and qmax) of the score function for given q

        :param observed_sum: sum of observed binary outcomes for all i
        :param expectations: predicted outcomes for each data element i
        :param penalty: penalty coefficient
        """
        direction = self.direction
        q_mle = self.qmle(observed_sum, expectations)

        if self.score(observed_sum, expectations, penalty, q_mle) > 0:
            exist = 1
            q_min = optim.bisection_q_min(self, observed_sum, expectations, penalty, q_mle)
            q_max = optim.bisection_q_max(self, observed_sum, expectations, penalty, q_mle)
        else:
            # there are no roots
            exist = 0
            q_min = 0
            q_max = 0

        # only consider the desired direction, positive or negative
        if exist:
            exist, q_min, q_max = optim.direction_assertions(direction, q_min, q_max)

        ans = [exist, q_mle, q_min, q_max]
        return ans

    def q_dscore(self, observed_sum: float, expectations: np.array, q: float):
        """
        This actually computes q times the slope, which has the same sign as the slope since q is positive.
        score = Y log q - \sum_i log(1-p_i+qp_i)
        dscore/dq = Y/q - \sum_i (p_i/(1-p_i+qp_i))
        q dscore/dq = Y - \sum_i (qp_i/(1-p_i+qp_i))

        :param observed_sum: sum of observed binary outcomes for all i
        :param expectations: predicted outcomes for each data element i
        :param q: current value of q
        :return: q dscore/dq
        """
        ans = observed_sum - (q * expectations / (1 - expectations + q * expectations)).sum()
        return ans


from aif360.detectors.mdss.ScoringFunctions.ScoringFunction import ScoringFunction
from aif360.detectors.mdss.ScoringFunctions import optim

import numpy as np


class Poisson(ScoringFunction):
    def __init__(self, **kwargs):
        """
        Poisson score function. May be appropriate to use when the outcome of
        interest is assumed to be Poisson distributed or Binary.

        kwargs must contain
        'direction (str)' - direction of the severity; could be higher than expected outcomes ('positive') or lower than expected ('negative')
        """

        super(Poisson, self).__init__(**kwargs)

    def score(self, observed_sum: float, expectations: np.array, penalty: float, q: float):
        """
        Computes poisson bias score for given q

        :param observed_sum: sum of observed binary outcomes for all i
        :param expectations: predicted outcomes for each data element i
        :param penalty: penalty term. Should be positive
        :param q: current value of q
        :return: bias score for the current value of q
        """

        assert q > 0, (
                "Warning: calling compute_score_given_q with "
                "observed_sum=%.2f, expectations of length=%d, penalty=%.2f, q=%.2f"
                % (observed_sum, len(expectations), penalty, q)
        )

        ans = observed_sum * np.log(q) + (expectations - q * expectations).sum() - penalty
        return ans

    def qmle(self, observed_sum: float, expectations: np.array):
        """
        Computes the q which maximizes score (q_mle).
        """
        direction = self.direction
        ans = optim.bisection_q_mle(self, observed_sum, expectations, direction=direction)
        return ans

    def compute_qs(self, observed_sum: float, expectations: np.array, penalty: float):
        """
        Computes roots (qmin and qmax) of the score function for given q

        :param observed_sum: sum of observed binary outcomes for all i
        :param expectations: predicted outcomes for each data element i
        :param penalty: penalty coefficient
        """

        direction = self.direction
        q_mle = self.qmle(observed_sum, expectations)

        if self.score(observed_sum, expectations, penalty, q_mle) > 0:
            exist = 1
            q_min = optim.bisection_q_min(self, observed_sum, expectations, penalty, q_mle)
            q_max = optim.bisection_q_max(self, observed_sum, expectations, penalty, q_mle)
        else:
            # there are no roots
            exist = 0
            q_min = 0
            q_max = 0

        # only consider the desired direction, positive or negative
        if exist:
            exist, q_min, q_max = optim.direction_assertions(direction, q_min, q_max)

        ans = [exist, q_mle, q_min, q_max]
        return ans

    def q_dscore(self, observed_sum, expectations, q):
        """
        This actually computes q times the slope, which has the same sign as the slope since q is positive.
        score = Y log q + \sum_i (p_i - qp_i)
        dscore/dq = Y / q - \sum_i(p_i)
        q dscore/dq = q_dscore = Y - (q * \sum_i(p_i))

        :param observed_sum: sum of observed binary outcomes for all i
        :param expectations: predicted outcomes for each data element i
        :param q: current value of q
        :return: q dscore/dq
        """
        ans = observed_sum - (q * expectations).sum()
        return ans


from aif360.detectors.mdss.ScoringFunctions.ScoringFunction import ScoringFunction
from aif360.detectors.mdss.ScoringFunctions import optim

import numpy as np


class BerkJones(ScoringFunction):
    def __init__(self, **kwargs):
        """
        Berk-Jones score function is a non parametric expectatation based
        scan statistic that also satisfies the ALTSS property; Non-parametric scoring functions
        do not make parametric assumptions about the model or outcome [1].

        kwargs must contain
        'direction (str)' - direction of the severity; could be higher than expected outcomes ('positive') or lower than expected ('negative')
        'alpha (float)' - the alpha threshold that will be used to compute the score.
            In practice, it may be useful to search over a grid of alpha thresholds and select the one with the maximum score.


        [1] Neill, D. B., & Lingwall, J. (2007). A nonparametric scan statistic for multivariate disease surveillance. Advances in
        Disease Surveillance, 4(106), 570
        """

        super(BerkJones, self).__init__(**kwargs)
        self.alpha = self.kwargs.get('alpha')
        assert self.alpha is not None, "Warning: calling Berk Jones without alpha"

        if self.direction == 'negative':
            self.alpha = 1 - self.alpha

    def score(self, observed_sum: float, expectations: np.array, penalty: float, q: float):
        """
        Computes berk jones score for given q

        :param observed_sum: sum of observed binary outcomes for all i
        :param expectations: predicted outcomes for each data element i
        :param penalty: penalty term. Should be positive
        :param q: current value of q
        :return: berk jones score for the current value of q
        """
        alpha = self.alpha

        if q < alpha:
            q = alpha

        assert q > 0, (
                "Warning: calling compute_score_given_q with "
                "observed_sum=%.2f, expectations of length=%d, penalty=%.2f, q=%.2f, alpha=%.3f"
                % (observed_sum, len(expectations), penalty, q, alpha)
        )
        if q == 1:
            ans = observed_sum * np.log(q / alpha) - penalty
            return ans

        a = observed_sum * np.log(q / alpha)
        b = (len(expectations) - observed_sum) * np.log((1 - q) / (1 - alpha))
        ans = (
                a
                + b
                - penalty
        )

        return ans

    def qmle(self, observed_sum: float, expectations: np.array):
        """
        Computes the q which maximizes score (q_mle).
        for berk jones this is given to be N_a/N
        :param observed_sum: sum of observed binary outcomes for all i
        :param expectations: predicted outcomes for each data element i
        :param direction: direction not considered
        :return: q MLE
        """
        alpha = self.alpha

        if len(expectations) == 0:
            return 0
        else:
            q = observed_sum / len(expectations)

        if (q < alpha):
            return alpha

        return q

    def compute_qs(self, observed_sum: float, expectations: np.array, penalty: float):
        """
        Computes roots (qmin and qmax) of the score function for given q

        :param observed_sum: sum of observed binary outcomes for all i
        :param expectations: predicted outcomes for each data element i
        :param penalty: penalty coefficient
        """
        alpha = self.alpha
        q_mle = self.qmle(observed_sum, expectations)

        if self.score(observed_sum, expectations, penalty, q_mle) > 0:
            exist = 1
            q_min = optim.bisection_q_min(
                self, observed_sum, expectations, penalty, q_mle, temp_min=alpha
            )
            q_max = optim.bisection_q_max(
                self, observed_sum, expectations, penalty, q_mle, temp_max=1
            )
        else:
            # there are no roots
            exist = 0
            q_min = 0
            q_max = 0

        ans = [exist, q_mle, q_min, q_max]
        return ans


from aif360.detectors.mdss.ScoringFunctions.ScoringFunction import ScoringFunction
from aif360.detectors.mdss.ScoringFunctions import optim

import numpy as np


class Gaussian(ScoringFunction):
    def __init__(self, **kwargs):
        """
        Gaussian score function. May be appropriate to use when the outcome of
        interest is assumed to be normally distributed.

        kwargs must contain
        'direction (str)' - direction of the severity; could be higher than expected outcomes ('positive') or lower than expected ('negative')
        """

        super(Gaussian, self).__init__(**kwargs)

    def score(
            self, observed_sum: float, expectations: np.array, penalty: float, q: float
    ):
        """
        Computes gaussian bias score for given q

        :param observed_sum: sum of observed binary outcomes for all i
        :param expectations: predicted outcomes for each data element i
        :param penalty: penalty term. Should be positive
        :param q: current value of q
        :return: bias score for the current value of q
        """

        assumed_var = self.var
        expected_sum = expectations.sum()
        penalty /= self.var

        C = (
                observed_sum * expected_sum / assumed_var * (q - 1)
        )

        B = (
                expected_sum ** 2 * (1 - q ** 2) / (2 * assumed_var)
        )

        if C > B and self.direction == 'positive':
            ans = C + B
        elif B > C and self.direction == 'negative':
            ans = C + B
        else:
            ans = 0

        ans -= penalty

        return ans

    def qmle(self, observed_sum: float, expectations: np.array):
        """
        Computes the q which maximizes score (q_mle).
        """
        expected_sum = expectations.sum()

        # Deals with case where observed_sum = expected_sum = 0
        if observed_sum == expected_sum:
            ans = 1
        else:
            ans = observed_sum / expected_sum

        assert np.isnan(ans) == False, f'{expected_sum}, {observed_sum}, {ans}'
        return ans

    def compute_qs(self, observed_sum: float, expectations: np.array, penalty: float):
        """
        Computes roots (qmin and qmax) of the score function for given q

        :param observed_sum: sum of observed binary outcomes for all i
        :param expectations: predicted outcomes for each data element i
        :param penalty: penalty coefficient
        """

        direction = self.direction

        q_mle = self.qmle(observed_sum, expectations)
        q_mle_score = self.score(observed_sum, expectations, penalty, q_mle)

        if q_mle_score > 0:
            exist = 1
            q_min = optim.bisection_q_min(self, observed_sum, expectations, penalty, q_mle, temp_min=-1e6)
            q_max = optim.bisection_q_max(self, observed_sum, expectations, penalty, q_mle, temp_max=1e6)
        else:
            # there are no roots
            exist = 0
            q_min = 0
            q_max = 0

        # only consider the desired direction, positive or negative
        if exist:
            exist, q_min, q_max = optim.direction_assertions(direction, q_min, q_max)

        ans = [exist, q_mle, q_min, q_max]
        return ans
