from methods.subgroup.biasscan.generator import get_entire_subset, get_random_subset
from methods.subgroup.biasscan.scoring_function import (
    Bernoulli,
    BerkJones,
    Gaussian,
    ScoringFunction,
    Poisson,
)
import pandas as pd
import numpy as np


class DualMDSS:
    def __init__(self, scoring_function: ScoringFunction):
        self.scoring_function = scoring_function
        self.total_observed = None
        self.total_expectations = None

    def get_aggregates(self, coordinates, outcomes, expectations, current_subsets, column_name, penalty):
        aggregates = {}
        thresholds = set()
        total_attr_observed = 0
        total_attr_expectations = []

        for subset in current_subsets:
            if subset:
                mask = coordinates[subset.keys()].isin(subset).all(axis=1)
            else:
                mask = pd.Series(True, index=coordinates.index)

            temp_df = pd.concat([coordinates[mask], outcomes[mask], expectations[mask]], axis=1)

            for name, group in temp_df.groupby(column_name):
                obs_sum = group.iloc[:, -2].sum()
                exp = group.iloc[:, -1].values
                exist, q_mle, q_min, q_max = self.scoring_function.compute_qs(obs_sum, exp, penalty)

                if name not in aggregates:
                    aggregates[name] = []
                aggregates[name].append({
                    'q_mle': q_mle,
                    'q_min': q_min,
                    'q_max': q_max,
                    'observed_sum': obs_sum,
                    'expectations': exp
                })
                thresholds.update([q_min, q_max])

                total_attr_observed += obs_sum
                total_attr_expectations.extend(exp.tolist())

        return aggregates, sorted(thresholds), total_attr_observed, np.array(total_attr_expectations)

    def choose_aggregates(self, aggregates, thresholds, penalty, total_observed, total_expectations):
        best_diff = -np.inf
        best_pair = ([], [])

        for i in range(len(thresholds) - 1):
            threshold = (thresholds[i] + thresholds[i + 1]) / 2
            group1, group2 = [], []
            sum_obs1, sum_exp1 = 0, 0
            sum_obs2, sum_exp2 = 0, 0

            for key, values in aggregates.items():
                for val in values:
                    if val['q_min'] < threshold < val['q_max']:
                        group1.append(key)
                        sum_obs1 += val['observed_sum']
                        sum_exp1 += val['expectations'].sum()
                    else:
                        group2.append(key)
                        sum_obs2 += val['observed_sum']
                        sum_exp2 += val['expectations'].sum()

            q_mle1 = self.scoring_function.qmle(sum_obs1, [sum_exp1])
            q_mle2 = self.scoring_function.qmle(sum_obs2, [sum_exp2])

            penalty1 = penalty * len(group1)
            penalty2 = penalty * len(group2)

            score1 = self.scoring_function.score(sum_obs1, [sum_exp1], penalty1, q_mle1)
            score2 = self.scoring_function.score(sum_obs2, [sum_exp2], penalty2, q_mle2)

            if (score1 - score2) > best_diff:
                best_diff = score1 - score2
                best_pair = (group1, group2)

        all_obs = total_observed
        all_exp = total_expectations.sum()
        q_mle_all = self.scoring_function.qmle(all_obs, [all_exp])
        score_all = self.scoring_function.score(all_obs, [all_exp], 0, q_mle_all)

        if (score_all - score_all) > best_diff:
            return ([], [])

        return best_pair

    def scan(self, coordinates, expectations, outcomes, penalty, num_iters, verbose=False, seed=0, mode='binary'):
        np.random.seed(seed)
        coordinates = coordinates.reset_index(drop=True)
        expectations = expectations.reset_index(drop=True)
        outcomes = outcomes.reset_index(drop=True)

        best_subsets = ([], [])
        best_score = -np.inf

        for _ in range(num_iters):
            current_subsets = [get_entire_subset() if _ == 0 else get_random_subset(coordinates) for _ in range(2)]
            flags = np.zeros(len(coordinates.columns))

            while flags.sum() < len(coordinates.columns):
                attr_idx = np.random.choice(len(coordinates.columns))
                while flags[attr_idx]:
                    attr_idx = np.random.choice(len(coordinates.columns))
                attr = coordinates.columns[attr_idx]

                for i in range(2):
                    if attr in current_subsets[i]:
                        del current_subsets[i][attr]

                aggregates, thresholds, total_obs, total_exp = self.get_aggregates(
                    coordinates, outcomes, expectations, current_subsets, attr, penalty
                )

                group1, group2 = self.choose_aggregates(aggregates, thresholds, penalty, total_obs, total_exp)

                new_subsets = [
                    {**current_subsets[0], attr: group1},
                    {**current_subsets[1], attr: group2}
                ]

                scores = []
                for subset in new_subsets:
                    obs_sum = 0
                    exp_sum = 0
                    penalty_total = 0
                    for k, v in subset.items():
                        mask = coordinates[k].isin(v)
                        obs_sum += outcomes[mask].sum()
                        exp_sum += expectations[mask].sum()
                        penalty_total += len(v) * penalty
                    q_mle = self.scoring_function.qmle(obs_sum, [exp_sum])
                    scores.append(self.scoring_function.score(obs_sum, [exp_sum], penalty_total, q_mle))

                current_diff = scores[0] - scores[1]
                if current_diff > best_score:
                    best_score = current_diff
                    best_subsets = new_subsets

                flags[attr_idx] = 1

        return best_subsets, best_score
