from __future__ import division

import pandas as pd

from data_generator.main import get_real_data
import itertools
import random
import time
import lime
import shap
import numpy as np
from lime.lime_tabular import LimeTabularExplainer

from data_generator.main_old import DiscriminationData
from methods.exp_ga.genetic_algorithm import GA
from methods.utils import train_sklearn_model

global_disc_inputs = set()
global_disc_inputs_list = []
local_disc_inputs = set()
local_disc_inputs_list = []
tot_inputs = set()
location = np.zeros(21)
all_found_discriminations = []

threshold_l = 10  # replace census-7,credit-14,bank-10
threshold = 0


def ConstructExplainer(train_vectors, feature_names, class_names):
    explainer = lime.lime_tabular.LimeTabularExplainer(train_vectors, feature_names=feature_names,
                                                       class_names=class_names, discretize_continuous=False)
    return explainer


def Shap_value(model, test_vectors):
    background = shap.kmeans(test_vectors, 10)
    explainer = shap.KernelExplainer(model.predict_proba, background)
    shap_values = explainer.shap_values(test_vectors)
    return shap_values


def Searchseed(model, feature_names, sens_name, explainer, train_vectors, num, X_ori):
    seed = []
    for x in train_vectors:
        tot_inputs.add(tuple(x))
        exp = explainer.explain_instance(x, model.predict_proba, num_features=num)
        explain_labels = exp.available_labels()
        exp_result = exp.as_list(label=explain_labels[0])
        rank = []
        for j in range(len(exp_result)):
            rank.append(exp_result[j][0])

        # Check if any of the sensitive attributes are highly ranked
        is_sensitive_important = False
        for sens in sens_name:
            try:
                loc = rank.index(feature_names[sens])
                location[loc] = location[loc] + 1
                if loc < threshold_l:
                    is_sensitive_important = True
                    break
            except ValueError:
                continue

        if is_sensitive_important:
            seed.append(x)
            imp = []
            for item in feature_names:
                try:
                    pos = rank.index(item)
                    imp.append(exp_result[pos][1])
                except ValueError:
                    imp.append(0)  # Feature not in explanation

        if len(seed) >= 100:
            return seed
    return seed


def Searchseed_Shap(feature_names, sens_name, shap_values, train_vectors):
    seed = []
    for i in range(len(shap_values[0])):
        sample = shap_values[0][i]
        sorted_shapValue = []
        for j in range(len(sample)):
            temp = []
            temp.append(feature_names[j])
            temp.append(sample[j])
            sorted_shapValue.append(temp)
        sorted_shapValue.sort(key=lambda x: abs(x[1]), reverse=True)
        exp_result = sorted_shapValue
        print('shap_value:' + str(exp_result))
        rank = []
        for k in range(len(exp_result)):
            rank.append(exp_result[k][0])
        loc = rank.index(sens_name)
        if loc < 10:
            seed.append(train_vectors[i])
        if len(seed) > 10:
            return seed
    return seed


class Global_Discovery(object):
    def __init__(self, stepsize=1):
        self.stepsize = stepsize

    def __call__(self, iteration, ge: DiscriminationData):
        s = self.stepsize
        samples = []
        while len(samples) < iteration:
            x = np.zeros(len(ge.attr_columns))
            for i in range(len(ge.attr_columns)):
                random.seed(time.time())
                x[i] = random.randint(ge.input_bounds[i][0], ge.input_bounds[i][1])
            for sens_param in ge.sensitive_indices.values():
                x[sens_param] = 0
            samples.append(x)
        return samples


def xai_fair_testing(max_global, max_local):
    # Load and preprocess data based on dataset
    ge, ge_schema = get_real_data('adult')

    # Train the model using the new function
    model, X_train, X_test, y_train, y_test, feature_names = train_sklearn_model(
        data=ge.dataframe,
        model_type='rf',  # You can change this to 'svm', 'lr', or 'dt'
        sensitive_attrs=ge.protected_attributes,
        target_col=ge.outcome_column
    )

    def evaluate_local(inp):
        """
        Evaluate local discrimination by checking individual changes in protected attributes.
        Returns the maximum discrimination found from changing any single protected attribute.
        """

        tot_inputs.add(tuple(inp))

        # Get original prediction
        org_df = pd.DataFrame([inp], columns=ge.attr_columns)
        label = model.predict(org_df)
        org_df['outcome'] = label

        # Get all possible values for each sensitive attribute
        sensitive_values = {}
        for sens_name, sens_idx in ge.sensitive_indices.items():
            sensitive_values[sens_name] = np.unique(ge.xdf.iloc[:, sens_idx]).tolist()

        # Generate all possible combinations of sensitive attribute values
        sensitive_names = list(ge.sensitive_indices.keys())
        value_combinations = list(itertools.product(*[sensitive_values[name] for name in sensitive_names]))

        # Create new test cases with all combinations
        new_targets = []
        for values in value_combinations:
            # Skip if combination is identical to original
            if all(inp[ge.sensitive_indices[name]] == value for name, value in zip(sensitive_names, values)):
                continue

            tnew = pd.DataFrame([inp], columns=ge.attr_columns)
            for name, value in zip(sensitive_names, values):
                tnew[name] = value

            tot_inputs.add(tuple(tnew.to_numpy().tolist()[0]))
            new_targets.append(tnew)

        if not new_targets:  # If no new combinations were generated
            return 0

        new_targets = pd.concat(new_targets)
        new_targets['outcome'] = model.predict(new_targets)

        # Check if any combination leads to a different prediction
        is_discr = new_targets['outcome'] != label[0]
        is_above_threshold = np.abs(new_targets['outcome'] - label[0]) > threshold
        discriminations = new_targets[is_discr & is_above_threshold]

        if discriminations.shape[0] == 0:
            return 0

        for i in range(len(discriminations)):
            disc_tuple = tuple(discriminations.iloc[i][ge.attr_columns].tolist())
            if disc_tuple not in local_disc_inputs:
                local_disc_inputs.add(disc_tuple)
                local_disc_inputs_list.append(list(discriminations.iloc[i].values))
                all_found_discriminations.append((org_df, discriminations.iloc[i]))

        max_discrimination = np.max(np.abs(discriminations['outcome'] - label[0]))

        return 2 * max_discrimination + 1

    start = time.time()

    global_discovery = Global_Discovery()

    train_samples = global_discovery(max_global, ge)
    train_samples = np.array(train_samples)

    np.random.shuffle(train_samples)

    print(train_samples.shape)

    explainer = ConstructExplainer(ge.xdf, ge.attr_columns, list(ge.ydf.unique()))

    seed = Searchseed(model, ge.attr_columns, ge.sensitive_indices.values(), explainer, train_samples,
                      len(ge.attr_columns),
                      ge.xdf)

    print('Finish Searchseed')
    for inp in seed:
        inp0 = [int(i) for i in inp]
        inp0 = np.asarray(inp0)
        inp0 = np.reshape(inp0, (1, -1))
        global_disc_inputs.add(tuple(map(tuple, inp0)))
        global_disc_inputs_list.append(inp0.tolist()[0])

    print("Finished Global Search")
    print('length of total input is:' + str(len(tot_inputs)))
    print('length of global discovery is:' + str(len(global_disc_inputs_list)))

    end = time.time()

    print('Total time:' + str(end - start))

    print("")
    print("Starting Local Search")

    cross_rate = 0.9
    mutation = 0.05
    iteration = max_local
    ga = GA(nums=global_disc_inputs_list, bound=ge.input_bounds, func=evaluate_local, DNA_SIZE=len(ge.input_bounds),
            cross_rate=cross_rate, mutation=mutation)

    count = 300
    for i in range(iteration):
        ga.evolve()
        end = time.time()
        use_time = end - start
        if use_time >= count:
            print("Percentage discriminatory inputs - " + str(float(len(local_disc_inputs_list))
                                                              / float(len(tot_inputs)) * 100))
            print("Number of discriminatory inputs are " + str(len(local_disc_inputs_list)))

            print('use time:' + str(end - start))
            count += 300

    print("Total Inputs are " + str(len(tot_inputs)))
    print("Number of discriminatory inputs are " + str(len(local_disc_inputs_list)))
    print("Percentage discriminatory inputs - " + str(
        float(len(local_disc_inputs_list)) / float(len(tot_inputs)) * 100))
    print("Total Inputs are " + str(len(tot_inputs)))
    print("Number of discriminatory inputs are " + str(len(local_disc_inputs_list)))


def main():
    xai_fair_testing(max_global=1000, max_local=1000)


if __name__ == '__main__':
    main()
