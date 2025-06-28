import pandas as pd
import numpy as np
from sklearn.tree import _tree
from methods.ml_check.util import local_load, local_save, run_z3_solver, conv_z3_out_to_data


class DataTypeConverter:
    @staticmethod
    def get_data_type(value, df_orig, i):
        data_type = str(df_orig.dtypes[i])
        if 'int' in data_type:
            return int(value)
        elif 'float' in data_type:
            return float(value)
        return value


class DecisionTreePathExtractor:
    @staticmethod
    def get_path(tree, df_main, no_cex, output_class_name):
        feature_names = df_main.columns
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        df_t = local_load('TestDataSMTMain')

        node = 0
        sample_file = "(assert (=> (and "
        path_cond_file = ""

        while True:
            if tree_.feature[node] == _tree.TREE_UNDEFINED:
                sample_file += f") (= {output_class_name} {np.argmax(tree_.value[node][0])})))"
                break

            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == "undefined!":
                # Skip this node and move to the left child
                node = tree_.children_left[node]
                continue

            index = df_t.columns.get_loc(name)
            threshold = DataTypeConverter.get_data_type(threshold, df_main, index)

            if df_t.iloc[no_cex][index] <= threshold:
                node = tree_.children_left[node]
                operator = "<="
            else:
                node = tree_.children_right[node]
                operator = ">"

            condition = f"({operator} {name}{no_cex} {threshold})"
            sample_file += condition + " "
            path_cond_file += condition + "\n"

        local_save(sample_file, 'SampleFile', force_rewrite=True)
        local_save(path_cond_file, 'ConditionFile', force_rewrite=True)

    @staticmethod
    def get_path_for_multi_label(tree, df_main, no_cex, no_param):
        feature_names = df_main.columns
        tree_ = tree.tree_
        feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]

        df_t = local_load('TestDataSMTMain')
        pred_arr = np.zeros((tree_.n_outputs))

        node = 0
        sample_file = "(assert (=> (and "
        path_cond_file = ""

        while True:
            if tree_.feature[node] == _tree.TREE_UNDEFINED:
                for i in range(tree_.n_outputs):
                    pred_arr[i] = np.argmax(tree_.value[node][i])
                sample_file += f") (= {str(pred_arr)} {'' + str(no_cex) if no_param != 1 else ''})))"
                break

            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == "undefined!":
                node = tree_.children_left[node]
                continue

            index = df_t.columns.get_loc(name)
            threshold = DataTypeConverter.get_data_type(threshold, df_main, index)
            threshold = round(threshold, 5)

            if df_t.iloc[0][index] <= threshold:
                node = tree_.children_left[node]
                operator = "<="
            else:
                node = tree_.children_right[node]
                operator = ">"

            condition = f"({operator} {name}{' ' + str(no_cex) if no_param != 1 else ''} {threshold})"
            sample_file += condition + " "
            path_cond_file += condition + "\n"

        local_save(sample_file, 'SampleFile', force_rewrite=True)
        local_save(path_cond_file, 'ConditionFile', force_rewrite=True)


class Pruner:
    @staticmethod
    def prune_instance(df_orig):
        local_save(pd.DataFrame(columns=df_orig.columns), 'CandidateSetInst')

        param_dict = local_load('param_dict')
        no_class = int(param_dict['no_of_class']) if param_dict['multi_label'] else 1

        df_read = local_load('TestDataSMTMain')
        data_read = df_read.values

        for j in range(df_read.shape[0]):
            for i in range(df_read.columns.values.shape[0] - no_class):
                Pruner._process_feature(df_orig, df_read, data_read, j, i, param_dict)

    @staticmethod
    def _process_feature(df_orig, df_read, data_read, j, i, param_dict):
        smt_file_content = local_load('DecSmt').splitlines()
        smt_file_content = [x.strip() for x in smt_file_content]
        local_save('\n'.join(smt_file_content), 'ToggleFeatureSmt', force_rewrite=True)

        toggle_feature_smt = local_load('ToggleFeatureSmt')
        toggle_feature_smt = toggle_feature_smt.replace("(check-sat)", '').replace("(get-model)", '')

        name = str(df_read.columns.values[i])
        digit = Pruner._get_digit_value(df_orig, data_read, j, i)

        if ((int(param_dict['no_of_params']) == 1) and (param_dict['multi_label']) and
                (param_dict['white_box_model'] == 'Decision tree')):
            toggle_feature_smt += f"(assert (not (= {name} {digit}))) \n"
        else:
            toggle_feature_smt += f"(assert (not (= {name}{j} {digit}))) \n"

        toggle_feature_smt += "(check-sat) \n(get-model) \n"

        local_save(toggle_feature_smt, 'ToggleFeatureSmt', force_rewrite=True)
        run_z3_solver("ToggleFeatureSmt", "z3_raw_output")

        if conv_z3_out_to_data(df_orig):
            df_smt = local_load('formatted_z3_output')
            local_save(df_smt, 'CandidateSetInst')

    @staticmethod
    def _get_digit_value(df_orig, data_read, j, i):
        data_type = str(df_orig.dtypes[i])
        if 'int' in data_type:
            digit = int(data_read[j][i])
        elif 'float' in data_type:
            digit = float(data_read[j][i])
        else:
            digit = data_read[j][i]

        digit = str(digit)
        if 'e' in digit:
            digit = digit.split('e')[0]
        return digit

    @staticmethod
    def prune_branch(df_orig, tree_model, output_class_name):
        local_save(pd.DataFrame(columns=df_orig.columns.values), 'CandidateSetBranch', force_rewrite=True)

        param_dict = local_load('param_dict')
        df_read = local_load('TestDataSMTMain')

        for row in range(df_read.shape[0]):
            if param_dict['multi_label']:
                DecisionTreePathExtractor.get_path_for_multi_label(tree_model, df_orig, row,
                                                                   int(param_dict['no_of_params']))
            else:
                DecisionTreePathExtractor.get_path(tree_model, df_orig, row, output_class_name)

            condition_file = local_load('ConditionFile')
            if not condition_file:
                return

            for index in range(len(condition_file.splitlines())):
                condition_file_content = local_load('ConditionFile').splitlines()
                smt_file_content = local_load('DecSmt').splitlines()
                smt_file_content = [x.strip() for x in smt_file_content]

                toggle_branch_smt = '\n'.join(smt_file_content)
                toggle_branch_smt = toggle_branch_smt.replace("(check-sat)", '').replace("(get-model)", '')

                temp_cond_content = condition_file_content[index]
                toggle_branch_smt += f"(assert (not {temp_cond_content}))\n(check-sat)\n(get-model)\n"

                local_save(toggle_branch_smt, 'ToggleBranchSmt', force_rewrite=True)
                run_z3_solver('ToggleBranchSmt', 'z3_raw_output')
                if conv_z3_out_to_data(df_orig):
                    df_smt = local_load('formatted_z3_output')
                    local_save(df_smt, 'CandidateSetBranch')
