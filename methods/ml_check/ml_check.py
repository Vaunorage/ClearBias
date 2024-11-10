import pandas as pd
import numpy as np
import random as rd
from parsimonious.nodes import NodeVisitor
from parsimonious.grammar import Grammar
import re
import torch
from joblib import load
import time
from methods.ml_check import util, assume2logic, assert2logic
from methods.ml_check.pruning import Pruner
from methods.ml_check.tree2logic import gen_tree_smt_fairness
from methods.ml_check.util import local_save, local_load, run_z3_solver


class DataGenerator:
    def __init__(self, df, categorical_columns):
        self.categorical_columns = categorical_columns
        self.df = df
        self.param_dict = local_load('param_dict')
        self.feature_info = self._analyze_data()

    def _analyze_data(self):
        feature_info = {}
        for column in self.df.columns:
            dtype = self.df[column].dtype
            if column in self.categorical_columns:
                dtype = 'categorical'
            feature_info[column] = self._get_column_info(column, dtype)
        return feature_info

    def _get_column_info(self, column, dtype):
        if pd.api.types.is_numeric_dtype(dtype):
            return self._get_numeric_info(column)
        elif dtype == 'categorical':
            return self._get_categorical_info(column)
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            return self._get_datetime_info(column)
        elif pd.api.types.is_bool_dtype(dtype):
            return self._get_boolean_info(column)
        else:
            return {'type': str(dtype), 'info': 'Data type not specifically handled'}

    def _get_numeric_info(self, column):
        return {
            'mean': self.df[column].mean(),
            'std': self.df[column].std(),
            'min': self.df[column].min(),
            'max': self.df[column].max(),
            'distribution': 'normal'
        }

    def _get_categorical_info(self, column):
        return {
            'categories': self.df[column].value_counts(normalize=True).to_dict(),
            'most_frequent': self.df[column].mode()[0] if not self.df[column].mode().empty else None
        }

    def _get_datetime_info(self, column):
        return {
            'min_date': self.df[column].min(),
            'max_date': self.df[column].max(),
            'range': self.df[column].max() - self.df[column].min()
        }

    def _get_boolean_info(self, column):
        return {
            'true_count': self.df[column].sum(),
            'false_count': len(self.df[column]) - self.df[column].sum(),
            'true_ratio': self.df[column].mean()
        }

    def generate_sample(self):
        sample_data = []

        for column, info in self.feature_info.items():
            if 'distribution' in info:
                value = self._generate_numeric_value(info)
            else:
                value = self._generate_categorical_value(info)
            sample_data.append(value)

        return np.array(sample_data, dtype=object).reshape(1, -1)

    def _generate_numeric_value(self, info):
        value = np.random.uniform(info['min'], info['max'])
        return np.clip(value, info['min'], info['max'])

    def _generate_categorical_value(self, info):
        categories = list(info['categories'].keys())
        probabilities = list(info['categories'].values())
        return np.random.choice(categories, p=probabilities)

    @staticmethod
    def is_duplicate(matrix, row):
        return row.tolist() in matrix.tolist()

    def generate_test_data(self):
        num_samples = int(self.param_dict['no_of_train'])
        test_matrix = np.zeros((num_samples + 1, len(self.df.columns)), dtype=object)

        i = 0
        while i <= num_samples:
            sample = self.generate_sample()
            if not self.is_duplicate(test_matrix, sample):
                test_matrix[i] = sample[0]
                i += 1

        local_save(pd.DataFrame(test_matrix, columns=self.df.columns), 'TestingData', force_rewrite=True)

        if self.param_dict['train_data_available']:
            self._generate_test_train()

    def _generate_test_train(self):
        df_train_data = pd.read_csv(self.param_dict['train_data_loc'])
        train_ratio = int(self.param_dict['train_ratio'])
        num_samples = round((train_ratio * df_train_data.shape[0]) / 100)
        data = df_train_data.values
        test_matrix = np.zeros((num_samples + 1, df_train_data.shape[1]))
        selected_indices = set()

        while len(selected_indices) <= num_samples:
            index = rd.randint(0, df_train_data.shape[0] - 1)
            if index not in selected_indices:
                selected_indices.add(index)
                test_matrix[len(selected_indices) - 1] = data[index]

        local_save(pd.DataFrame(test_matrix), 'TestingData')


class DataFrameCreator(NodeVisitor):
    def __init__(self):
        self.feName = None
        self.feType = None
        self.feMinVal = -99999
        self.feMaxVal = 0

    def visit_feName(self, node, children):
        self.feName = node.text

    def visit_feType(self, node, children):
        self.feType = node.text

    def visit_minimum(self, node, children):
        self.feMinVal = float(re.search(r'\d+', node.text).group(0))

    def visit_maximum(self, node, children):
        self.feMaxVal = float(re.search(r'\d+', node.text).group(0))


class OracleDataGenerator:
    def __init__(self, model):
        self.model = model
        self.param_dict = local_load('param_dict')

    def generate_oracle(self):
        df_test = local_load('TestingData')
        X_test = df_test.drop(columns=self.param_dict['output_class_name']).values
        df_test[self.param_dict['output_class_name']] = self.model.predict(X_test)
        local_save(df_test, 'OracleData', force_rewrite=True)


class PropertyChecker:
    def __init__(self, output_class_name, all_attributes, categorical_columns, max_samples=1000,
                 deadline=500000, model=None,
                 no_of_params=None, mul_cex=False, white_box_model="Decision tree", no_of_class=None,
                 no_EPOCHS=100, train_data_available=False, train_data_loc="", multi_label=False,
                 model_path="", no_of_train=1000, train_ratio=100):

        self.paramDict = {
            "max_samples": max_samples,
            "deadlines": deadline,
            "white_box_model": white_box_model,
            "no_of_class": no_of_class,
            "no_EPOCHS": no_EPOCHS,
            "no_of_params": no_of_params,
            "mul_cex_opt": mul_cex,
            "multi_label": multi_label,
            "no_of_train": no_of_train,
            "train_data_available": train_data_available,
            "train_data_loc": train_data_loc,
            "train_ratio": train_ratio,
            'output_class_name': output_class_name,
            'categorical_columns': categorical_columns,
            'attributes': all_attributes
        }

        self._validate_params(no_of_params)
        self.model = self._initialize_model(model, model_path)
        self._handle_training_data(train_data_available, train_data_loc, train_ratio)
        local_save(self.paramDict, "param_dict", force_rewrite=True)
        self._generate_data_and_oracle(train_data_loc)

    def _create_param_dict(self, params):
        return {k: v for k, v in params.items() if k != 'self' and v is not None}

    def _validate_params(self, no_of_params):
        if no_of_params is None or no_of_params > 3:
            raise ValueError("Please provide a valid value for no_of_params (<= 3).")

    def _initialize_model(self, model, model_path):
        if model is None:
            if not model_path:
                raise ValueError("Please provide a classifier to check.")
            model = load(model_path)
            self.paramDict["model_path"] = model_path
        self.paramDict["model_type"] = "sklearn"
        local_save(model, "MUT", force_rewrite=True)
        return model

    def _handle_training_data(self, train_data_available, train_data_loc, train_ratio):
        if train_data_available and not train_data_loc:
            raise ValueError("Please provide the training data location.")
        self.paramDict["train_ratio"] = train_ratio

    def _generate_data_and_oracle(self, train_data_loc):
        df = pd.read_csv(train_data_loc)
        local_save(df.dtypes.apply(str).to_dict(), "feNameType", force_rewrite=True)
        data_generator = DataGenerator(df, self.paramDict['categorical_columns'])
        data_generator.generate_test_data()
        oracle_generator = OracleDataGenerator(self.model)
        oracle_generator.generate_oracle()


class RunChecker:
    def __init__(self):
        self._initialize_data()
        self._initialize_model()
        self._save_initial_data()
        self.discriminatory_cases = []

    def _initialize_data(self):
        self.df = local_load('OracleData')
        self.paramDict = local_load('param_dict')

    def _initialize_model(self):
        self._initialize_sklearn_model()

    def _initialize_sklearn_model(self):
        self.model_type = self.paramDict['model_type']
        if 'model_path' in self.paramDict:
            self.model = local_load(self.paramDict['model_path'])
        else:
            self.model = local_load('MUT')

    def _save_initial_data(self):
        local_save(self.df, 'TestSet', force_rewrite=True)
        local_save(self.df, 'CexSet', force_rewrite=True)

    def create_oracle(self):
        dfTest = local_load('TestingData')
        X = dfTest.drop(self.paramDict['output_class_name'], axis=1)
        self._create_sklearn_oracle(dfTest, X)

    def _create_sklearn_oracle(self, dfTest, X):
        dfTest[self.paramDict['output_class_name']] = self.model.predict(X)
        local_save(dfTest, 'OracleData', force_rewrite=True)

    def _create_weight_oracle(self, dfTest, X):
        predict_list = np.sign(np.dot(self.model, X.T)).flatten()
        dfTest[self.paramDict['output_class_name']] = predict_list.astype(int)
        local_save(dfTest, 'OracleData', force_rewrite=True)

    def check_pair_belongs(self, tempMatrix, noAttr):
        firstTest = tempMatrix[0]
        secTest = np.zeros(noAttr)
        dfT = local_load('TestingSet')
        tstMatrix = dfT.values

        for i in range(len(tstMatrix) - 1):
            if np.array_equal(firstTest, tstMatrix[i]) and np.array_equal(secTest, tstMatrix[i + 1]):
                return True
        return False

    def check_attack(self, target_class):
        dfTest = local_load('TestingSet')
        X = torch.tensor(dfTest.values, dtype=torch.float32)

        for i in range(dfTest.shape[0] - 1):
            predict_prob = self.model(X[i].view(-1, X.shape[1]))
            pred_class = int(torch.argmax(predict_prob))
            if pred_class != target_class:
                print('A counter example is found \n')
                print(X[i])
                return X[i], True
        return None, False

    def add_model_predictions(self):
        dfCexSet = local_load('CexSet')
        X = dfCexSet.drop(self.paramDict['output_class_name'], axis=1)

        dfCexSet[self.paramDict['output_class_name']] = self.model.predict(X)
        local_save(dfCexSet, 'CexSet', force_rewrite=True)

    def check_duplicate(self, pairfirst, pairsecond, testMatrix):
        if self._check_duplicate_in_matrix(pairfirst, pairsecond, testMatrix):
            return True
        return self._check_duplicate_in_test_set(pairfirst, pairsecond)

    def _check_duplicate_in_matrix(self, pairfirst, pairsecond, testMatrix):
        testDataList = testMatrix.tolist()
        for i in range(len(testDataList) - 1):
            if np.array_equal(pairfirst, testDataList[i]) and np.array_equal(pairsecond, testDataList[i + 1]):
                return True
        return False

    def _check_duplicate_in_test_set(self, pairfirst, pairsecond):
        dfTest = local_load('TestSet')
        dataTest = dfTest.values
        for i in range(len(dataTest) - 1):
            if np.array_equal(pairfirst, dataTest[i]) and np.array_equal(pairsecond, dataTest[i + 1]):
                return True
        return False

    def run_prop_check(self):
        self.initialize_parameters()

        while self.count < self.max_samples and not self.is_timeout():
            # print(f'count is: {self.count}')

            tree = self.train_and_prepare_tree()

            if not self.check_satisfiability():
                return self.handle_unsatisfiable_case()

            self.process_candidate_set(tree)

            if not self.process_pruned_candidates():
                continue

            self.update_count()

            if self.process_counterexamples():
                if not self.mul_cex:
                    self.save_discriminatory_cases()
                    return 1

            if self.retrain_flag:
                self.create_oracle()

        result = self.finalize_results()
        self.save_discriminatory_cases()
        return result

    def initialize_parameters(self):
        self.retrain_flag = False
        self.MAX_CAND_ZERO = 5
        self.count_cand_zero = 0
        self.count = 0
        self.max_samples = int(self.paramDict['max_samples'])
        self.no_of_params = int(self.paramDict['no_of_params'])
        self.mul_cex = self.paramDict['mul_cex_opt']
        self.deadline = int(self.paramDict['deadlines'])
        self.start_time = time.time()

    def is_timeout(self):
        return (time.time() - self.start_time) > self.deadline

    def train_and_prepare_tree(self):
        tree = util.train_decision_tree(self.paramDict['output_class_name'])
        df = local_load('OracleData')
        gen_tree_smt_fairness(tree, df, self.no_of_params, self.paramDict['output_class_name'])
        self.prepare_smt_file()
        return tree

    def prepare_smt_file(self):
        f2_content = local_load('assumeStmnt')
        f3_content = local_load('assertStmnt')
        final_smt = f2_content + f3_content + "\n (check-sat) \n (get-model)"
        local_save(final_smt, 'DecSmt')

    def check_satisfiability(self):
        run_z3_solver('DecSmt', 'z3_raw_output')
        return util.conv_z3_out_to_data(self.df)

    def handle_unsatisfiable_case(self):
        if self.count == 0:
            print('No CEX is found by the checker at the first trial')
            return 0
        elif self.mul_cex:
            return self.handle_multiple_cex_case()
        else:
            print(f'No Cex is found after {self.count} no. of trials')
            return 0

    def handle_multiple_cex_case(self):
        dfCexSet = local_load('CexSet')
        cex_count = round(dfCexSet.shape[0] / self.no_of_params)
        if cex_count == 0:
            print('No CEX is found')
            return 0
        print(f'Total number of cex found is: {cex_count}')
        self.add_model_predictions()
        return cex_count

    def process_candidate_set(self, tree):
        df = local_load('formatted_z3_output')
        local_save(df, 'CandidateSet', force_rewrite=True)
        local_save(df, 'TestDataSMTMain', force_rewrite=True)

        df = local_load('OracleData')
        Pruner.prune_instance(df)
        Pruner.prune_branch(df, tree, self.paramDict['output_class_name'])

        df_smt_inst = local_load('CandidateSetInst')
        df_smt_branch = local_load('CandidateSetBranch')

        pruned_candidates_result = pd.concat([df_smt_inst, df_smt_branch])
        local_save(pruned_candidates_result, 'CandidateSet')

        dfCandidate = local_load('CandidateSet')

        testMatrix = self.remove_duplicates_from_candidate_set(dfCandidate)

        testMatrix = pd.DataFrame(testMatrix, columns=dfCandidate.columns)
        testMatrix = testMatrix[(testMatrix.T != 0).any()]

        local_save(testMatrix, 'TestSet')
        if not testMatrix.empty:
            testMatrix['indv_key'] = testMatrix.apply(
                lambda row: '|'.join(str(int(row[col])) for col in self.paramDict['attributes']), axis=1)
            testMatrix['couple_key'] = testMatrix.groupby(testMatrix.index // 2)['indv_key'].transform('-'.join)

        else:
            testMatrix['indv_key'] = ''
            testMatrix['couple_key'] = ''

        local_save(testMatrix, 'Cand-Set', force_rewrite=True)

    def remove_duplicates_from_candidate_set(self, dfCandidate):

        dataCandidate = dfCandidate.values
        testMatrix = np.zeros((dfCandidate.shape[0], dfCandidate.shape[1]))

        candIndx = 0
        testIndx = 0

        while candIndx < dfCandidate.shape[0] - 1:
            pairfirst = dataCandidate[candIndx]
            pairsecond = dataCandidate[candIndx + 1]
            if self.check_duplicate(pairfirst, pairsecond, testMatrix):
                candIndx += 2
            else:
                testMatrix[testIndx:testIndx + 2] = dataCandidate[candIndx:candIndx + 2]
                testIndx += 2
                candIndx += 2

        return testMatrix

    def process_pruned_candidates(self):
        dfCand = local_load('Cand-Set')
        if round(dfCand.shape[0] / self.no_of_params) == 0:
            self.count_cand_zero += 1
            if self.count_cand_zero == self.MAX_CAND_ZERO:
                return self.handle_max_cand_zero_case()
        return True

    def handle_max_cand_zero_case(self):
        if self.mul_cex:
            return self.finalize_multiple_cex()
        else:
            print('No CEX is found by the checker')
            return False

    def finalize_multiple_cex(self):
        dfCexSet = local_load('CexSet')
        cex_count = round(dfCexSet.shape[0] / self.no_of_params)
        print(f'Total number of cex found is: {cex_count}')
        if cex_count > 0:
            self.add_model_predictions()
        return cex_count + 1

    def update_count(self):
        dfCand = local_load('Cand-Set')
        self.count += round(dfCand.shape[0] / self.no_of_params)

    def convert_data_instance(self, X, df, index):
        param_dict = local_load('param_dict')
        no_of_class = 1 if param_dict['multi_label'] else param_dict.get('no_of_class', 1)

        if index >= X.shape[0]:
            raise ValueError(
                'Index out of bounds: Z3 might have produced a counter example with all 0 values of the features. Please run the script again.')

        return X[index:index + 1, :df.shape[1] - no_of_class]

    def predict(self, X, index=None):
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)
        elif index is not None:
            X = X[index:index + 1]

        return self.model.predict(X)[0]

    def process_counterexamples(self):
        dfCand = local_load('Cand-Set')
        X = dfCand[self.paramDict['attributes']].values
        y = dfCand[self.paramDict['output_class_name']]

        dfCand.rename(columns={self.paramDict['output_class_name']: 'z3_pred'}, inplace=True)
        try:
            if not dfCand.empty:
                dfCand['whitebox_pred'] = self.model.predict(X)
        except:
            pass

        self.discriminatory_cases.append(dfCand)

        return len(self.discriminatory_cases) > 0

    def save_discriminatory_cases(self):
        if not self.discriminatory_cases:
            print("No discriminatory cases found.")
            return

        df = pd.concat(self.discriminatory_cases)

        local_save(df, 'DiscriminatoryCases', force_rewrite=True)
        print(f"Saved {len(df)} discriminatory cases in {df['couple_key'].nunique()} couples to file.")

    def is_counterexample_group(self, group):
        misclassified = []
        for x, y_true in group:
            y_pred = self.predict(x.reshape(1, -1))
            if y_pred != y_true:
                misclassified.append(x)
                self.retrain_flag = True

        return len(misclassified) == len(group)

    def group_candidates(self, X, y):
        group_size = self.no_of_params
        for i in range(0, len(X) - len(X) % group_size, group_size):
            yield list(zip(X[i:i + group_size], y[i:i + group_size]))

    def _handle_counterexample(self, temp_store):
        if self.mul_cex:
            local_save(temp_store, 'CexSet', force_rewrite=False)
        else:
            print('A counter example is found, check it in files/CexSet.csv file: ', temp_store)
            local_save(temp_store, 'CexSet', force_rewrite=False)
            self.add_model_predictions()
            return True
        return False

    def get_cex_count(self):
        dfCexSet = local_load('CexSet')
        return round(dfCexSet.shape[0] / self.no_of_params)

    def finalize_results(self):
        dfCexSet = local_load('CexSet')
        cex_count = round(dfCexSet.shape[0] / self.no_of_params)
        if cex_count > 0 and self.count >= self.max_samples:
            self.add_model_predictions()
            print(f'Total number of cex found is: {cex_count}')
            print('No. of Samples looked for counter example has exceeded the max_samples limit')
        else:
            print('No counter example has been found')
        return cex_count


def Assume(*args):
    grammar = Grammar(
        r"""

    expr        = expr1 / expr2 / expr3 /expr4 /expr5 / expr6 /expr7
    expr1       = expr_dist1 logic_op num_log
    expr2       = expr_dist2 logic_op num_log
    expr3       = classVar ws logic_op ws value
    expr4       = classVarArr ws logic_op ws value
    expr5       = classVar ws logic_op ws classVar
    expr6       = classVarArr ws logic_op ws classVarArr
    expr7       = "True"
    expr_dist1  = op_beg?abs?para_open classVar ws arith_op ws classVar para_close op_end?
    expr_dist2  = op_beg?abs?para_open classVarArr ws arith_op ws classVarArr para_close op_end?
    classVar    = variable brack_open number brack_close
    classVarArr = variable brack_open variable brack_close
    para_open   = "("
    para_close  = ")"
    brack_open  = "["
    brack_close = "]"
    variable    = ~"([a-zA-Z_][a-zA-Z0-9_]*)"
    logic_op    = ws (geq / leq / eq / neq / and / lt / gt) ws
    op_beg      = number arith_op
    op_end      = arith_op number
    arith_op    = (add/sub/div/mul)
    abs         = "abs"
    add         = "+"
    sub         = "-"
    div         = "/"
    mul         = "*"
    lt          = "<"
    gt          = ">"
    geq         = ">="
    leq         = "<="
    eq          = "="
    neq         = "!="
    and         = "&"
    ws          = ~"\s*"
    value       = ~"\d+"
    num_log     = ~"[+-]?([0-9]*[.])?[0-9]+"
    number      = ~"[+-]?([0-9]*[.])?[0-9]+"
    """)

    tree = grammar.parse(args[0])
    assumeVisitObj = assume2logic.AssumptionVisitor()
    if len(args) == 3:
        assumeVisitObj.storeInd(args[1])
        assumeVisitObj.storeArr(args[2])
        assumeVisitObj.visit(tree)
    elif len(args) == 2:
        assumeVisitObj.storeInd(args[1])
        assumeVisitObj.visit(tree)
    elif len(args) == 1:
        assumeVisitObj.visit(tree)


def Assert(*args):
    grammar = Grammar(
        r"""
        expr        = expr1 / expr2/ expr3
        expr1       = classVar ws operator ws number
        expr2       = classVar ws operator ws classVar
        expr3       = classVar mul_cl_var ws operator ws neg? classVar mul_cl_var
        classVar    = class_pred brack_open variable brack_close
        model_name  = ~"([a-zA-Z_][a-zA-Z0-9_]*)"
        class_pred  = model_name classSymbol
        classSymbol = ~".predict"
        brack_open  = "("
        brack_close = ")"
        variable    = ~"([a-zA-Z_][a-zA-Z0-9_]*)"
        brack3open  = "["
        brack3close = "]"
        class_name  = ~"([a-zA-Z_][a-zA-Z0-9_]*)"
        mul_cl_var  = brack3open class_name brack3close
        operator    = ws (gt/ lt/ geq / leq / eq / neq / and/ implies) ws
        lt          = "<"
        gt          = ">"
        geq         = ">="
        implies     = "=>"
        neg         = "~"
        leq         = "<="
        eq          = "=="
        neq         = "!="
        and         = "&"
        ws          = ~"\s*"
        number      = ~"[+-]?([0-9]*[.])?[0-9]+"
        """
    )

    tree = grammar.parse(args[0])
    assertVisitObj = assert2logic.AssertionVisitor()
    assertVisitObj.visit(tree)
