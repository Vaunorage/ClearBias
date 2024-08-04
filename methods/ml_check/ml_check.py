import pandas as pd
import numpy as np
import random as rd
from parsimonious.nodes import NodeVisitor
from parsimonious.grammar import Grammar
import re
import torch

from methods.ml_check import util, assume2logic, assert2logic
from joblib import load
import time

from methods.ml_check.tree2logic import tree_to_code, funcGenSMTFairness
from methods.ml_check.util import local_save, local_load, file_exists, run_z3
from methods.ml_check import pruning


class generateData:
    def __init__(self, df, categorical_columns):
        self.categorical_columns = categorical_columns
        self.df = df
        self.param_dict = local_load('param_dict')
        self.feature_info = self.analyze_data()

    def analyze_data(self):
        """ Analyze the original data to extract distribution information. """
        feature_info = {}
        for column in self.df.columns:
            dtype = self.df[column].dtype
            if column in self.categorical_columns:
                dtype = 'categorical'

            if pd.api.types.is_numeric_dtype(dtype):
                feature_info[column] = {
                    'mean': self.df[column].mean(),
                    'std': self.df[column].std(),
                    'min': self.df[column].min(),
                    'max': self.df[column].max(),
                    'distribution': 'normal'  # Placeholder; you can implement actual distribution fitting
                }
            elif column in self.categorical_columns:
                feature_info[column] = {
                    'categories': self.df[column].value_counts(normalize=True).to_dict(),
                    'most_frequent': self.df[column].mode()[0] if not self.df[column].mode().empty else None
                }
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                feature_info[column] = {
                    'min_date': self.df[column].min(),
                    'max_date': self.df[column].max(),
                    'range': self.df[column].max() - self.df[column].min()
                }
            elif pd.api.types.is_bool_dtype(dtype):
                feature_info[column] = {
                    'true_count': self.df[column].sum(),
                    'false_count': len(self.df[column]) - self.df[column].sum(),
                    'true_ratio': self.df[column].mean()
                }
            else:
                feature_info[column] = {
                    'type': str(dtype),
                    'info': 'Data type not specifically handled'
                }
        return feature_info

    def generate_sample(self):
        sample_data = []
        weight_content = local_load('MUTWeight')

        for column, info in self.feature_info.items():
            if 'distribution' in info:
                if weight_content == 'False\n':
                    value = np.random.normal(info['mean'], info['std'])
                else:
                    value = np.random.uniform(info['min'], info['max'])
                value = np.clip(value, info['min'], info['max'])  # Ensure value is within original range
            else:
                categories = list(info['categories'].keys())
                probabilities = list(info['categories'].values())
                value = np.random.choice(categories, p=probabilities)

            sample_data.append(value)

        return np.array(sample_data, dtype=object).reshape(1, -1)

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
            df_train_data = pd.read_csv(self.param_dict['train_data_loc'])
            self.generate_test_train(df_train_data, int(self.param_dict['train_ratio']))

    def generate_test_train(self, df_train_data, train_ratio):
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


class dataFrameCreate(NodeVisitor):
    def __init__(self):
        self.feName = None
        self.feType = None
        self.feMinVal = -99999
        self.feMaxVal = 0

    def generic_visit(self, node, children):
        pass

    def visit_feName(self, node, children):
        self.feName = node.text

    def visit_feType(self, node, children):
        self.feType = node.text

    def visit_minimum(self, node, children):
        digit = float(re.search(r'\d+', node.text).group(0))
        self.feMinVal = digit

    def visit_maximum(self, node, children):
        digit = float(re.search(r'\d+', node.text).group(0))
        self.feMaxVal = digit


class MakeOracleData:

    def __init__(self, model):
        self.model = model
        self.param_dict = local_load('param_dict')

    def generate_oracle(self):
        df_test = local_load('TestingData')
        X_test = df_test.drop(columns='Class').values

        df_test['Class'] = self.model.predict(X_test)
        local_save(df_test, 'OracleData', force_rewrite=True)


class propCheck:
    def __init__(self, output_class_name, categorical_columns, max_samples=1000, deadline=500000, model=None,
                 no_of_params=None, mul_cex=False, white_box_model="Decision tree", no_of_class=None,
                 no_EPOCHS=100, train_data_available=False,
                 train_data_loc="", multi_label=False, model_path="", no_of_train=1000,
                 train_ratio=100):

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
            'categorical_columns': categorical_columns
        }

        self._validate_params(no_of_params)
        self.model = self._initialize_model(model, model_path)
        self._handle_training_data(train_data_available, train_data_loc, train_ratio)
        local_save(self.paramDict, "param_dict", force_rewrite=True)
        self._generate_data_and_oracle(train_data_loc)

    def _validate_params(self, no_of_params):
        if no_of_params is None or no_of_params > 3:
            raise ValueError("Please provide a valid value for no_of_params (<= 3).")

    def _initialize_model(self, model, model_path):
        mut_weight = "False"
        if model is None:
            if not model_path:
                raise ValueError("Please provide a classifier to check.")
            else:
                model = load(model_path)
                self.paramDict["model_path"] = model_path
                self.paramDict["model_type"] = "sklearn"
        else:
            self.paramDict["model_type"] = "sklearn"
            local_save(model, "MUT")

        local_save(mut_weight, "MUTWeight", force_rewrite=True)
        return model

    def _handle_training_data(self, train_data_available, train_data_loc, train_ratio):
        if train_data_available:
            if not train_data_loc:
                raise ValueError("Please provide the training data location.")
            self.paramDict["train_ratio"] = train_ratio

    def _generate_data_and_oracle(self, train_data_loc):
        df = pd.read_csv(train_data_loc)

        local_save(df.dtypes.apply(str).to_dict(), "feNameType", force_rewrite=True)

        genDataObj = generateData(df, self.paramDict['categorical_columns'])
        genDataObj.generate_test_data()

        genOrcl = MakeOracleData(self.model)
        genOrcl.generate_oracle()


class runChecker:

    def __init__(self):
        self.df = local_load('OracleData')
        self.MUTcontent = file_exists('MUTWeigsht')
        self.paramDict = local_load('param_dict')

        if not self.MUTcontent:
            self.model_type = self.paramDict['model_type']
            if 'model_path' in self.paramDict:
                model_path = self.paramDict['model_path']
                self.model = local_load(model_path)
            else:
                self.model = local_load('MUT')
        else:
            dfWeight = local_load('MUTWeight')
            pred_weight = dfWeight.values
            pred_weight = pred_weight[:, :-1]
            self.model = pred_weight

        local_save(self.df, 'TestSet', force_rewrite=True)
        local_save(self.df, 'CexSet', force_rewrite=True)

    def funcCreateOracle(self):
        dfTest = local_load('TestingData')
        X = dfTest.drop(self.paramDict['output_class_name'], axis=1)
        if not self.MUTcontent:
            predict_class = self.model.predict(X)
            for i in range(0, X.shape[0]):
                dfTest.loc[i, 'Class'] = predict_class[i]
            local_save(dfTest, 'OracleData', force_rewrite=True)
        else:
            predict_list = np.zeros((1, dfTest.shape[0]))
            for i in range(0, X.shape[0]):
                predict_list[0][i] = np.sign(np.dot(self.model, X[i]))
                dfTest.loc[i, 'Class'] = int(predict_list[0][i])
            local_save(dfTest, 'OracleData', force_rewrite=True)

    def chkPairBel(self, tempMatrix, noAttr):
        firstTest = np.zeros((noAttr,))
        secTest = np.zeros((noAttr,))
        dfT = local_load('TestingSet')
        tstMatrix = dfT.values

        for i in range(0, noAttr):
            firstTest[i] = tempMatrix[0][i]

        firstTestList = firstTest.tolist()
        secTestList = secTest.tolist()
        testMatrixList = tstMatrix.tolist()
        for i in range(0, len(testMatrixList) - 1):
            if firstTestList == testMatrixList[i]:
                if secTestList == testMatrixList[i + 1]:
                    return True
        return False

    def chkAttack(self, target_class):
        cexPair = ()
        dfTest = local_load('TestingSet')
        dataTest = dfTest.values
        i = 0
        X = torch.tensor(dataTest, dtype=torch.float32)
        while i < dfTest.shape[0] - 1:
            predict_prob = self.model(X[i].view(-1, X.shape[1]))
            pred_class = int(torch.argmax(predict_prob))
            if pred_class != target_class:
                cexPair = (X[i])
                print('A counter example is found \n')
                print(cexPair)
                return cexPair, True
            i = i + 1
        return cexPair, False

    def funcPrediction(self, X, dfCand, testIndx):
        if not self.MUTcontent:
            return self.model.predict(util.convDataInst(X, dfCand, testIndx, 1))
        else:
            temp_class = np.sign(np.dot(self.model, X[testIndx]))
            if temp_class < 0:
                return 0
            else:
                return temp_class

    def addModelPred(self):
        dfCexSet = local_load('CexSet')
        X = dfCexSet.drop(self.paramDict['output_class_name'], axis=1)
        if not self.MUTcontent:
            predict_class = self.model.predict(X)
            for i in range(0, dfCexSet.shape[0]):
                dfCexSet.loc[i, 'Class'] = predict_class[i]
        else:
            predict_list = np.zeros((1, dfCexSet.shape[0]))
            for i in range(0, X.shape[0]):
                predict_list[0][i] = np.sign(np.dot(self.model, X[i]))
                if predict_list[0][i] < 0:
                    predict_list[0][i] = 0
                dfCexSet.loc[i, 'Class'] = predict_list[0][i]
        local_save(dfCexSet, 'CexSet', force_rewrite=True)

    def funcCheckDuplicate(self, pairfirst, pairsecond, testMatrix):
        pairfirstList = pairfirst.tolist()
        pairsecondList = pairsecond.tolist()
        testDataList = testMatrix.tolist()

        for i in range(0, len(testDataList) - 1):
            if (pairfirstList == testDataList[i]):
                if (pairsecondList == testDataList[i + 1]):
                    return True

        dfTest = local_load('TestSet')
        dataTest = dfTest.values
        dataTestList = dataTest.tolist()
        for i in range(0, len(dataTestList) - 1):
            if (pairfirstList == dataTestList[i]):
                if (pairsecondList == dataTestList[i + 1]):
                    return True
        return False

    def runWithTree(self):
        retrain_flag = False
        MAX_CAND_ZERO = 5
        count_cand_zero = 0
        count = 0
        self.max_samples = int(self.paramDict['max_samples'])
        self.no_of_params = int(self.paramDict['no_of_params'])
        self.mul_cex = self.paramDict['mul_cex_opt']
        self.deadline = int(self.paramDict['deadlines'])
        start_time = time.time()

        while count < self.max_samples:
            print('count is:', count)
            tree = util.train_decision_tree()
            df = local_load('OracleData')
            tree_to_code(tree, df.columns)
            funcGenSMTFairness(df, self.no_of_params)
            # tree2logic.functree2LogicMain(tree, self.no_of_params)
            # util.storeAssumeAssert('DecSmt')
            f2_content = local_load('assumeStmnt')
            local_save(f2_content, 'DecSmt')
            f3_content = local_load('assertStmnt')
            local_save(f3_content, 'DecSmt')
            hh = "\n (check-sat) \n (get-model)"
            local_save(hh, 'DecSmt')
            run_z3('DecSmt', 'FinalOutput')
            satFlag = util.conv_z3_out_to_data(self.df)
            if not satFlag:
                if count == 0:
                    print('No CEX is found by the checker at the first trial')
                    return 0
                elif (count != 0) and (self.mul_cex):
                    dfCexSet = local_load('CexSet')
                    if round(dfCexSet.shape[0] / self.no_of_params) == 0:
                        print('No CEX is found')
                        return 0
                    print('Total number of cex found is:', round(dfCexSet.shape[0] / self.no_of_params))
                    self.addModelPred()
                    return round(dfCexSet.shape[0] / self.no_of_params)
                elif (count != 0) and (self.mul_cex):
                    print('No Cex is found after ' + str(count) + ' no. of trials')
                    return 0
            else:
                df = local_load('TestDataSMT')
                local_save(df, 'CandidateSet', force_rewrite=True)

                df = local_load('TestDataSMT')

                local_save(df, 'TestDataSMTMain', force_rewrite=True)

                df = local_load('OracleData')
                # Pruning by negating the data instance
                pruning.funcPrunInst(df, False)
                dfInst = local_load('CandidateSetInst')
                local_save(dfInst, 'CandidateSet')

                # Pruning by toggling the branch conditions
                pruning.funcPrunBranch(df, tree)
                dfBranch = local_load('CandidateSetBranch')
                local_save(dfBranch, 'CandidateSet')

                dfCandidate = local_load('CandidateSet')
                dataCandidate = dfCandidate.values
                testMatrix = np.zeros((dfCandidate.shape[0], dfCandidate.shape[1]))

                candIndx = 0
                testIndx = 0

                while (candIndx < dfCandidate.shape[0] - 1):
                    pairfirst = dataCandidate[candIndx]
                    pairsecond = dataCandidate[candIndx + 1]
                    if (self.funcCheckDuplicate(pairfirst, pairsecond, testMatrix)):
                        candIndx = candIndx + 2
                    else:
                        for k in range(0, dfCandidate.shape[1]):
                            testMatrix[testIndx][k] = dataCandidate[candIndx][k]
                            testMatrix[testIndx + 1][k] = dataCandidate[candIndx + 1][k]
                        testIndx = testIndx + 2
                        candIndx = candIndx + 2

                local_save(pd.DataFrame(testMatrix), 'TestSet')

                local_save(pd.DataFrame(testMatrix, columns=dfCandidate.columns.values), 'Cand-Set')

                dfTest = local_load('TestSet')
                dfTest = dfTest[(dfTest.T != 0).any()]
                local_save(dfTest, 'TestSet', force_rewrite=True)

                dfCand = local_load('Cand-Set')
                dfCand = dfCand[(dfCand.T != 0).any()]
                local_save(dfCand, 'Cand-Set', force_rewrite=True)

                # processCandCex.funcCheckCex()
                # Increase the count if no further candidate cex has been found
                dfCand = local_load('Cand-Set')
                if round(dfCand.shape[0] / self.no_of_params) == 0:
                    count_cand_zero += 1
                    if count_cand_zero == MAX_CAND_ZERO:
                        if self.mul_cex:
                            dfCexSet = local_load('CexSet')
                            print('Total number of cex found is:', round(dfCexSet.shape[0] / self.no_of_params))
                            if round(dfCexSet.shape[0] / self.no_of_params) > 0:
                                self.addModelPred()
                            return round(dfCexSet.shape[0] / self.no_of_params) + 1
                        else:
                            print('No CEX is found by the checker')
                            return 0
                else:
                    count = count + round(dfCand.shape[0] / self.no_of_params)
                X = dfCand.drop(self.paramDict['output_class_name'], axis=1).values
                y = dfCand[self.paramDict['output_class_name']]
                if dfCand.shape[0] % self.no_of_params == 0:
                    arr_length = dfCand.shape[0]
                else:
                    arr_length = dfCand.shape[0] - 1
                testIndx = 0
                while testIndx < arr_length:
                    temp_count = 0
                    temp_store = []
                    temp_add_oracle = []
                    for i in range(0, self.no_of_params):
                        if self.funcPrediction(X, dfCand, testIndx) == y[testIndx]:
                            temp_store.append(X[testIndx])
                            temp_count += 1
                            testIndx += 1
                        else:
                            retrain_flag = True
                            temp_add_oracle.append(X[testIndx])
                            testIndx += 1
                    if temp_count == self.no_of_params:
                        if self.mul_cex:
                            local_save(temp_store, 'CexSet')
                        else:
                            print('A counter example is found, check it in files/CexSet.csv file: ', temp_store)
                            local_save(temp_store, 'CexSet')
                            self.addModelPred()
                            return 1
                    else:
                        local_save(dfCand, 'TestingData')

                if retrain_flag:
                    self.funcCreateOracle()

                if (time.time() - start_time) > self.deadline:
                    print("Time out")
                    break

        dfCexSet = local_load('CexSet')
        if (round(dfCexSet.shape[0] / self.no_of_params) > 0) and (count >= self.max_samples):
            self.addModelPred()
            print('Total number of cex found is:', round(dfCexSet.shape[0] / self.no_of_params))
            print('No. of Samples looked for counter example has exceeded the max_samples limit')
        else:
            print('No counter example has been found')

    def runPropCheck(self):
        self.max_samples = int(self.paramDict['max_samples'])
        self.no_of_params = int(self.paramDict['no_of_params'])
        self.mul_cex = self.paramDict['mul_cex_opt']
        self.deadline = int(self.paramDict['deadlines'])

        self.runWithTree()


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
