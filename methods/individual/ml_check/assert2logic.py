import re
from parsimonious.nodes import NodeVisitor
from methods.ml_check.util import local_save, local_load


class AssertionVisitor(NodeVisitor):

    def __init__(self):
        self.currentClass = []
        self.modelVarList = []
        self.classNameList = []
        self.currentOperator = ""
        self.negOp = ""
        self.varList = []
        self.mydict = {}
        self.varMap = {}
        self.feVal = 0
        self.count = 0
        self.dfOracle = local_load('OracleData')
        self.mydict = local_load('dict')
        self.paramDict = local_load('param_dict')

    def generic_visit(self, node, children):
        pass

    def visit_classVar(self, node, children):
        if not self.mydict['no_mapping']:
            for el in self.varList:
                if el in node.text:
                    if self.mydict['no_assumption']:
                        className = self.paramDict['output_class_name'] + str(self.mydict[el])
                    else:
                        className = self.paramDict['output_class_name'] + str(self.count - 1)
            self.currentClass.append(className)

    def visit_neg(self, node, children):
        self.negOp = node.text

    def visit_model_name(self, node, children):
        self.modelVarList.append(node.text)

    def visit_class_name(self, node, children):
        if node.text in self.dfOracle.columns.values:
            self.classNameList.append(node.text)
        else:
            raise Exception(f'Class name {node.text} does not exist')

    def visit_variable(self, node, children):
        if not self.mydict['no_mapping']:
            self.varList.append(node.text)
            if self.mydict['no_assumption']:
                num = str(int(re.search(r'\d+', self.mydict[node.text]).group(0)))
                self.mydict[node.text] = num[len(num) - 1]
            else:
                if node.text not in self.varMap:
                    self.varMap[node.text] = self.count
                    self.count += 1

    def visit_operator(self, node, children):
        if '!=' in node.text:
            self.currentOperator = 'not(= '
        elif '==' in node.text:
            self.currentOperator = '= '
        else:
            self.currentOperator = node.text

    def visit_number(self, node, children):
        self.feVal = float(node.text)

    def visit_expr1(self, node, children):
        if self.mydict['no_mapping']:
            assertStmnt = f'(assert(not ({self.currentOperator} Class0 {self.feVal})))'
        else:
            assertStmnt = f'(assert(not ({self.currentOperator} {self.currentClass[0]} {self.feVal})))'

        local_save(assertStmnt, 'assertStmnt')

    def checkModelName(self):
        if self.modelVarList[0] != self.modelVarList[1]:
            raise Exception('Model names do not match')

    def visit_expr2(self, node, children):
        self.checkFeConsist()
        self.checkModelName()
        assertStmnt = f'(assert(not ({self.currentOperator} {self.currentClass[0]} {self.currentClass[1]})))'

        local_save(assertStmnt, 'assertStmnt')

    def visit_expr3(self, node, children):
        if self.count > int(self.paramDict['no_of_params']):
            raise Exception('The number of parameters mentioned exceeded in assert statement')
        self.checkModelName()
        if self.negOp == '~':
            if self.paramDict['white_box_model'] == 'DNN':
                assertStmnt = (
                    f'(assert(not ({self.currentOperator} (= {self.classNameList[0]} {self.count - 1} 1) '
                    f'(not (= {self.classNameList[1]} {self.count - 1} 1)))))'
                )
            else:
                assertStmnt = (
                    f'(assert(not ({self.currentOperator} (= {self.classNameList[0]} 1) '
                    f'(not (= {self.classNameList[1]} 1)))))'
                )
        else:
            if self.paramDict['white_box_model'] == 'DNN':
                assertStmnt = (
                    f'(assert(not ({self.currentOperator} (= {self.classNameList[0]} {self.count - 1} 1) '
                    f'(= {self.classNameList[1]} {self.count - 1} 1))))'
                )
            else:
                assertStmnt = (
                    f'(assert(not ({self.currentOperator} (= {self.classNameList[0]} 1) '
                    f'(= {self.classNameList[1]} 1))))'
                )

        local_save(assertStmnt, 'assertStmnt')

    def checkFeConsist(self):
        if len(self.varList) == len(self.mydict) - 2:
            for el in self.varList:
                if el not in self.mydict.keys():
                    raise Exception("Unknown feature vector")
        else:
            raise Exception("Number of feature vectors do not match with the assumption")
