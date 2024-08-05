import numpy as np

from methods.ml_check.util import local_load, local_save

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import re

from sklearn.tree import _tree


def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    f = "def tree({}):".format(", ".join(feature_names))
    f += "\n"

    def recurse(node, depth):
        nonlocal f  # This allows us to modify the string 'f' within the nested function
        indent = "  " * depth

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            f += "{}if {} <= {}:".format(indent, name, threshold)
            f += "\n"

            f += "{}".format(indent) + "{"
            f += "\n"

            recurse(tree_.children_left[node], depth + 1)

            f += "{}".format(indent) + "}"
            f += "\n"

            f += "{}else:  # if {} > {}".format(indent, name, threshold)
            f += "\n"

            f += "{}".format(indent) + "{"
            f += "\n"

            recurse(tree_.children_right[node], depth + 1)

            f += "{}".format(indent) + "}"
            f += "\n"

        else:
            f += "{}return {}".format(indent, np.argmax(tree_.value[node][0]))
            f += "\n"

    recurse(0, 1)

    local_save(f, 'TreeOutput', force_rewrite=True)


def funcConvBranch(single_branch, dfT, rep):
    f = local_load('DecSmt')
    f += "(assert (=> (and "
    for i in range(0, len(single_branch)):
        temp_Str = single_branch[i]
        if ('if' in temp_Str):
            # temp_content[i] = content[i]
            for j in range(0, dfT.columns.values.shape[0]):
                if (dfT.columns.values[j] in temp_Str):
                    fe_name = str(dfT.columns.values[j])
                    fe_index = j

            data_type = str(dfT.dtypes[fe_index])

            if ('<=' in temp_Str):
                sign = '<='
            elif ('<=' in temp_Str):
                sign = '>'
            elif ('>' in temp_Str):
                sign = '>'
            elif ('>=' in temp_Str):
                sign = '>='

            if ('int' in data_type):
                digit = int(re.search(r'\d+', temp_Str).group(0))
            elif ('float' in data_type):
                digit = float(re.search(r'\d+', temp_Str).group(0))
            digit = str(digit)
            f += "(" + sign + " " + fe_name + str(rep) + " " + digit + ") "

        elif ('return' in temp_Str):
            digit_class = int(re.search(r'\d+', temp_Str).group(0))
            digit_class = str(digit_class)
            f += ") (= Class" + str(rep) + " " + digit_class + ")))"
            f += '\n'
    local_save(f, 'DecSmt', force_rewrite=True)


def funcGetBranch(sinBranch, dfT, rep):
    for i in range(0, len(sinBranch)):
        tempSt = sinBranch[i]
        if ('return' in tempSt):
            funcConvBranch(sinBranch, dfT, rep)


def gen_smt_branch(dfT, rep):
    file_content = local_load('TreeOutput').splitlines()
    file_content = [x.strip() for x in file_content]

    noOfLines = len(file_content)
    temp_file_cont = ["" for x in range(noOfLines)]

    i = 1
    k = 0
    while (i < noOfLines):

        j = k - 1
        if temp_file_cont[j] == '}':
            funcGetBranch(temp_file_cont, dfT, rep)
            while True:
                if (temp_file_cont[j] == '{'):
                    temp_file_cont[j] = ''
                    temp_file_cont[j - 1] = ''
                    j = j - 1
                    break
                elif (j >= 0):
                    temp_file_cont[j] = ''
                    j = j - 1

            k = j

        else:
            temp_file_cont[k] = file_content[i]
            k = k + 1
            i = i + 1

    if ('return' in file_content[1]):
        digit = int(re.search(r'\d+', file_content[1]).group(0))
        f = local_load('DecSmt')
        f += "(assert (= Class" + str(rep) + " " + str(digit) + "))"
        f += "\n"
        local_save(f, 'DecSmt', force_rewrite=True)
    else:
        funcGetBranch(temp_file_cont, dfT, rep)


def gen_tree_smt_fairness(tree, dfT, no_of_instances):
    tree_to_code(tree, dfT.columns)
    feName_type = local_load('feNameType')

    f = ''
    for j in range(0, no_of_instances):
        for col in dfT.columns.tolist():
            fe_type = feName_type[col]

            if ('int' in fe_type):
                f += "(declare-fun " + col + str(j) + " () Int)"
                f += '\n'
                f += '\n'
            elif ('float' in fe_type):
                f += "(declare-fun " + col + str(j) + " () Real)"
                f += '\n'
                f += '\n'
        f += "; " + str(j) + "th element"
        f += '\n'

    f += '(define-fun absoluteInt ((x Int)) Int \n'
    f += '  (ite (>= x 0) x (- x))) \n'

    f += '(define-fun absoluteReal ((x Real)) Real \n'
    f += '  (ite (>= x 0) x (- x))) \n'

    local_save(f, 'DecSmt', force_rewrite=True)

    # Calling function to get the branch and convert it to z3 form,  creating alias
    for i in range(0, no_of_instances):
        f = local_load('DecSmt')
        f += '\n;-----------' + str(i) + '-----------number instance-------------- \n'
        local_save(f, 'DecSmt', force_rewrite=True)
        gen_smt_branch(dfT, i)
