import pandas as pd
import numpy as np
from sklearn.tree import _tree
from methods.ml_check.util import local_load, local_save, run_z3, conv_z3_out_to_data


def getDataType(value, dfOrig, i):
    data_type = str(dfOrig.dtypes[i])
    if ('int' in data_type):
        digit = int(value)
    elif ('float' in data_type):
        digit = float(value)
    return digit


def funcAddCond2File(index):
    condition_file_content = local_load('ConditionFile').splitlines()

    smt_file_content = local_load('DecSmt').splitlines()

    smt_file_content = [x.strip() for x in smt_file_content]
    local_save('\n'.join(smt_file_content), 'ToggleBranchSmt', force_rewrite=True)

    toggle_branch_smt = local_load('ToggleBranchSmt')

    toggle_branch_smt = toggle_branch_smt.replace("(check-sat)", '')
    toggle_branch_smt = toggle_branch_smt.replace("(get-model)", '')

    local_save(toggle_branch_smt, 'ToggleBranchSmt', force_rewrite=True)

    toggle_branch_smt = local_load('ToggleBranchSmt')

    temp_cond_content = condition_file_content[index]

    toggle_branch_smt += "(assert (not " + temp_cond_content + "))"
    toggle_branch_smt += "\n"

    toggle_branch_smt += "(check-sat) \n"
    toggle_branch_smt += "(get-model) \n"

    local_save(toggle_branch_smt, 'ToggleBranchSmt', force_rewrite=True)


def funcgetPath4multiLbl(tree, dfMain, noCex, no_param):
    feature_names = dfMain.columns
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    dfT = pd.read_csv('files/TestDataSMTMain.csv')
    pred_arr = np.zeros((tree_.n_outputs))

    node = 0
    depth = 1
    f1 = open('files/SampleFile.txt', 'w')
    f1.write("(assert (=> (and ")
    pathCondFile = open('files/ConditionFile.txt', 'w')

    while (True):
        name = feature_name[node]
        threshold = tree_.threshold[node]

        for i in range(0, dfT.shape[1]):
            if (dfT.columns.values[i] == name):
                index = i

        if (tree_.feature[node] == _tree.TREE_UNDEFINED):
            for i in range(0, tree_.n_outputs):
                pred_arr[i] = np.argmax(tree_.value[node][i])
            if (no_param == 1):
                f1.write(") (= " + str(pred_arr) + ")))")
            else:
                f1.write(") (= " + str(pred_arr) + " " + str(noCex) + ")))")
            break

        index = int(index)

        if (dfT.iloc[0][index] <= threshold):

            node = tree_.children_left[node]

            depth = depth + 1

            threshold = getDataType(threshold, dfMain, index)
            threshold = round(threshold, 5)
            if (no_param == 1):
                f1.write("(<= " + str(name) + " " + str(threshold) + ") ")
            else:
                f1.write("(<= " + str(name) + " " + str(noCex) + " " + str(threshold) + ") ")
            pathCondFile.write("(<= " + str(name) + " " + str(threshold) + ") ")
            pathCondFile.write("\n")



        else:

            node = tree_.children_right[node]

            depth = depth + 1

            threshold = getDataType(threshold, dfMain, index)
            threshold = round(threshold, 5)
            # print(threshold)
            if (no_param == 1):
                f1.write("(> " + str(name) + " " + str(threshold) + ") ")
            else:
                f1.write("(> " + str(name) + " " + str(noCex) + " " + str(threshold) + ") ")

            pathCondFile.write("(> " + str(name) + " " + str(threshold) + ") ")
            pathCondFile.write("\n")

    f1.close()
    pathCondFile.close()


def funcgetPath(tree, dfMain, noCex):
    feature_names = dfMain.columns
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    dfT = local_load('TestDataSMTMain')

    node = 0
    depth = 1
    sample_file = ""
    sample_file += "(assert (=> (and "
    pathCondFile = ""

    while (True):
        name = feature_name[node]
        threshold = tree_.threshold[node]

        for i in range(0, dfT.shape[1]):
            if (dfT.columns.values[i] == name):
                index = i

        if (tree_.feature[node] == _tree.TREE_UNDEFINED):
            sample_file += ") (= Class " + str(np.argmax(tree_.value[node][0])) + ")))"
            break

        index = int(index)
        if (dfT.iloc[noCex][index] <= threshold):
            if noCex > 1:
                print('dddd')
            node = tree_.children_left[node]
            depth = depth + 1
            threshold = getDataType(threshold, dfMain, index)
            sample_file += "(<= " + str(name) + str(noCex) + " " + str(threshold) + ") "
            pathCondFile += "(<= " + str(name) + str(noCex) + " " + str(threshold) + ") "
            pathCondFile += "\n"
        else:
            node = tree_.children_right[node]
            depth = depth + 1
            threshold = getDataType(threshold, dfMain, index)
            sample_file += "(> " + str(name) + str(noCex) + " " + str(threshold) + ") "
            pathCondFile += "(> " + str(name) + str(noCex) + " " + str(threshold) + ") "
            pathCondFile += "\n"

    local_save(sample_file, 'SampleFile', force_rewrite=True)
    local_save(pathCondFile, 'ConditionFile', force_rewrite=True)


def funcPrunInst(dfOrig, dnn_flag):
    local_save(pd.DataFrame(columns=dfOrig.columns), 'CandidateSetInst')

    paramDict = local_load('param_dict')

    if (paramDict['multi_label']):
        noClass = int(paramDict['no_of_class'])
    else:
        noClass = 1

    # Getting the counter example pair (x, x') and saving it to a permanent storage
    dfRead = local_load('TestDataSMTMain')
    dataRead = dfRead.values

    # Combining loop in line 2 & 6 in a single loop
    for j in range(0, dfRead.shape[0]):
        for i in range(0, dfRead.columns.values.shape[0] - noClass):
            # writing content of files/DecSmt.smt2 to another file named files/ToggleFeatureSmt.smt2
            smt_file_content = local_load('DecSmt').splitlines()

            smt_file_content = [x.strip() for x in smt_file_content]

            local_save('\n'.join(smt_file_content), 'ToggleFeatureSmt', force_rewrite=True)

            toggle_feature_smt = local_load('ToggleFeatureSmt')

            toggle_feature_smt = toggle_feature_smt.replace("(check-sat)", '')
            toggle_feature_smt = toggle_feature_smt.replace("(get-model)", '')

            name = str(dfRead.columns.values[i])

            data_type = str(dfOrig.dtypes[i])
            if ('int' in data_type):
                digit = int(dataRead[j][i])
            elif ('float' in data_type):
                digit = float(dataRead[j][i])

            digit = str(digit)
            if 'e' in digit:
                dig = digit.split('e')
                digit = dig[0]
            if ((int(paramDict['no_of_params']) == 1) and (paramDict['multi_label']) and (
                    paramDict['white_box_model'] == 'Decision tree')):
                toggle_feature_smt += "(assert (not (= " + name + " " + digit + "))) \n"
            else:
                toggle_feature_smt += "(assert (not (= " + name + str(j) + " " + digit + "))) \n"
            toggle_feature_smt += "(check-sat) \n"
            toggle_feature_smt += "(get-model) \n"

            run_z3("ToggleFeatureSmt", "FinalOutput")

            satFlag = conv_z3_out_to_data(dfOrig)

            # If sat then add the counter example to the candidate set, refer line 8,9 in prunInst algorithm
            if (satFlag == True):
                dfSmt = local_load('TestDataSMT')
                local_save(dfSmt, 'CandidateSetInst')


def funcPrunBranch(dfOrig, tree_model):
    # data set to hold set of candidate counter examples, refer to cand-set of prunBranch algorithm
    local_save(pd.DataFrame(columns=dfOrig.columns.values), 'CandidateSetBranch', force_rewrite=True)

    paramDict = local_load('param_dict')

    dfRead = local_load('TestDataSMTMain')

    for row in range(0, dfRead.shape[0]):
        if (paramDict['multi_label']):
            funcgetPath4multiLbl(tree_model, dfOrig, row, int(paramDict['no_of_params']))
        else:
            funcgetPath(tree_model, dfOrig, row)

        condition_file = local_load('ConditionFile')
        if not condition_file:
            return
        for i in range(len(condition_file.splitlines())):
            funcAddCond2File(i)
            run_z3('ToggleBranchSmt', 'FinalOutput')
            satFlag = conv_z3_out_to_data(dfOrig)

            if (satFlag == True):
                dfSmt = local_load('TestDataSMT')
                local_save(dfSmt, 'CandidateSetBranch')
