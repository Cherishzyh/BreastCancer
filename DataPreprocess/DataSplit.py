import os
import pandas as pd
from random import shuffle


def DataSplit(cv = 5):
    csv_folder = r'Y:\ZYH\BreastClassification\DCEPost\all_train.csv'
    csv_df = pd.read_csv(csv_folder, index_col='CaseName')
    case_list = [[] for i in range(cv)]
    label_list = [[] for i in range(cv)]

    all_case = csv_df.index.tolist()
    shuffle(all_case)

    fold = 0
    for case in all_case:
        if len(case_list[fold]) > len(all_case) // 5 and fold < cv-1:
            fold += 1
        case_list[fold].append(case)
        label_list[fold].append(csv_df.loc[case].item())
    for fold in range(cv):
        df = pd.DataFrame({'CaseName': case_list[fold], 'Label': label_list[fold]})
        df.to_csv(os.path.join(r'Y:\ZYH\BreastClassification\DCEPost', 'cv_{}.csv'.format(str(fold))), index=False)
# DataSplit()


def CombineCSV():
    train_csv = r'/home/zhangyihong/Documents/BreastClassification/DCEPost/all_train.csv'
    test_csv = r'/home/zhangyihong/Documents/BreastClassification/DCEPost/test_label.csv'
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    new_df = train_df.append(test_df, ignore_index=True)
    new_df.to_csv(r'/home/zhangyihong/Documents/BreastClassification/DCEPost/label.csv', index=False)
CombineCSV()


