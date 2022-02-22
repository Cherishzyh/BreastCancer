import os
import shutil

import pandas as pd
from random import shuffle
import SimpleITK as sitk

from MeDIT.SaveAndLoad import LoadImage


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
# CombineCSV()


def CopyFolder():
    source_folder = r'V:\jzhang\breastFormatNew'
    des_folder = r'V:\yhzhang\Breast'
    for case in os.listdir(source_folder):
        if case == 'A029_QIN LU':
            continue
        source_path = os.path.join(source_folder, case)
        if not os.path.isdir(source_path):
            continue
        des_path = os.path.join(des_folder, case)
        if not os.path.exists(des_path):
            os.mkdir(des_path)
        shutil.copyfile(os.path.join(source_path, 'ESER_1.nii.gz'), os.path.join(des_path, 'ESER_1.nii.gz'))
        shutil.copyfile(os.path.join(source_path, 'ADC_Reg.nii.gz'), os.path.join(des_path, 'ADC_Reg.nii.gz'))
        shutil.copyfile(os.path.join(source_path, 't2_W_Reg.nii.gz'), os.path.join(des_path, 't2_W_Reg.nii.gz'))
        shutil.copyfile(os.path.join(source_path, 'roi3D.nii'), os.path.join(des_path, 'roi3D.nii'))
# CopyFolder()

def GenerateLabel():
    folder = r'V:\jzhang\breastFormatNew'
    case_list = []
    label_list = []
    for case in os.listdir(folder):
        case_folder = os.path.join(folder, case)
        if not os.path.isdir(case_folder): continue
        label_csv = pd.read_csv(os.path.join(case_folder, 'label.csv'))
        case_list.append(case)
        label_list.append(label_csv.columns.values.squeeze())
    df = pd.DataFrame({'CaseName': case_list, 'Label': label_list})
    df.to_csv(r'V:\yhzhang\label.csv', index=False)
# GenerateLabel()