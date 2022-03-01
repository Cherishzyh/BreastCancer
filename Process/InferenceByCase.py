import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

from MeDIT.Others import IterateCase
from Network2D.ResNet3D import i3_res50
from T4T.Utility.Data import MoveTensorsToDevice


class InferenceByCase():
    def __init__(self):
        super(InferenceByCase).__init__()


    def __GetCenter(self, roi):
        assert (np.ndim(roi) == 3)
        roi = np.squeeze(roi)
        non_zero = np.nonzero(roi)
        center_y = int(np.median(np.unique(non_zero[0])))
        center_x = int(np.median(np.unique(non_zero[1])))
        center_z = int(np.median(np.unique(non_zero[2])))
        return (center_x, center_y, center_z)


    def __AttentionMap(self, data):
        from scipy.ndimage.filters import median_filter, maximum_filter
        from copy import deepcopy

        def InterSliceFilter(attention, diff_value, kernel_size=(3, 1, 1)):
            raw_attention = deepcopy(attention)
            while True:
                new_attention = maximum_filter(raw_attention, size=kernel_size)
                new_attention[new_attention > raw_attention] -= diff_value
                new_attention[new_attention < 0] = 0

                if not (new_attention > raw_attention).any():
                    break

                raw_attention = new_attention
            new_attention = median_filter(new_attention, size=kernel_size)
            new_attention = median_filter(new_attention, size=kernel_size)

            return new_attention

        def IntraSliceFilter(attention, diff_value, kernel_size=(1, 3, 3)):
            raw_attention = deepcopy(attention)

            while True:
                new_attention = maximum_filter(raw_attention, size=kernel_size)
                new_attention[new_attention > raw_attention] -= diff_value
                new_attention[new_attention < 0] = 0

                if not (new_attention > raw_attention).any():
                    break

                raw_attention = new_attention

            new_attention = median_filter(new_attention, size=kernel_size)
            new_attention = median_filter(new_attention, size=kernel_size)

            return new_attention

        base_rate = 0.1
        resolution = (1.5, 1.0, 1.0)
        slice_rate = resolution[0] / resolution[2] * base_rate
        new_data = InterSliceFilter(data, slice_rate)
        new_data = IntraSliceFilter(new_data, base_rate)

        return new_data


    def CropData3D(self, data_list, crop_shape=(100, 100, 50), is_dilated=True):
        from MeDIT.ArrayProcess import ExtractBlock
        from MeDIT.Normalize import NormalizeZ
        '''
        data_list[-1] must be roi
        '''
        assert len(data_list) > 1
        x, y, z = self.__GetCenter(data_list[-1])
        cropped_data = []
        for index, data in enumerate(data_list):
            cropped, _ = ExtractBlock(data, crop_shape, center_point=(y, x, z), is_shift=True)
            cropped = cropped.transpose(2, 0, 1)         #输入网络是slice, height, width
            if index == len(data_list) - 1:
                cropped_data.append(cropped)
                if is_dilated:
                    cropped = self.__AttentionMap(cropped)   #对ROI求attention map
                    cropped_data.append(cropped)
            else:                                            #对图像做Normalization
                cropped_data.append(NormalizeZ(cropped))
        return cropped_data   #[ESER, ADC, t2, roi, roi_dilated]


    def LoadImage(self, data_folder,  sub_list, type_list, label_path=r''):
        ''' ADC_Reg.nii.gz, ESER_1.nii.gz, t2_W_Reg.nii.gz.....roi3D.nii '''
        from MeDIT.SaveAndLoad import LoadImage
        label_df = pd.read_csv(r'/home/zhangyihong/Documents/BreastNpyCorrect/label.csv', index_col='CaseName')
        for case in sorted(os.listdir(data_folder)):
            if len(sub_list) == 0: pass
            else:
                if case not in sub_list: continue
            case_folder = os.path.join(data_folder, case)
            if not os.path.isdir(case_folder): continue

            image_list, data_list = [], []
            label = int(label_df.loc[case, 'Label'])
            for data_type in type_list:
                image, data, _ = LoadImage(os.path.join(case_folder, data_type), is_show_info=False)
                image_list.append(image)
                data_list.append(data)
            yield case, image_list, data_list, label


    def Run(self, data_folder, model_folder, device, weights_list=None, sub_list=[], data_type='test',
            type_list=['ADC_Reg.nii.gz', 'ESER_1.nii.gz', 't2_W_Reg.nii.gz', 'roi3D.nii']):
        cv_folder_list = [one for one in IterateCase(model_folder, only_folder=True, verbose=0)]
        cv_pred_list, cv_label_list, case_list = [], [], []
        for cv_index, cv_folder in enumerate(cv_folder_list):
            model = i3_res50(len(type_list) - 1, 1)
            if torch.cuda.device_count() > 1:
                print("Use", torch.cuda.device_count(), 'gpus')
                model = nn.DataParallel(model)
            model = model.to(device)
            if weights_list is None:
                one_fold_weights_list = [one for one in IterateCase(cv_folder, only_folder=False, verbose=0) if
                                         one.is_file()]
                one_fold_weights_list = sorted(one_fold_weights_list, key=lambda x: os.path.getctime(str(x)))
                weights_path = one_fold_weights_list[-1]
            else:
                weights_path = weights_list[cv_index]

            print(weights_path.name)
            model.load_state_dict(torch.load(str(weights_path)))

            pred_list, label_list = [], []
            model.eval()
            with torch.no_grad():
                for case, _, data_list, label in self.LoadImage(data_folder,
                                                                label_path=os.path.join(data_folder, 'label.csv'),
                                                                sub_list=sub_list,
                                                                type_list=type_list):
                    inputs = self.CropData3D(data_list)
                    dis_map = MoveTensorsToDevice(torch.unsqueeze(torch.from_numpy(inputs[-1]), dim=0), device)
                    inputs = MoveTensorsToDevice(torch.stack([torch.from_numpy(inputs[0]),
                                                              torch.from_numpy(inputs[1]),
                                                              torch.from_numpy(inputs[2])], dim=0), device)

                    preds = model([torch.unsqueeze(inputs, dim=0), torch.unsqueeze(dis_map, dim=0)])

                    pred_list.append(torch.sigmoid(torch.squeeze(preds)).detach())
                    label_list.append(torch.tensor(label))
                    if cv_index == 0: case_list.append(case)

            cv_pred_list.append(torch.stack(pred_list))
            cv_label_list.append(torch.stack(label_list))

            auc = roc_auc_score(torch.stack(label_list).tolist(), torch.stack(pred_list).tolist())
            print(auc)

            del model, weights_path

        cv_pred = torch.stack(cv_pred_list)
        cv_label = torch.stack(cv_label_list)
        mean_pred = torch.mean(cv_pred, dim=0).cpu().detach().numpy()
        mean_label = torch.mean(cv_label.float(), dim=0).cpu().detach().numpy()
        auc = roc_auc_score(mean_label, mean_pred)
        print(auc)
        df = pd.DataFrame({'CaseName': case_list, 'Label': mean_label, 'Pred': mean_pred})
        df.to_csv(os.path.join(model_folder, '{}.csv'.format(data_type)), index=False)
        return case_list, mean_label, mean_pred

    def ShowCM(self, cm, save_folder=r''):
        import seaborn as sns
        sns.set()
        f, ax = plt.subplots(figsize=(8, 8))

        sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', cbar=False)  # 画热力图

        ax.set_title('confusion matrix')  # 标题
        ax.set_xlabel('predict')  # x轴
        ax.set_ylabel('true')  # y轴
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        if save_folder:
            plt.savefig(save_folder, dpi=500)
        else:
            plt.show()
        plt.close()

    def Result(self, label_list, pred_list, save_folder=r''):
        from sklearn import metrics
        from MeDIT.Statistics import BinaryClassification

        assert isinstance(label_list, list)
        assert isinstance(pred_list, list)

        bc = BinaryClassification()
        bc.Run(pred_list, label_list)
        fpr, tpr, threshold = metrics.roc_curve(label_list, pred_list)
        binary_pred = np.array(pred_list)
        index = np.argmax(1 - fpr + tpr)
        binary_pred[binary_pred >= threshold[index]] = 1
        binary_pred[binary_pred < threshold[index]] = 0
        cm = metrics.confusion_matrix(label_list, binary_pred.tolist())
        self.ShowCM(cm, save_folder=save_folder)

    def DrawROC(self, csv_folder, save_folder=r''):
        from sklearn import metrics

        plt.figure(0, figsize=(6, 5))
        plt.plot([0, 1], [0, 1], 'k--')

        train_df = pd.read_csv(os.path.join(csv_folder, '{}.csv'.format('alltrain')), index_col='CaseName')
        train_label = train_df.loc[:, 'Label'].tolist()
        train_pred = train_df.loc[:, 'Pred'].tolist()
        fpn, sen, the = metrics.roc_curve(train_label, train_pred)
        auc = metrics.roc_auc_score(train_label, train_pred)
        plt.plot(fpn, sen, label='Training: {:.3f}'.format(auc))

        test_df = pd.read_csv(os.path.join(csv_folder, '{}.csv'.format('test')), index_col='CaseName')
        test_label = test_df.loc[:, 'Label'].tolist()
        test_pred = test_df.loc[:, 'Pred'].tolist()
        fpn, sen, the = metrics.roc_curve(test_label, test_pred)
        auc = metrics.roc_auc_score(test_label, test_pred)
        plt.plot(fpn, sen, label='Testing:  {:.3f}'.format(auc))

        if os.path.exists(os.path.join(csv_folder, '{}.csv'.format('external'))):
            external_df = pd.read_csv(os.path.join(csv_folder, '{}.csv'.format('external')), index_col='CaseName')
            external_label = external_df.loc[:, 'Label'].tolist()
            external_pred = external_df.loc[:, 'Pred'].tolist()
            fpn, sen, the = metrics.roc_curve(external_label, external_pred)
            auc = metrics.roc_auc_score(external_label, external_pred)
            plt.plot(fpn, sen, label='External: {:.3f}'.format(auc))

        plt.xlabel('1 - Specificity')
        plt.ylabel('Sensitivity')
        plt.legend(loc='lower right')
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        if save_folder:
            plt.savefig(os.path.join(save_folder, 'ROC.jpg'), dpi=500)
        else:
            plt.show()
        plt.close()


def DataPreprocess(save_figure=r'V:\yhzhang\BreastNPYCorrect'):
    type_list = ['ADC_Reg.nii.gz', 'ESER_1.nii.gz', 't2_W_Reg.nii.gz', 'roi3D.nii']
    inference = InferenceByCase()
    for case, _, data_list, _ in inference.LoadImage(data_folder,
                                                     sub_list=[],
                                                     type_list=type_list,
                                                     label_path=r'V:\yhzhang\BreastNPYCorrect\label.csv'):
        cropped_data = inference.CropData3D(case, data_list, crop_shape=(120, 120, 50), is_dilated=True)
        np.save(os.path.join(r'V:\yhzhang\BreastNpyCorrect\Adc', '{}.npy'.format(case)), cropped_data[0])
        np.save(os.path.join(r'V:\yhzhang\BreastNpyCorrect\Eser', '{}.npy'.format(case)), cropped_data[1])
        np.save(os.path.join(r'V:\yhzhang\BreastNpyCorrect\T2', '{}.npy'.format(case)), cropped_data[2])
        np.save(os.path.join(r'V:\yhzhang\BreastNpyCorrect\Roi', '{}.npy'.format(case)), cropped_data[3])
        np.save(os.path.join(r'V:\yhzhang\BreastNpyCorrect\RoiDilated', '{}.npy'.format(case)), cropped_data[4])

        if save_figure:
            from MeDIT.Visualization import FlattenImages

            flatten_data = FlattenImages(cropped_data[3])
            flatten_roi = FlattenImages(cropped_data[4])

            plt.figure(figsize=(16, 8))
            plt.subplot(121)
            plt.axis('off')
            plt.imshow(flatten_data, cmap='gray')
            plt.subplot(122)
            plt.axis('off')
            plt.imshow(flatten_roi, cmap='gray')
            plt.savefig(
                os.path.join(save_figure, 'ImageDilated\{}.jpg'.format(case.split('.npy')[0])))
            plt.close()

            flatten_data = FlattenImages(cropped_data[2])
            flatten_roi = FlattenImages(cropped_data[3])
            plt.figure(figsize=(16, 16))
            plt.imshow(flatten_data, cmap='gray')
            plt.contour(flatten_roi, colors='r')
            plt.axis('off')
            plt.savefig(os.path.join(save_figure, 'Image\{}.jpg'.format(case.split('.npy')[0])))
            plt.close()


if __name__ == '__main__':
    model_root = r'/home/zhangyihong/Documents/BreastNpy/Model'
    data_root = r'/home/zhangyihong/Documents/BreastNpy'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    alltrain_list = pd.read_csv(r'/home/zhangyihong/Documents/BreastNpyCorrect/alltrain_label.csv', index_col='CaseName').index.tolist()
    test_list = pd.read_csv(r'/home/zhangyihong/Documents/BreastNpyCorrect/test.csv', index_col='CaseName').index.tolist()

    inference = InferenceByCase()
    for model_name in ['ResNet3D_20220222', 'ResNet3D_20220223_Adc', 'ResNet3D_20220223_Eser', 'ResNet3D_20220223_T2']:
        if 'Adc' in model_name:
            type_list = ['ADC_Reg.nii.gz', 'roi3D.nii']
        elif 'Eser' in model_name:
            type_list = ['ESER_1.nii.gz', 'roi3D.nii']
        elif 'T2' in model_name:
            type_list = ['t2_W_Reg.nii.gz', 'roi3D.nii']
        else:
            type_list = ['ADC_Reg.nii.gz', 'ESER_1.nii.gz', 't2_W_Reg.nii.gz', 'roi3D.nii']
        inference.Run(data_folder,
                      model_folder=os.path.join(model_root, model_name),
                      device=device,
                      sub_list=alltrain_list,
                      data_type='alltrain',
                      type_list=type_list)
        inference.Run(data_folder,
                      model_folder=os.path.join(model_root, model_name),
                      device=device,
                      sub_list=test_list,
                      data_type='test',
                      type_list=type_list)
        inference.DrawROC(os.path.join(model_root, model_name), save_folder=os.path.join(model_root, model_name))


