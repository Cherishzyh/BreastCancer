from copy import deepcopy

import torch
from sklearn.metrics import confusion_matrix
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from MeDIT.Others import IterateCase
from T4T.Utility.Data import *
from MeDIT.Statistics import BinaryClassification

from Network2D.ResNeXt2D import *


def EnsembleInference(model_root, data_root, model_name, data_type, weights_list=None):
    input_shape = (80, 80)
    batch_size = 36
    model_folder = os.path.join(model_root, model_name)

    sub_list = pd.read_csv(os.path.join(data_root, '{}_label.csv'.format(data_type)), index_col='CaseName').index.tolist()

    data = DataManager(sub_list=sub_list)
    data.AddOne(Image2D(data_root + '/t1_post_crop_norm', shape=input_shape))
    data.AddOne(Label(data_root + '/label.csv'), is_input=False)

    loader = DataLoader(data, batch_size=batch_size, num_workers=36, pin_memory=True)

    cv_folder_list = [one for one in IterateCase(model_folder, only_folder=True, verbose=0)]
    cv_pred_list, cv_label_list = [], []
    for cv_index, cv_folder in enumerate(cv_folder_list):
        model = ResNeXt(input_channels=3, num_classes=1, num_blocks=[3, 4, 6, 3], block=CBAMBlock).to(device)
        if weights_list is None:
            one_fold_weights_list = [one for one in IterateCase(cv_folder, only_folder=False, verbose=0) if one.is_file()]
            one_fold_weights_list = sorted(one_fold_weights_list,  key=lambda x: os.path.getctime(str(x)))
            weights_path = one_fold_weights_list[-1]
        else:
            weights_path = weights_list[cv_index]

        print(weights_path.name)
        model.load_state_dict(torch.load(str(weights_path)))

        pred_list, label_list = [], []
        model.eval()
        with torch.no_grad():
            for inputs, outputs in loader:
                inputs = MoveTensorsToDevice(inputs, device)

                preds = model(inputs)

                pred_list.append(torch.squeeze(torch.sigmoid(preds).detach()))
                label_list.append(torch.squeeze((outputs)))

        cv_pred_list.append(torch.cat(pred_list))
        cv_label_list.append(torch.cat(label_list))

        auc = roc_auc_score(torch.cat(label_list).tolist(), torch.cat(pred_list).tolist())
        print(auc)

        del model, weights_path

    cv_pred = torch.stack(cv_pred_list)
    cv_label = torch.stack(cv_label_list)
    mean_pred = torch.mean(cv_pred, dim=0).cpu().detach().numpy()
    mean_label = torch.mean(cv_label, dim=0).cpu().detach().numpy()
    auc = roc_auc_score(mean_label, mean_pred)
    print(auc)
    np.save(os.path.join(model_folder, '{}_preds.npy'.format(data_type)), mean_pred)
    np.save(os.path.join(model_folder, '{}_label.npy'.format(data_type)), mean_label)
    return mean_label, mean_pred


def EnsembleInferenceWithROI(model_root, data_root, model_name, data_type, weights_list=None):
    input_shape = (80, 80)
    batch_size = 36
    model_folder = os.path.join(model_root, model_name)

    sub_list = pd.read_csv(os.path.join(data_root, '{}_label.csv'.format(data_type)), index_col='CaseName').index.tolist()

    data = DataManager(sub_list=sub_list)
    data.AddOne(Image2D(data_root + '/t1_post_crop_norm', shape=input_shape))
    data.AddOne(Image2D(data_root + '/lesion_roi_crop_norm', shape=input_shape))
    data.AddOne(Label(data_root + '/label.csv'), is_input=False)


    loader = DataLoader(data, batch_size=batch_size, num_workers=36, pin_memory=True)

    cv_folder_list = [one for one in IterateCase(model_folder, only_folder=True, verbose=0)]
    cv_pred_list, cv_label_list = [], []
    for cv_index, cv_folder in enumerate(cv_folder_list):
        model = ResNeXt(input_channels=3, num_classes=1, num_blocks=[3, 4, 6, 3], block=CBAMBlock).to(device)
        if weights_list is None:
            one_fold_weights_list = [one for one in IterateCase(cv_folder, only_folder=False, verbose=0) if one.is_file()]
            one_fold_weights_list = sorted(one_fold_weights_list,  key=lambda x: os.path.getctime(str(x)))
            weights_path = one_fold_weights_list[-1]
        else:
            weights_path = weights_list[cv_index]

        print(weights_path.name)
        model.load_state_dict(torch.load(str(weights_path)))

        pred_list, label_list = [], []
        model.eval()
        with torch.no_grad():
            for inputs, outputs in loader:
                dilation_roi = torch.from_numpy(binary_dilation(inputs[1].numpy(), structure=np.ones((1, 1, 11, 11))))
                image = inputs[0] * dilation_roi
                image = MoveTensorsToDevice(image, device)

                preds = model(image)

                pred_list.append(torch.squeeze(torch.sigmoid(preds).detach()))
                label_list.append(torch.squeeze((outputs)))

        cv_pred_list.append(torch.cat(pred_list))
        cv_label_list.append(torch.cat(label_list))

        auc = roc_auc_score(torch.cat(label_list).tolist(), torch.cat(pred_list).tolist())
        print(auc)

        del model, weights_path

    cv_pred = torch.stack(cv_pred_list)
    cv_label = torch.stack(cv_label_list)
    mean_pred = torch.mean(cv_pred, dim=0).cpu().detach().numpy()
    mean_label = torch.mean(cv_label, dim=0).cpu().detach().numpy()
    auc = roc_auc_score(mean_label, mean_pred)
    print(auc)
    np.save(os.path.join(model_folder, '{}_preds.npy'.format(data_type)), mean_pred)
    np.save(os.path.join(model_folder, '{}_label.npy'.format(data_type)), mean_label)
    return mean_label, mean_pred


def Result4NPY(model_folder, data_type):
    pred = np.load(os.path.join(model_folder, '{}_preds.npy'.format(data_type)))
    label = np.load(os.path.join(model_folder, '{}_label.npy'.format(data_type)))

    bc = BinaryClassification()
    bc.Run(pred.tolist(), np.asarray(label, dtype=np.int32).tolist())


def DrawROC(model_folder):
    train_pred = np.load(os.path.join(model_folder, '{}_preds.npy'.format('all_train')))
    train_label = np.load(os.path.join(model_folder, '{}_label.npy'.format('all_train')))

    test_pred = np.load(os.path.join(model_folder, '{}_preds.npy'.format('test')))
    test_label = np.load(os.path.join(model_folder, '{}_label.npy'.format('test')))

    plt.figure(0, figsize=(6, 5))
    plt.plot([0, 1], [0, 1], 'k--')

    fpn, sen, the = roc_curve(train_label.tolist(), train_pred.tolist())
    auc = roc_auc_score(train_label.tolist(), train_pred.tolist())
    plt.plot(fpn, sen, label='Train: {:.3f}'.format(auc))

    fpn, sen, the = roc_curve(test_label.tolist(), test_pred.tolist())
    auc = roc_auc_score(test_label.tolist(), test_pred.tolist())
    plt.plot(fpn, sen, label='Test:  {:.3f}'.format(auc))

    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.legend(loc='lower right')
    plt.show()
    plt.close()


if __name__ == '__main__':
    model_root = r'/home/zhangyihong/Documents/BreastClassification/Model'
    data_root = r'/home/zhangyihong/Documents/BreastClassification/DCEPost'

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    # cm = Inference(device, 'ResNeXt_0914_5slice_focal', data_type='train', n_classes=4, weights_list=None)
    # ShowCM(cm)
    # cm = Inference(device, 'ResNeXt_0914_5slice_focal', data_type='val', n_classes=4, weights_list=None)
    # ShowCM(cm)
    # cm = Inference(device, 'ResNeXt_0914_5slice_focal', data_type='test', n_classes=4, weights_list=None)
    # ShowCM(cm)

    # model_name = 'ResNeXt_CBAM_0929_80x80'
    # EnsembleInference(model_root, data_root, model_name, 'all_train', weights_list=None)
    # EnsembleInference(model_root, data_root, model_name, 'test', weights_list=None)

    model_name = 'ResNeXt_CBAM_0929_80x80_cancer'
    EnsembleInference(model_root, data_root, model_name, 'all_train', weights_list=None)
    EnsembleInference(model_root, data_root, model_name, 'test', weights_list=None)


    # Result4NPY(os.path.join(model_root, model_name), data_type='all_train')
    # Result4NPY(os.path.join(model_root, model_name), data_type='test')
    # DrawROC(os.path.join(model_root, model_name))

    # EnsembleInferenceBySeg(model_root, data_root, model_name, 'non_alltrain', weights_list=None, n_class=2)
    # EnsembleInferenceBySeg(model_root, data_root, model_name, 'non_test', weights_list=None, n_class=2)
    # Result4NPY(os.path.join(model_root, model_name), data_type='non_alltrain', n_class=2)
    # Result4NPY(os.path.join(model_root, model_name), data_type='non_test', n_class=2)
    # DrawROC(os.path.join(model_root, model_name))