import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from MeDIT.Others import IterateCase
from T4T.Utility.Data import *
from MeDIT.Statistics import BinaryClassification

from Network2D.ResNet3D import i3_res50


def EnsembleInference(model_root, data_root, model_name, data_type, type_list, weights_list=None):
    input_shape = (100, 100)
    batch_size = 48
    model_folder = os.path.join(model_root, model_name)

    sub_list = pd.read_csv(os.path.join(data_root, '{}'.format(data_type)), index_col='CaseName').index.tolist()

    data = DataManager(sub_list=sub_list)
    for type in type_list:
        data.AddOne(Image2D(data_root + '/{}'.format(type), shape=input_shape))
        # data.AddOne(Image2D(data_root + '/Eser', shape=input_shape))
        # data.AddOne(Image2D(data_root + '/T2', shape=input_shape))
    # data.AddOne(Image2D(data_root + '/Adc', shape=input_shape))
    # data.AddOne(Image2D(data_root + '/Eser', shape=input_shape))
    # data.AddOne(Image2D(data_root + '/T2', shape=input_shape))
    data.AddOne(Image2D(data_root + '/RoiDilated', shape=input_shape))
    data.AddOne(Label(data_root + '/label.csv'), is_input=False)

    loader = DataLoader(data, batch_size=batch_size, num_workers=2, pin_memory=True)

    cv_folder_list = [one for one in IterateCase(model_folder, only_folder=True, verbose=0)]
    cv_pred_list, cv_label_list = [], []
    for cv_index, cv_folder in enumerate(cv_folder_list):
        # if cv_index > 2: continue
        model = i3_res50(len(type_list), 1)
        if torch.cuda.device_count() > 1:
            # print("Use", torch.cuda.device_count(), 'gpus')
            model = nn.DataParallel(model)
        model = model.to(device)
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
                dis_map = MoveTensorsToDevice(torch.unsqueeze(inputs[-1], dim=1), device)
                inputs = MoveTensorsToDevice(torch.stack(inputs[:-1], dim=1), device)

                preds = model([inputs, dis_map])
                pred_list.append(torch.sigmoid(torch.squeeze(preds, dim=1)).detach())
                label_list.append(outputs)

        cv_pred_list.append(torch.cat(pred_list))
        cv_label_list.append(torch.cat(label_list))

        auc = roc_auc_score(torch.cat(label_list).tolist(), torch.cat(pred_list).tolist())
        print(auc)

        del model, weights_path

    cv_pred = torch.stack(cv_pred_list)
    cv_label = torch.stack(cv_label_list)
    mean_pred = torch.mean(cv_pred, dim=0).cpu().detach().numpy()
    mean_label = torch.mean(cv_label, dim=0).cpu().detach().numpy()

    if 'alltrain' in data_type:
        np.save(os.path.join(model_folder, 'alltrain_preds.npy'), mean_pred)
        np.save(os.path.join(model_folder, 'alltrain_label.npy'), mean_label)
    else:
        np.save(os.path.join(model_folder, 'test_preds.npy'), mean_pred)
        np.save(os.path.join(model_folder, 'test_label.npy'), mean_label)

    bc = BinaryClassification()
    mean_pred = [float(pred) for pred in mean_pred.tolist()]
    mean_label = [int(label) for label in mean_label.tolist()]
    bc.Run(mean_pred, mean_label)
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
    model_root = r'/home/zhangyihong/Documents/BreastNpy/Model'
    data_root = r'/home/zhangyihong/Documents/BreastNpy'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    # cm = Inference(device, 'ResNeXt_0914_5slice_focal', data_type='train', n_classes=4, weights_list=None)
    # ShowCM(cm)
    # cm = Inference(device, 'ResNeXt_0914_5slice_focal', data_type='val', n_classes=4, weights_list=None)
    # ShowCM(cm)
    # cm = Inference(device, 'ResNeXt_0914_5slice_focal', data_type='test', n_classes=4, weights_list=None)
    # ShowCM(cm)

    # model_name = 'ResNet3D_20220225_Einitial'
    # print(model_name)
    # EnsembleInference(model_root, data_root, model_name, 'alltrain_label.csv', weights_list=None, type_list=['Einitial'])
    # EnsembleInference(model_root, data_root, model_name, 'test.csv', weights_list=None, type_list=['Einitial'])
    #
    # model_name = 'ResNet3D_20220225_DWIb2000'
    # print(model_name)
    # EnsembleInference(model_root, data_root, model_name, 'alltrain_label.csv', weights_list=None, type_list=['DWIb2000'])
    # EnsembleInference(model_root, data_root, model_name, 'test.csv', weights_list=None, type_list=['DWIb2000'])
    #
    # model_name = 'ResNet3D_20220225_DWIb1000'
    # print(model_name)
    # EnsembleInference(model_root, data_root, model_name, 'alltrain_label.csv', weights_list=None, type_list=['DWIb1000'])
    # EnsembleInference(model_root, data_root, model_name, 'test.csv', weights_list=None, type_list=['DWIb1000'])
    #
    # model_name = 'ResNet3D_20220225_T1WI_pos'
    # print(model_name)
    # EnsembleInference(model_root, data_root, model_name, 'alltrain_label.csv', weights_list=None, type_list=['T1WI_pos'])
    # EnsembleInference(model_root, data_root, model_name, 'test.csv', weights_list=None, type_list=['T1WI_pos'])



    label_df = pd.read_csv(r'/home/zhangyihong/Documents/BreastNpy/label.csv', index_col='CaseName')
    alltrain_list = pd.read_csv(r'/home/zhangyihong/Documents/BreastNpy/alltrain_label.csv', index_col='CaseName').index.tolist()
    test_list = pd.read_csv(r'/home/zhangyihong/Documents/BreastNpy/test.csv', index_col='CaseName').index.tolist()
    case_list, label_list = [], []
    combine_model = r'/home/zhangyihong/Documents/BreastNpy/Model/ResNet3D_20220225_Einitial'
    adc_model = r'/home/zhangyihong/Documents/BreastNpy/Model/ResNet3D_20220225_DWIb2000'
    t2_model = r'/home/zhangyihong/Documents/BreastNpy/Model/ResNet3D_20220225_DWIb1000'
    eser_model = r'/home/zhangyihong/Documents/BreastNpy/Model/ResNet3D_20220225_T1WI_pos'

    for case in sorted(test_list):
        case_list.append(case)
        label_list.append(label_df.loc[case]['Label'])
    combine_pred = np.load(os.path.join(combine_model, 'test_preds.npy')).tolist()
    adc_pred = np.load(os.path.join(adc_model, 'test_preds.npy')).tolist()
    eser_pred = np.load(os.path.join(eser_model, 'test_preds.npy')).tolist()
    t2_pred = np.load(os.path.join(t2_model, 'test_preds.npy')).tolist()
    combine_label = np.load(os.path.join(combine_model, 'test_label.npy')).tolist()
    adc_label = np.load(os.path.join(adc_model, 'test_label.npy')).tolist()
    eser_label = np.load(os.path.join(eser_model, 'test_label.npy')).tolist()
    t2_label = np.load(os.path.join(t2_model, 'test_label.npy')).tolist()
    df = pd.DataFrame({'CaseName': case_list, 'Label': label_list,
                       'Einitial': combine_pred,
                       'DWIb2000': t2_pred,
                       'DWIb1000': eser_pred,
                       'T1WI_pos': adc_pred})
    df.to_csv(r'/home/zhangyihong/Documents/BreastNpy/Model/ResultTest.csv', index=False)

    case_list, label_list = [], []
    for case in sorted(alltrain_list):
        case_list.append(case)
        label_list.append(label_df.loc[case]['Label'])
    combine_pred = np.load(os.path.join(combine_model, 'alltrain_preds.npy')).tolist()
    adc_pred = np.load(os.path.join(adc_model, 'alltrain_preds.npy')).tolist()
    eser_pred = np.load(os.path.join(eser_model, 'alltrain_preds.npy')).tolist()
    t2_pred = np.load(os.path.join(t2_model, 'alltrain_preds.npy')).tolist()
    combine_label = np.load(os.path.join(combine_model, 'alltrain_label.npy')).tolist()
    adc_label = np.load(os.path.join(adc_model, 'alltrain_label.npy')).tolist()
    eser_label = np.load(os.path.join(eser_model, 'alltrain_label.npy')).tolist()
    t2_label = np.load(os.path.join(t2_model, 'alltrain_label.npy')).tolist()
    df = pd.DataFrame({'CaseName': case_list, 'Label': label_list,
                       'Einitial': combine_pred,
                       'DWIb2000': t2_pred,
                       'DWIb1000': eser_pred,
                       'T1WI_pos': adc_pred})
    df.to_csv(r'/home/zhangyihong/Documents/BreastNpy/Model/Result.csv', index=False)


