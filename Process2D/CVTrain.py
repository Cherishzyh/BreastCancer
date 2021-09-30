import os.path
import shutil

import pandas as pd
from scipy.ndimage import binary_dilation
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score

from MeDIT.Augment import *
from MeDIT.Others import MakeFolder, CopyFile

from T4T.Utility.Data import *
from T4T.Utility.CallBacks import EarlyStopping
from T4T.Utility.Initial import HeWeightInit

from Network2D.ResNeXt2D import *


def ClearGraphPath(graph_path):
    if not os.path.exists(graph_path):
        os.mkdir(graph_path)
    else:
        shutil.rmtree(graph_path)
        os.mkdir(graph_path)


def _GetLoader(data_root, sub_list, aug_param_config, input_shape, batch_size, shuffle, is_balance=True):
    data = DataManager(sub_list=sub_list, augment_param=aug_param_config)
    data.AddOne(Image2D(data_root + '/t1_post_crop_norm', shape=input_shape))
    data.AddOne(Label(data_root + '/label.csv'), is_input=False)
    if is_balance:
        data.Balance(Label(data_root + '/label.csv'))

    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=36, pin_memory=True)
    batches = np.ceil(len(data.indexes) / batch_size)
    return loader, batches


def EnsembleTrain(device, model_root, model_name, data_root):
    torch.autograd.set_detect_anomaly(True)

    input_shape = (80, 80)
    total_epoch = 10000
    batch_size = 128

    model_folder = MakeFolder(model_root + '/{}'.format(model_name))
    if os.path.exists(model_folder):
        ClearGraphPath(model_folder)

    param_config = {
        RotateTransform.name: {'theta': ['uniform', -10, 10]},
        ShiftTransform.name: {'horizontal_shift': ['uniform', -0.05, 0.05],
                              'vertical_shift': ['uniform', -0.05, 0.05]},
        ZoomTransform.name: {'horizontal_zoom': ['uniform', 0.95, 1.05],
                             'vertical_zoom': ['uniform', 0.95, 1.05]},
        FlipTransform.name: {'horizontal_flip': ['choice', True, False]},
        BiasTransform.name: {'center': ['uniform', -1., 1., 2],
                             'drop_ratio': ['uniform', 0., 1.]},
        NoiseTransform.name: {'noise_sigma': ['uniform', 0., 0.03]},
        ContrastTransform.name: {'factor': ['uniform', 0.8, 1.2]},
        GammaTransform.name: {'gamma': ['uniform', 0.8, 1.2]},
        ElasticTransform.name: ['elastic', 1, 0.1, 256]
    }

    alltrain_list = pd.read_csv(os.path.join(data_root, 'all_train.csv'), index_col='CaseName').index.tolist()
    for cv_index in range(5):
        sub_model_folder = MakeFolder(model_folder / 'CV_{}'.format(cv_index))
        sub_val = pd.read_csv(os.path.join(data_root, 'cv_{}.csv'.format(cv_index)), index_col='CaseName').index.tolist()
        sub_train = [case for case in alltrain_list if case not in sub_val]
        train_loader, train_batches = _GetLoader(data_root, sub_train, param_config, input_shape, batch_size, True)
        val_loader, val_batches = _GetLoader(data_root, sub_val, None, input_shape, batch_size, False)

        model = ResNeXt(input_channels=3, num_classes=1, num_blocks=[3, 4, 6, 3], block=CBAMBlock).to(device)
        model.apply(HeWeightInit)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        bce_loss = torch.nn.BCEWithLogitsLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, verbose=True)
        early_stopping = EarlyStopping(store_path=str(sub_model_folder / '{}-{:.6f}.pt'), patience=50, verbose=True)
        writer = SummaryWriter(log_dir=str(sub_model_folder / 'log'), comment='Net')

        for epoch in range(total_epoch):
            train_loss, val_loss = 0., 0.
            train_pred, val_pred = [], []
            train_label, val_label = [], []

            model.train()
            for ind, (inputs, outputs) in enumerate(train_loader):
                inputs = MoveTensorsToDevice(inputs, device)
                outputs = MoveTensorsToDevice(torch.unsqueeze(outputs, dim=1), device)

                preds = model(inputs)

                optimizer.zero_grad()

                loss = bce_loss(preds, outputs)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                train_pred.append(torch.squeeze(torch.sigmoid(preds).detach()))
                train_label.append(torch.squeeze((outputs.data)))

            train_label = torch.cat(train_label)
            train_pred = torch.cat(train_pred)
            binary_pred = deepcopy(train_pred)
            binary_pred[binary_pred >= 0.5] = 1
            binary_pred[binary_pred < 0.5] = 0

            train_acc = torch.sum(train_label == binary_pred) / train_pred.shape[0]
            train_auc = roc_auc_score(train_label.tolist(), train_pred.tolist())

            model.eval()
            with torch.no_grad():
                for ind, (inputs, outputs) in enumerate(val_loader):
                    inputs = MoveTensorsToDevice(inputs, device)
                    outputs = MoveTensorsToDevice(torch.unsqueeze(outputs, dim=1), device)

                    preds = model(inputs)

                    loss = bce_loss(preds, outputs)

                    val_loss += loss.item()

                    val_pred.append(torch.squeeze(torch.sigmoid(preds).detach()))
                    val_label.append(torch.squeeze((outputs.data)))

            val_label = torch.cat(val_label)
            val_pred = torch.cat(val_pred)
            binary_pred = deepcopy(val_pred)
            binary_pred[binary_pred >= 0.5] = 1
            binary_pred[binary_pred < 0.5] = 0

            val_acc = torch.sum(val_label == binary_pred) / val_pred.shape[0]
            val_auc = roc_auc_score(val_label.tolist(), val_pred.tolist())

            # Save Tensor Board
            for index, (name, param) in enumerate(model.named_parameters()):
                if 'bn' not in name:
                    writer.add_histogram(name + '_data', param.cpu().detach().numpy(), epoch + 1)

            writer.add_scalars('Loss', {'train_loss': train_loss / train_batches, 'val_loss': val_loss / val_batches},
                               epoch + 1)
            writer.add_scalars('Acc', {'train_acc': train_acc, 'val_acc': val_acc}, epoch + 1)
            writer.add_scalars('AUC', {'train_auc': train_auc, 'val_auc': val_auc}, epoch + 1)

            print('Epoch {}:\tloss: {:.3f}, val-loss: {:.3f}; acc: {:.3f}, val-acc: {:.3f}; auc: {:.3f}, val-auc: {:.3f}'.format(
                epoch + 1, train_loss / train_batches, val_loss / val_batches, train_acc, val_acc, train_auc, val_auc))

            scheduler.step(val_loss)
            early_stopping(val_loss, model, (epoch + 1, val_loss))

            if early_stopping.early_stop:
                print("Early stopping")
                break
            writer.flush()
        writer.close()

        del writer, optimizer, scheduler, early_stopping, model


if __name__ == '__main__':
    model_root = r'/home/zhangyihong/Documents/BreastClassification/Model'
    data_root = r'/home/zhangyihong/Documents/BreastClassification/DCEPost'
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    EnsembleTrain(device, model_root, 'ResNeXt_CBAM_0929_80x80', data_root)

