from __future__ import print_function
# import sys
# sys.path.append(r'home/zhangyihong/SSHProject/ProstateECE')
import matplotlib.pyplot as plt
import torch

from grad_cam import GradCAM
from grad_cam_main import save_gradcam


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def demo_my(model, input_list, input_class):
    """
    Generate Grad-GradCam at different layers of ResNet-152
    """
    model.eval()

    target_layers = ["module.layer4"]
    target_class = torch.argmax(input_class)

    gcam = GradCAM(model=model)
    probs = gcam.forward(input_list)

    ids_ = torch.tensor([[target_class]] * 1).long().to(device)
    gcam.backward(ids=ids_)

    for target_layer in target_layers:
        print("Generating Grad-GradCam @{}".format(target_layer))

        # Grad-GradCam
        regions = gcam.generate(target_layer=target_layer, target_shape=(50, 100, 100))

        print("\t#{} ({:.5f})".format(target_class, float(probs)))

        gradcam = save_gradcam(
            gcam=regions[0, 0, ...],
        )

    return probs, gradcam


def GetCenter(roi):
    assert (np.ndim(roi) == 3)
    roi = np.squeeze(roi)
    non_zero = np.nonzero(roi)
    center_y = int(np.median(np.unique(non_zero[1])))
    center_x = int(np.median(np.unique(non_zero[2])))
    center_z = int(np.median(np.unique(non_zero[0])))
    return (center_z, center_x, center_y, )


def TvT(test_list, idx):
    model = i3_res50(len(type_list), 1)
    if torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count())
        model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_root))
    model.to(device)

    input_shape = (100, 100)


    data = DataManager(sub_list=test_list)

    for type in type_list:
        data.AddOne(Image2D(data_root + '/{}'.format(type), shape=input_shape))
    data.AddOne(Image2D(data_root + '/T1WI_pos', shape=input_shape))
    data.AddOne(Image2D(data_root + '/RoiDilated', shape=input_shape))
    data.AddOne(Label(data_root + '/label.csv'), is_input=False)
    loader = DataLoader(data, batch_size=1, num_workers=2, pin_memory=True)
    for i, (inputs, outputs) in enumerate(loader):
        if i == idx:
            dis_map = MoveTensorsToDevice(torch.unsqueeze(inputs[-1], dim=1), device)
            inputs_list = MoveTensorsToDevice(torch.stack(inputs[:-2], dim=1), device)
            label = MoveTensorsToDevice(outputs, device)

            prob, gradcam = demo_my(model, [inputs_list, dis_map], label.long())

            center = GetCenter(np.squeeze(dis_map.data.cpu().numpy()))

            # t2_cropped, _ = ExtractBlock(np.squeeze(inputs[0, -2].data.cpu().numpy()), patch_size=shape, center_point=center)
            # gradcam_cropped, _ = ExtractBlock(np.squeeze(gradcam), patch_size=shape, center_point=center)
            # dis_map_cropped, _ = ExtractBlock(np.squeeze(dis_map.data.cpu()), patch_size=shape, center_point=center)
            # roi_cropped = deepcopy(dis_map_cropped)
            # roi_cropped[roi_cropped > 0] = 1
            merged_image = FusionImage(Normalize01(np.squeeze(inputs[-2][0][center[0]])),
                                       Normalize01(np.squeeze(gradcam[center[0]])), is_show=False)

            plt.suptitle("label: {}, pred: {:.3f}".format(int(label.cpu().data), float(torch.sigmoid(prob).cpu())))
            plt.subplot(121)
            plt.axis('off')
            plt.imshow(np.squeeze(inputs[-2][0][center[0]]), cmap='gray')
            plt.subplot(122)
            plt.axis('off')
            plt.imshow(np.squeeze(merged_image), cmap='jet')

            plt.gca().set_axis_off()
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())

            plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                                wspace=0.00, hspace=0.01)
            plt.savefig(r'/home/zhangyihong/Documents/BreastNpy/Model/ResNet3D_20220222/CV_0/GradCam/{}.jpg'.format(str(i)), format='jpg', dpi=600, bbox_inches='tight', pad_inches=0.00)
            plt.close()
            plt.clf()

        else:
            continue
    torch.cuda.empty_cache()



if __name__ == '__main__':
    from copy import deepcopy
    from T4T.Utility.Data import *
    import torch.nn as nn
    from MeDIT.Normalize import Normalize01
    from MeDIT.Visualization import FusionImage, ShowColorByRoi
    from MeDIT.ArrayProcess import ExtractPatch, ExtractBlock
    from torch.utils.data import DataLoader
    from Network2D.ResNet3D import i3_res50
    from T4T.Utility.Data import MoveTensorsToDevice


    model_root = r'/home/zhangyihong/Documents/BreastNpy/Model/ResNet3D_20220222/CV_0/34-2.419943.pt'
    # model_root = r'/home/zhangyihong/Documents/BreastNpy/Model/ResNet3D_20220226/CV_0/1-8.730039.pt'
    data_root = r'/home/zhangyihong/Documents/BreastNpy'
    output_dir = r'/home/zhangyihong/Documents/ProstateECE/Model'

    type_list = ['Adc', 'Eser', 'T2']
    alltrain_list = pd.read_csv(r'/home/zhangyihong/Documents/BreastNpy/alltrain_label.csv',
                                index_col='CaseName').index.tolist()
    test_list = pd.read_csv(r'/home/zhangyihong/Documents/BreastNpy/test.csv', index_col='CaseName').index.tolist()

    for idx, case in enumerate(test_list):

        TvT(test_list, idx)



