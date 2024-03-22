import argparse
import os
import shutil
from glob import glob
import torch
from networks.unet_3D import unet_3D
from test_3D_util import test_all_case
#from networks.vnet_sdf import VNet
from networks.vnet import VNet
from networks.VoxResNet import VoxRex

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='D:/qinchendong/SSL4MIS_master/data/changhai', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='changhai', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_3D', help='model_name')

def Inference(FLAGS):
    #snapshot_path = "../model/{}/{}".format(FLAGS.exp, FLAGS.model)
    snapshot_path = "D:/qinchendong/model/changhai_hosptial_130/unet_3D"
    num_classes = 2
    test_save_path = "../model/{}/Prediction".format(FLAGS.exp)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    #net = VoxRex( in_channels=1).cuda()
    net = unet_3D(n_classes=num_classes, in_channels=1).cuda()
    #net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True).cuda()
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    avg_metric = test_all_case(net, base_dir=FLAGS.root_path, method=FLAGS.model, test_list="test.txt", num_classes=num_classes,
                               patch_size=(96, 96, 96), stride_xy=64, stride_z=64, test_save_path=test_save_path)
    return avg_metric

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print(metric)
