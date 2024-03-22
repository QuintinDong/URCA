import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.dataset import (BaseDataSets, RandomGenerator,
                                 TwoStreamBatchSampler)
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume
from networks.vision_transformer import SwinUnet as ViT_seg
from config import get_config

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default=r'D:/qinchendong/SSL4MIS_master/data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Regularized_Dropout', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=10000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=12,#12
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],#256
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=6,#6
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=14,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.999, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=4.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
args = parser.parse_args()

def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def update_ema_variables_co(model1, model2, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param1, param2 in zip(ema_model.parameters(), model1.parameters(),model2.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, (param1.data + param2.data)/2)

def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=1,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model1 = create_model()
    model2 = create_model()
    ema_model = create_model(ema=True)
    ema_model_high = create_model(ema=True)
    
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)

    model1.train()
    model2.train()
    ema_model.train()
    ema_model_high.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    optimizer3 = optim.SGD(ema_model.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    optimizer4 = optim.SGD(ema_model_high.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            unlabeled_volume_batch = volume_batch[args.labeled_bs:].cuda()  # volume_batch是字典，把args.labeled_bs:赋值给unlabeled_volume_batch
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise  # 加入噪声

            outputs1  = model1(volume_batch)
            outputs1_unlabeled = model1(unlabeled_volume_batch)
            #print(outputs1.shape)
            outputs_soft1 = torch.softmax(outputs1, dim=1)
            outputs_soft1_unlabeled = torch.softmax(outputs1_unlabeled, dim=1)

            outputs2 = model2(volume_batch)
            outputs2_unlabeled = model2(unlabeled_volume_batch)
            outputs_soft2 = torch.softmax(outputs2, dim=1)
            outputs_soft2_unlabeled = torch.softmax(outputs2_unlabeled, dim=1)
            consistency_weight = get_current_consistency_weight(iter_num // 150)

            with torch.no_grad():
                ema_outputs = ema_model(ema_inputs)
                #print(ema_outputs.shape)
                ema_outputs_soft = torch.softmax(ema_outputs,dim=1)

            T = 8
            _, _, w, h = unlabeled_volume_batch.shape
            volume_batch_r = unlabeled_volume_batch.repeat(2, 1, 1, 1)
            stride = volume_batch_r.shape[0] // 2
            preds = torch.zeros([stride * T, num_classes, w, h]).cuda()  # 返回一个由标量0填充的张量，形状由size决定
            for i in range(T // 2):  # torch.randn_like返回一个与输入张量大小相同的张量，其中填充了均值为 0 方差为 1 的正态分布的随机值。
                ema_inputs = volume_batch_r + \
                             torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)  # clamp（）函数的功能将输入input张量每个元素的值压缩到区间 [min,max]，并返回结果到一个新张量
                with torch.no_grad():
                    preds[2 * stride * i:2 * stride * (i + 1)] = ema_model(ema_inputs)
            preds = F.softmax(preds, dim=1)
            preds = preds.reshape(T, stride, num_classes, w, h)
            preds = torch.mean(preds, dim=0)
            uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)

            threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(iter_num,#0.75
                                                            max_iterations)) * np.log(2)

            threshold_high_level = (0.8 + 0.25 * ramps.sigmoid_rampup(iter_num,  # 高级的不确定性有更高的阈值
                                                                      max_iterations)) * np.log(2)
            mask = (uncertainty < threshold).float()
            uncertainty_sample = torch.matmul(mask, unlabeled_volume_batch)
            #print(uncertainty_sample.shape)
            #conv_layer = torch.nn.Conv2d(4, 1, kernel_size=1).cuda()
            #uncertainty_sample = conv_layer(uncertainty_sample)

            with torch.no_grad():
                ema_output_high_level = ema_model_high(uncertainty_sample)  # todo---------高级教师的输出是ema_output_high_level--------
                ema_outputs_high_soft = torch.softmax(ema_output_high_level, dim=1)

            # print(outputs1[args.labeled_bs:].shape)
            # print(ema_outputs.shape)
            _, _, w2, h2 = uncertainty_sample.shape#todo---------------高级教师生成的不确定性------
            preds_2 = torch.zeros([stride * T, num_classes, w2, h2]).cuda()
            uncertainty_sample_input = uncertainty_sample.repeat(2, 1, 1, 1)
            for i in range(T // 2):
                # uncertainty_sample_high = volume_batch_r_high_level + \
                #                      torch.clamp(torch.randn_like(
                #                          # torch.randn_like返回一个与输入张量大小相同的张量，其中填充了均值为 0 方差为 1 的正态分布的随机值。
                #                          volume_batch_r_high_level) * 0.1, -0.2,
                #                                  0.2)  # clamp（）函数的功能将输入input张量每个元素的值压缩到区间 [min,max]，并返回结果到一个新张量
                uncertainty_sample_high = uncertainty_sample_input +  torch.clamp(torch.randn_like(uncertainty_sample_input) * 0.1, -0.2,0.2)
                                         # clamp（）函数的功能将输入input张量每个元素的值压缩到区间 [min,max]，并返回结果到一个新张量
                                         # torch.randn_like返回一个与输入张量大小相同的张量，其中填充了均值为 0 方差为 1 的正态分布的随机值。
                with torch.no_grad():
                    #uncertainty_sample_high = conv_layer(uncertainty_sample_high)
                    preds_2[2 * stride * i:2 * stride *
                                         (i + 1)] = ema_model_high(uncertainty_sample_high)

                    #preds_2[2 * stride * i:2 * stride *(i + 1)] = ema_model_high(uncertainty_sample_high)
            preds_2 = F.softmax(preds_2, dim=1)
            preds_2 = preds_2.reshape(T, stride, num_classes, w, h)
            preds_2 = torch.mean(preds_2, dim=0)
            uncertainty_2 = -1.0 * torch.sum(preds_2 * torch.log(preds_2 + 1e-6), dim=1, keepdim=True)

            mask_2 = (uncertainty_2 < threshold_high_level).float()

            mask_3 = torch.add(mask, mask_2)

            consistency_dist_1 = losses.softmax_mse_loss(  # ------------todo-------dice_loss_________
                outputs1[args.labeled_bs:], ema_outputs)  # (batch, 2, 112,112,80)

            consistency_dist_2 = losses.softmax_mse_loss(  # ------------todo-------dice_loss_________
                outputs2[args.labeled_bs:], ema_outputs)  # (batch, 2, 112,112,80)

            consistency_dist_high_level = losses.softmax_mse_loss(
                # todo-----------------------低级教师与高级教师之间的loss------------dice_loss-----
                ema_output_high_level, ema_outputs)

            consistency_dist_high_level_outputs_2 = losses.softmax_mse_loss(
                # todo-----------------------低级教师与高级教师之间的loss------------dice_loss-----
                outputs2[args.labeled_bs:],ema_output_high_level)

            consistency_loss_1 = torch.sum(mask * consistency_dist_1) / (2 * torch.sum(mask) + 1e-16)
            consistency_loss_2 = torch.sum(mask_2 * consistency_dist_2) / (2 * torch.sum(mask_2) + 1e-16)
            consistency_loss_high_level = torch.sum(mask_3 * consistency_dist_high_level) / (2 * torch.sum(mask_3) + 1e-16)
            consistency_loss_high_level_2 = torch.sum(mask_3 * consistency_dist_high_level_outputs_2) / (2 * torch.sum(mask_3) + 1e-16)

            model1_loss =  (ce_loss(outputs1[:args.labeled_bs], label_batch[:args.labeled_bs].long()) + dice_loss(
                outputs_soft1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))
            model2_loss = (ce_loss(outputs2[:args.labeled_bs], label_batch[:args.labeled_bs].long()) + dice_loss(
                outputs_soft2[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))
            ema_model_loss = (ce_loss(ema_outputs, label_batch[:args.labeled_bs].long()) + dice_loss(
                ema_outputs_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))
            ema_model_high_supervised_loss = (ce_loss(ema_output_high_level, label_batch[:args.labeled_bs].long()) + dice_loss(
                ema_outputs_high_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))

            ema_model_high_loss = 0.5 * (losses.compute_kl_loss(ema_output_high_level, ema_outputs) + losses.dice_loss(ema_outputs_high_soft, ema_outputs))
            # ema_output1_loss = losses.compute_kl_loss(outputs1[args.labeled_bs:], ema_outputs) + losses.dice_loss(outputs_soft1_unlabeled,ema_outputs)
            #ema_output2_loss = losses.compute_kl_loss(outputs2[args.labeled_bs:], ema_outputs) + losses.dice_loss(outputs_soft2_unlabeled,ema_outputs)
            ema_output1_high_loss = 0.5 * losses.compute_kl_loss(outputs1[args.labeled_bs:], ema_output_high_level) + losses.dice_loss(outputs_soft1_unlabeled,ema_output_high_level)
            #ema_output2_high_loss = losses.compute_kl_loss(outputs2[args.labeled_bs:], ema_output_high_level) + losses.dice_loss(outputs_soft2_unlabeled,ema_output_high_level)
            r_drop_loss = losses.compute_kl_loss(outputs1[args.labeled_bs:], outputs2[args.labeled_bs:])

            loss = model1_loss + model2_loss + ema_model_loss +ema_model_high_supervised_loss + consistency_weight *  r_drop_loss + \
                   consistency_weight * consistency_loss_1 + consistency_weight * consistency_loss_2 + consistency_weight * consistency_loss_high_level + \
                   consistency_weight * ema_output1_high_loss + consistency_weight * ema_model_high_loss + consistency_weight * consistency_loss_high_level_2
            # loss = model1_loss + model2_loss + ema_model_loss + ema_model_high_supervised_loss + consistency_weight * r_drop_loss + consistency_weight * consistency_loss_1

            optimizer1.zero_grad()
            optimizer2.zero_grad()


            loss.backward()

            optimizer1.step()
            optimizer2.step()


            update_ema_variables_co(model1, model2, ema_model, args.ema_decay, iter_num)#todo-----------更新ema_model的参数
            # update_ema_variables(model2, ema_model, args.ema_decay, iter_num)
            update_ema_variables(ema_model, ema_model_high, args.ema_decay, iter_num)

            iter_num = iter_num + 1

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9

            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar(
                'consistency_weight/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss/model1_loss',
                              model1_loss, iter_num)
            writer.add_scalar('loss/model2_loss',
                              model2_loss, iter_num)
            writer.add_scalar('loss/r_drop_loss',
                              r_drop_loss, iter_num)
            logging.info('iteration %d : model1 loss : %f model2 loss : %f r_drop_loss: %f' % (iter_num, model1_loss.item(), model2_loss.item(), r_drop_loss.item()))

            if iter_num % 50 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs1, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model1_Prediction',
                                 outputs[1, ...] * 50, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs2, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model2_Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model1.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model1, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/model1_val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model1_val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)
                    writer.add_scalar('info/model1_val_{}_asd'.format(class_i + 1),
                                      metric_list[class_i, 2], iter_num)
                    writer.add_scalar('info/model1_val_{}_hd'.format(class_i + 1),
                                      metric_list[class_i, 3], iter_num)
                    writer.add_scalar('info/model1_val_{}_jc'.format(class_i + 1),
                                      metric_list[class_i, 4], iter_num)

                performance1 = np.mean(metric_list, axis=0)[0]
                mean_hd951 = np.mean(metric_list, axis=0)[1]
                mean_asd1 = np.mean(metric_list, axis=0)[2]
                mean_hd1 = np.mean(metric_list, axis=0)[3]
                mean_jc1 = np.mean(metric_list, axis=0)[4]

                writer.add_scalar('info/model1_val_mean_dice', performance1, iter_num)
                writer.add_scalar('info/model1_val_mean_hd95', mean_hd951, iter_num)
                writer.add_scalar('info/model1_val_mean_asd', mean_asd1, iter_num)
                writer.add_scalar('info/model1_val_mean_hd', mean_hd1, iter_num)
                writer.add_scalar('info/model1_val_mean_jc', mean_jc1, iter_num)

                if performance1 > best_performance1:
                    best_performance1 = performance1
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model1_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance1, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model1.pth'.format(args.model))
                    torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best)

                logging.info(
                    'iteration %d : model1_mean_dice : %f model1_mean_hd95 : %f  model1_mean_asd: %f model1_mean_hd: %f model1_mean_jc: %f' % (iter_num, performance1, mean_hd951,mean_asd1,mean_hd1,mean_jc1))
                model1.train()
                model2.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model2, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/model2_val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model2_val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)
                    writer.add_scalar('info/model2_val_{}_asd'.format(class_i + 1),
                                      metric_list[class_i, 2], iter_num)
                    writer.add_scalar('info/model2_val_{}_hd'.format(class_i + 1),
                                      metric_list[class_i, 3], iter_num)
                    writer.add_scalar('info/model2_val_{}_jc'.format(class_i + 1),
                                      metric_list[class_i, 4], iter_num)

                performance2 = np.mean(metric_list, axis=0)[0]
                mean_hd952 = np.mean(metric_list, axis=0)[1]
                mean_asd2 = np.mean(metric_list, axis=0)[2]
                mean_hd2 = np.mean(metric_list, axis=0)[3]
                mean_jc2 = np.mean(metric_list, axis=0)[4]

                writer.add_scalar('info/model2_val_mean_dice', performance2, iter_num)
                writer.add_scalar('info/model2_val_mean_hd95', mean_hd952, iter_num)
                writer.add_scalar('info/model2_val_mean_asd', mean_asd2, iter_num)
                writer.add_scalar('info/model2_val_mean_hd', mean_hd2, iter_num)
                writer.add_scalar('info/model2_val_mean_jc', mean_jc2, iter_num)

                if performance2 > best_performance2:
                    best_performance2 = performance2
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model2_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance2, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model2.pth'.format(args.model))
                    torch.save(model2.state_dict(), save_mode_path)
                    torch.save(model2.state_dict(), save_best)

                logging.info(
                    'iteration %d : model2_mean_dice : %f model2_mean_hd95 : %f model2_mean_asd: %f model2_mean_hd: %f model2_mean_jc: %f' % (iter_num, performance2, mean_hd952,mean_asd2,mean_hd2,mean_jc2))
                model2.train()






                ema_model_high.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], ema_model_high, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/ema_model_high_val_{}_dice'.format(class_i + 1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/ema_model_high_val_{}_hd95'.format(class_i + 1),
                                      metric_list[class_i, 1], iter_num)
                    writer.add_scalar('info/ema_model_high_val_{}_asd'.format(class_i + 1),
                                      metric_list[class_i, 2], iter_num)
                    writer.add_scalar('info/ema_model_high_val_{}_hd'.format(class_i + 1),
                                      metric_list[class_i, 3], iter_num)
                    writer.add_scalar('info/ema_model_high_val_{}_jc'.format(class_i + 1),
                                      metric_list[class_i, 4], iter_num)

                performance2 = np.mean(metric_list, axis=0)[0]
                mean_hd952 = np.mean(metric_list, axis=0)[1]
                mean_asd2 = np.mean(metric_list, axis=0)[2]
                mean_hd2 = np.mean(metric_list, axis=0)[3]
                mean_jc2 = np.mean(metric_list, axis=0)[4]

                writer.add_scalar('info/ema_model_high_val_mean_dice', performance2, iter_num)
                writer.add_scalar('info/ema_model_high_val_mean_hd95', mean_hd952, iter_num)
                writer.add_scalar('info/ema_model_high_val_mean_asd', mean_asd2, iter_num)
                writer.add_scalar('info/ema_model_high_val_mean_hd', mean_hd2, iter_num)
                writer.add_scalar('info/ema_model_high_val_mean_jc', mean_jc2, iter_num)

                if performance2 > best_performance2:
                    best_performance2 = performance2
                    save_mode_path = os.path.join(snapshot_path,
                                                  'ema_model_high_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance2, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_ema_model_high.pth'.format(args.model))
                    torch.save(ema_model_high.state_dict(), save_mode_path)
                    torch.save(ema_model_high.state_dict(), save_best)

                logging.info(
                    'iteration %d : model2_mean_dice : %f model2_mean_hd95 : %f model2_mean_asd: %f model2_mean_hd: %f model2_mean_jc: %f' % (
                    iter_num, performance2, mean_hd952, mean_asd2, mean_hd2, mean_jc2))
                ema_model_high.train()






                ema_model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], ema_model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/ema_model_val_{}_dice'.format(class_i + 1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/ema_model_val_{}_hd95'.format(class_i + 1),
                                      metric_list[class_i, 1], iter_num)
                    writer.add_scalar('info/ema_model_val_{}_asd'.format(class_i + 1),
                                      metric_list[class_i, 2], iter_num)
                    writer.add_scalar('info/ema_model_val_{}_hd'.format(class_i + 1),
                                      metric_list[class_i, 3], iter_num)
                    writer.add_scalar('info/ema_model_val_{}_jc'.format(class_i + 1),
                                      metric_list[class_i, 4], iter_num)

                performance2 = np.mean(metric_list, axis=0)[0]
                mean_hd952 = np.mean(metric_list, axis=0)[1]
                mean_asd2 = np.mean(metric_list, axis=0)[2]
                mean_hd2 = np.mean(metric_list, axis=0)[3]
                mean_jc2 = np.mean(metric_list, axis=0)[4]

                writer.add_scalar('info/ema_model_val_mean_dice', performance2, iter_num)
                writer.add_scalar('info/ema_model_val_mean_hd95', mean_hd952, iter_num)
                writer.add_scalar('info/ema_model_val_mean_asd', mean_asd2, iter_num)
                writer.add_scalar('info/ema_model_val_mean_hd', mean_hd2, iter_num)
                writer.add_scalar('info/ema_model_val_mean_jc', mean_jc2, iter_num)

                if performance2 > best_performance2:
                    best_performance2 = performance2
                    save_mode_path = os.path.join(snapshot_path,
                                                  'ema_model_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance2, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_ema_high.pth'.format(args.model))
                    torch.save(ema_model.state_dict(), save_mode_path)
                    torch.save(ema_model.state_dict(), save_best)

                logging.info(
                    'iteration %d : model2_mean_dice : %f model2_mean_hd95 : %f model2_mean_asd: %f model2_mean_hd: %f model2_mean_jc: %f' % (
                        iter_num, performance2, mean_hd952, mean_asd2, mean_hd2, mean_jc2))
                ema_model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'model1_iter_' + str(iter_num) + '.pth')
                torch.save(model1.state_dict(), save_mode_path)
                logging.info("save model1 to {}".format(save_mode_path))

                save_mode_path = os.path.join(
                    snapshot_path, 'model2_iter_' + str(iter_num) + '.pth')
                torch.save(model2.state_dict(), save_mode_path)
                logging.info("save model2 to {}".format(save_mode_path))

                # save_mode_path = os.path.join(
                #     snapshot_path, 'ema_model_high_iter_' + str(iter_num) + '.pth')
                # torch.save(ema_model_high.state_dict(), save_mode_path)
                # logging.info("save ema_model_high to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
