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

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='D:/qinchendong/SSL4MIS_master/data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/our_model_Uncertainty_Aware_Mean_Teacher', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=12,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=6,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=14,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.999, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')#200
args = parser.parse_args()

# class OutConv(nn.Sequential):
#     def __init__(self, in_channels, num_classes):
#         super(OutConv, self).__init__(
#             nn.Conv2d(in_channels, num_classes, kernel_size=1)
#         )

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


def update_ema_variables(model, ema_model,alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def update_ema_variables_plus(model,model2, ema_model,alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param1,param2 in zip(ema_model.parameters(), model.parameters(), model2.parameters()):
        param = torch.add(param1, param2)/2
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

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

    model = create_model()
    ema_model = create_model(ema=True)
    model2 = create_model()

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

    model.train()
    model2.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    optimizer_1 = optim.SGD(ema_model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    optimizer_2 = optim.SGD(model2.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)



    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_performance2 = 0.0
    best_performance3 = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']#sampled_batch是字典，把变量image,label赋给volume_batch,label_batch
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()#把变量放到gpu
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]#volume_batch是字典，把args.labeled_bs:赋值给unlabeled_volume_batch

            noise = torch.clamp(torch.randn_like(
                unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise#加入噪声
            #print(ema_inputs.shape)#torch.Size([12, 1, 256, 256])

            outputs = model(volume_batch)#输出
            outputs_1 = ema_model(volume_batch)
            outputs_2 = model2(volume_batch)
            #print(outputs.shape)#torch.Size([24, 4, 256, 256])
            outputs_soft = torch.softmax(outputs, dim=1)#对输出做softmax
            outputs_soft_1 = torch.softmax(outputs_1, dim=1)
            outputs_soft_2 = torch.softmax(outputs_2, dim=1)

            with torch.no_grad():
                ema_output = ema_model(ema_inputs)

            T = 8
            _, _, w, h = unlabeled_volume_batch.shape
            volume_batch_r = unlabeled_volume_batch.repeat(2, 1, 1, 1)
            stride = volume_batch_r.shape[0] // 2
            preds = torch.zeros([stride * T, num_classes, w, h]).cuda()#返回一个由标量0填充的张量，形状由size决定
            for i in range(T//2):
                ema_inputs = volume_batch_r + \
                    torch.clamp(torch.randn_like(#torch.randn_like返回一个与输入张量大小相同的张量，其中填充了均值为 0 方差为 1 的正态分布的随机值。
                        volume_batch_r) * 0.1, -0.2, 0.2)#clamp（）函数的功能将输入input张量每个元素的值压缩到区间 [min,max]，并返回结果到一个新张量
                with torch.no_grad():
                    preds[2 * stride * i:2 * stride *
                          (i + 1)] = ema_model(ema_inputs)
            preds = F.softmax(preds, dim=1)
            preds = preds.reshape(T, stride, num_classes, w, h)
            preds = torch.mean(preds, dim=0)
            uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)

            loss_ce = ce_loss(outputs[:args.labeled_bs],
                              label_batch[:args.labeled_bs][:].long())
            loss_dice = dice_loss(
                outputs_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            loss_ce_1 = ce_loss(outputs_1[:args.labeled_bs],
                              label_batch[:args.labeled_bs][:].long())
            loss_dice_1 = dice_loss(
                outputs_soft_1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            loss_ce_2 = ce_loss(outputs_2[:args.labeled_bs],
                              label_batch[:args.labeled_bs][:].long())
            loss_dice_2 = dice_loss(
                outputs_soft_2[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))



            # supervised_loss = 0.5 * (loss_dice + loss_ce)
            # supervised_loss = (0.7*loss_dice + 0.3*loss_ce+0.7*loss_dice_1 + 0.3*loss_ce_1+0.7*loss_dice_2 + 0.3*loss_ce_2)
            supervised_loss = (0.5*loss_ce+0.5*loss_dice+0.5*loss_dice_1+0.5*loss_ce_1+0.5*loss_dice_2+0.5*loss_ce_2)
            consistency_weight = get_current_consistency_weight(iter_num//150)
            consistency_dist = losses.softmax_mse_loss(#------------todo-------dice_loss_________
                outputs[args.labeled_bs:], ema_output)  # (batch, 2, 112,112,80)

            threshold = (0.75+0.25*ramps.sigmoid_rampup(iter_num,max_iterations))*np.log(2)
            threshold_high_level = (0.78 + 0.25 * ramps.sigmoid_rampup(iter_num,#高级的不确定性有更高的阈值
                                                            max_iterations)) * np.log(2)

            mask = (uncertainty < threshold).float()

            #print(mask.shape)#torch.Size([12, 1, 256, 256])
            #conv_layer_1 = torch.nn.Conv2d(4,1,kernel_size=1).cuda()
            #outputs = conv_layer_1(outputs)
            #print(outputs.shape)#torch.Size([24, 1, 256, 256])

            uncertainty_sample=torch.matmul(mask,unlabeled_volume_batch)#todo------------数据增强新的样本---------------



            with torch.no_grad():
                ema_output_high_level = model2(uncertainty_sample)#todo---------高级教师的输出是ema_output_high_level--------
            #print(ema_output_high_level.shape)##torch.Size([12, 4, 256, 256])
            # print(uncertainty_sample.shape)
            # print(ema_output.shape)
            #ema_output_high_level = conv_layer_1(ema_output_high_level)#torch.Size([12, 1, 256, 256])



            _, _, w2, h2 = uncertainty_sample.shape
            preds_2 = torch.zeros([stride * T, num_classes, w2, h2]).cuda()
            volume_batch_r_high_level = uncertainty_sample.repeat(2, 1, 1, 1)
            for i in range(T//2):
                uncertainty_sample_high = volume_batch_r_high_level + \
                    torch.clamp(torch.randn_like(#torch.randn_like返回一个与输入张量大小相同的张量，其中填充了均值为 0 方差为 1 的正态分布的随机值。
                        volume_batch_r_high_level) * 0.1, -0.2, 0.2)#clamp（）函数的功能将输入input张量每个元素的值压缩到区间 [min,max]，并返回结果到一个新张量
                with torch.no_grad():
                    preds_2[2 * stride * i:2 * stride *
                          (i + 1)] = model2(uncertainty_sample_high)
            preds_2 = F.softmax(preds_2, dim=1)
            preds_2 = preds_2.reshape(T, stride, num_classes, w, h)
            preds_2 = torch.mean(preds_2, dim=0)
            uncertainty_2 = -1.0 * torch.sum(preds_2 * torch.log(preds_2 + 1e-6), dim=1, keepdim=True)
            mask_2 = (uncertainty_2 < threshold_high_level).float()

            #conv_layer_2 = torch.nn.Conv2d(1, 4, kernel_size=1).cuda()
            #outputs = conv_layer_2(outputs)
            #ema_output_high_level = conv_layer_2(ema_output_high_level)

            consistency_dist_high_level = losses.softmax_mse_loss(
                # todo-----------------------低级教师与高级教师之间的loss------------dice_loss-----
                ema_output_high_level, ema_output)

            consistency_dist_high_level_student = losses.softmax_mse_loss(#todo-------dice_loss_________
                outputs[args.labeled_bs:], ema_output_high_level)

            consistency_loss = torch.sum(mask_2 * consistency_dist)/(2 * torch.sum(mask_2)+1e-16)

            consistency_loss_high_level = torch.sum(
                mask_2*consistency_dist_high_level)/(2*torch.sum(mask_2)+1e-16)

            #mask_3 = torch.add(mask,mask_2)

            consistency_loss_student_high_level_teacher = torch.sum(mask_2 * consistency_dist_high_level_student)/((2 * torch.sum(mask_2)+1e-16))

            #r_drop_loss = losses.compute_kl_loss(outputs[args.labeled_bs:], outputs_1[args.labeled_bs:]) + losses.compute_kl_loss(outputs_1[args.labeled_bs:], outputs_2[args.labeled_bs:])

            loss = supervised_loss +  consistency_loss + consistency_loss_high_level+ consistency_loss_student_high_level_teacher

            update_ema_variables_plus(model, model2,ema_model, args.ema_decay, iter_num)
            # update_ema_variables_plus(ema_model, ema_model_high_level, args.ema_decay, iter_num)

            optimizer.zero_grad()
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()

            loss.backward()

            optimizer.step()
            optimizer_1.step()
            optimizer_2.step()


            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/consistency_loss',
                              consistency_loss, iter_num)
            writer.add_scalar('info/consistency_weight',
                              consistency_weight, iter_num)
            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)





            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)
                    writer.add_scalar('info/val_{}_asd'.format(class_i + 1),
                                      metric_list[class_i, 2], iter_num)
                    writer.add_scalar('info/val_{}_hd'.format(class_i + 1),
                                      metric_list[class_i, 3], iter_num)
                    writer.add_scalar('info/val_{}_jc'.format(class_i + 1),
                                      metric_list[class_i, 4], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                mean_hd95 = np.mean(metric_list, axis=0)[1]
                mean_asd = np.mean(metric_list, axis=0)[2]
                mean_hd = np.mean(metric_list, axis=0)[3]
                mean_jc = np.mean(metric_list, axis=0)[4]

                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)
                writer.add_scalar('info/val_mean_asd', mean_asd, iter_num)
                writer.add_scalar('info/val_mean_hd', mean_hd, iter_num)
                writer.add_scalar('info/val_mean_jc', mean_jc, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f mean_asd: %f mean_hd: %f mean_jc: %f' % (iter_num, performance, mean_hd95,mean_asd,mean_hd,mean_jc))
                model.train()








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
                                             '{}_best_ema_model.pth'.format(args.model))
                    torch.save(ema_model.state_dict(), save_mode_path)
                    torch.save(ema_model.state_dict(), save_best)

                logging.info(
                    'iteration %d : model2_mean_dice : %f model2_mean_hd95 : %f model2_mean_asd: %f model2_mean_hd: %f model2_mean_jc: %f' % (
                        iter_num, performance2, mean_hd952, mean_asd2, mean_hd2, mean_jc2))
                ema_model.train()







                model2.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model2, classes=num_classes)
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

                performance3 = np.mean(metric_list, axis=0)[0]
                mean_hd953= np.mean(metric_list, axis=0)[1]
                mean_asd3 = np.mean(metric_list, axis=0)[2]
                mean_hd3 = np.mean(metric_list, axis=0)[3]
                mean_jc3 = np.mean(metric_list, axis=0)[4]

                writer.add_scalar('info/ema_model_high_val_mean_dice', performance3, iter_num)
                writer.add_scalar('info/ema_model_high_val_mean_hd95', mean_hd953, iter_num)
                writer.add_scalar('info/ema_model_high_val_mean_asd', mean_asd3, iter_num)
                writer.add_scalar('info/ema_model_high_val_mean_hd', mean_hd3, iter_num)
                writer.add_scalar('info/ema_model_high_val_mean_jc', mean_jc3, iter_num)

                if performance3 > best_performance3:
                    best_performance3 = performance3
                    save_mode_path = os.path.join(snapshot_path,
                                                  'ema_model_high_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance3, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_ema_high_model.pth'.format(args.model))
                    torch.save(model2.state_dict(), save_mode_path)
                    torch.save(model2.state_dict(), save_best)

                logging.info(
                    'iteration %d : ema_high_model_dice : %f ema_high_model_mean_hd95 : %f ema_high_model_mean_asd: %f ema_high_model_mean_hd: %f ema_high_model_mean_jc: %f' % (
                        iter_num, performance3,mean_hd953, mean_asd3, mean_hd3, mean_jc3))
                model2.train()







            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"

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

    snapshot_path = "../model/{}_{}_labeled/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
