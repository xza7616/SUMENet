#os 库是Python标准库，包含几百个函数，常用的有路径操作、进程管理、环境参数等。
import os
#高级的 文件、文件夹、压缩包 处理模块
import shutil
#JSON(JavaScript Object Notation, JS 对象简谱) 是一种轻量级的数据交换格式。
import json
import time
#加速
# from apex import amp
from torch.cuda import amp
import tqdm
# import apex
import numpy as np
#分布式通信包
import torch.distributed as dist

import torch
import torch.nn as nn
import torch.nn.functional as F
#寻找最适合当前配置的高效算法，来达到优化运行效率的问题
import torch.backends.cudnn as cudnn
#调整学习率（learning rate）的方法

from torch.optim.lr_scheduler import LambdaLR, StepLR
#实现自由的数据读取,dataloadateset读取训练集dataset (Dataset): 加载数据的数据集
# * batch_size (int, optional): 每批加载多少个样本
# * shuffle (bool, optional): 设置为“真”时,在每个epoch对数据打乱.（默认：False）
# * sampler (Sampler, optional): 定义从数据集中提取样本的策略,返回一个样本
# * batch_sampler (Sampler, optional): like sampler, but returns a batch of indices at a time 返回一批样本. 与atch_size, shuffle, sampler和 drop_last互斥.
# * num_workers (int, optional): 用于加载数据的子进程数。0表示数据将在主进程中加载​​。（默认：0）
# * collate_fn (callable, optional): 合并样本列表以形成一个 mini-batch.  #　callable可调用对象
# * pin_memory (bool, optional): 如果为 True, 数据加载器会将张量复制到 CUDA 固定内存中,然后再返回它们.
# * drop_last (bool, optional): 设定为 True 如果数据集大小不能被批量大小整除的时候, 将丢掉最后一个不完整的batch,(默认：False).
# * timeout (numeric, optional): 如果为正值，则为从工作人员收集批次的超时值。应始终是非负的。（默认：0）
# * worker_init_fn (callable, optional): If not None, this will be called on each worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as input, after seeding and before data loading. (default: None)．

#from toolbox.datasets.nyuv2 import train_collate_fn
#from lib.data_fetcher import DataPrefetcher
from torch.utils.data import DataLoader

from toolbox import MscCrossEntropyLoss
from toolbox.loss import lovaszSoftmax
from toolbox import get_dataset
from toolbox import get_logger
from toolbox import get_model
# from toolbox import get_model_t
from toolbox import averageMeter, runningScore
from toolbox import ClassWeight, save_ckpt,load_ckpt
# from KD_loss.ContrastiveSeg.lib.loss.loss_contrast import PixelContrastLoss
from toolbox import Ranger
# from toolbox.kdlosses import *
torch.manual_seed(123)
#程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。
cudnn.benchmark = True

def run(args):
#载configs下的配置文件
    with open(args.config, 'r') as fp:
        cfg = json.load(fp)
    #用于保存日志文件或其他的与时间相关的数据
    logdir = f'run2/{time.strftime("%Y-%m-%d-%H-%M")}-NO-UWCAdapter + CUAM+(NIG偶然+HAE认知) + DSIR + DSIR + UMoE '

    args.logdir = logdir

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    #将源文件路径复制到logdir
    shutil.copy(args.config, logdir)

    #方便调试维护代码
    logger = get_logger(logdir)
    if args.local_rank == 0:
        logger.info(f'Conf | use logdir {logdir}')

    model = get_model(cfg)
    # model.initialize_weights()
    # model._load_resnet_pretrained()
    # model._freeze_parameters()
    # model._unfreeze_bn()
    # model.load_pre('/media/yuride/date/XZA/Pretrain/mit_b2.pth')
    # model.load_pre('/media/yuride/date/XZA/toolbox/models/PATNet/saved_models/p2t_base.pth')
    # model.load_state_dict(torch.load('/home/hjk/桌面/mymodel_new(pre change)/toolbox/models/CIRNet/CIRNet_R50.pth'), strict=False)
    print('****************student_PTH loading Finish!*************')
    #将get_dataset返回的对象分别传给train、test
    trainset, *testset = get_dataset(cfg)
#torch.device代表将torch.Tensor分配到的设备的对象

    device = torch.device('cuda:0')
    args.distributed = False
#environ是一个字符串所对应环境的映像对象
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        if args.local_rank == 0:
            print(f"WORLD_SIZE is {os.environ['WORLD_SIZE']}")

    train_sampler = None
    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()

        # model = apex.parallel.convert_syncbn_model(model)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)

    model.to(device)
    # teacher.to(device)
    train_loader = DataLoader(trainset, batch_size=cfg['ims_per_gpu'], shuffle=(train_sampler is None),
                              num_workers=cfg['num_workers'], pin_memory=True, sampler=train_sampler, drop_last=True)
    #                                             drop_last=True解决照片留单然后导致batch变成1
    val_loader = DataLoader(testset[0], batch_size=1, shuffle=False,num_workers=cfg['num_workers'],pin_memory=True, drop_last=True)
    # ── 参数分组：Adapter 用更大 weight_decay 抑制过拟合 ──────────
    # 日志分析：UWCA 在小数据集上过拟合 gap 比原 Adapter 大 0.05
    # 对 Adapter 参数加 5x weight_decay 是最直接的正则手段
    adapter_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'adapter' in name:
            adapter_params.append(p)
        else:
            other_params.append(p)
    params_list = [
        {'params': other_params,  'weight_decay': cfg['weight_decay']},
        {'params': adapter_params,'weight_decay': cfg['weight_decay'] * 5},  # Adapter 更强正则
    ]
    # ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.SGD(params_list, lr=cfg['lr_start'], weight_decay=cfg['weight_decay'], momentum=cfg['momentum'])
    Scaler = amp.GradScaler()
    #optimizer = Ranger(params_list, lr=cfg['lr_start'], weight_decay=cfg['weight_decay']
    scheduler = LambdaLR(optimizer, lr_lambda=lambda ep: (1 - ep / cfg['epochs']) ** 0.9)

    # model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)
    # if args.distributed:
    #   model = torch.nn.parallel.DistributedDataParallel(model)

    # class weight 计算
    if hasattr(trainset, 'class_weight'):
        print('using classweight in dataset')
        class_weight = trainset.class_weight
    else:
        classweight = ClassWeight(cfg['class_weight'])
        class_weight = classweight.get_weight(train_loader, cfg['n_classes'])

    class_weight = torch.from_numpy(class_weight).float().to(device)
    # print(class_weight)
    # class_weight[cfg['id_unlabel']] = 0

    # 损失函数 & 类别权重平衡 & 训练时ignore unlabel
    criterion = MscCrossEntropyLoss(weight=class_weight).to(device)
    # contrastive = PixelContrastLoss().to(device)
    # criterion = MscCrossEntropyLoss().to(device)
    # criterion = lovaszSoftmax(ignore_index=0).to(device)

    # ── 辅助损失配置 ──────────────────────────────────────────────
    # warmup_epochs: 前N个epoch只用主损失让模型先收敛，之后再引入辅助损失
    # 这样避免模型还没学到基本特征时辅助损失的梯度噪声干扰主任务
    aux_warmup_epochs = max(5, cfg['epochs'] // 10)
    w_nig    = 1e-3
    w_router = 1e-4    # 同时作用于 RGB 和 Depth 两个路由损失
    w_expert = 1e-4
    # ─────────────────────────────────────────────────────────────

    # 指标 包含unlabel
    train_loss_meter = averageMeter()
    val_loss_meter = averageMeter()
    # running_metrics_val = runningScore(cfg['n_classes'], ignore_index=cfg['id_unlabel'])
    running_metrics_val = runningScore(cfg['n_classes'], ignore_index=None)
    # 每个epoch迭代循环

    flag = True #为了先保存一次模型做的判断
    #设置一个初始miou
    miou = 0
    for ep in range(cfg['epochs']):
        if args.distributed:
            train_sampler.set_epoch(ep)

        # training
        model.train()
        train_loss_meter.reset()
        # teacher.eval()

        for i, sample in enumerate(train_loader):
            optimizer.zero_grad()  # 梯度清零

            ################### train edit #######################
            depth = sample['depth'].to(device)
            image = sample['image'].to(device)
            label = sample['label'].to(device)
            # print("image:", image.dtype)

            # print("label:", label.shape)

            with amp.autocast(False):

                # predict = model(torch.cat((image.unsqueeze(2), depth.unsqueeze(2)), dim=2))

                # predict = model(torch.squeeze(image, dim=1), depth)
                predict = model(image,depth)[0]
                # print('predict', predict.dtype)
                # loss = criterion(predict[0], label) + criterion(predict[1], label) + criterion(predict[2], label)

                seg_loss = criterion(predict, label)

                # ── 辅助损失（warmup后才启用）────────────────────────────
                if ep >= aux_warmup_epochs:
                    aux = model.get_aux_losses()
                    nig_loss    = aux.get('nig_reg',              torch.tensor(0.0, device=device))
                    # 双流路由各自一个负载均衡损失，取均值
                    router_rgb  = aux.get('router_balance_rgb',   torch.tensor(0.0, device=device))
                    router_dep  = aux.get('router_balance_depth', torch.tensor(0.0, device=device))
                    router_loss = (router_rgb + router_dep) / 2.0
                    expert_loss = aux.get('expert_balance',       torch.tensor(0.0, device=device))

                    nig_loss    = torch.clamp(nig_loss,              0.0, 10.0)
                    router_loss = torch.clamp(router_loss.abs(),     0.0, 10.0)
                    expert_loss = torch.clamp(expert_loss.abs(),     0.0, 10.0)

                    loss = (seg_loss
                            + w_nig    * nig_loss
                            + w_router * router_loss
                            + w_expert * expert_loss)
                else:
                    loss = seg_loss
                # ────────────────────────────────────────────────────────

            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()

            Scaler.scale(loss).backward()
            Scaler.step(optimizer)
            Scaler.update()
            # optimizer.step()

            if args.distributed:
                reduced_loss = loss.clone()
                dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
                reduced_loss /= args.world_size
            else:
                reduced_loss = loss
            train_loss_meter.update(reduced_loss.item())

        scheduler.step(ep)
        # ── CLIR 温度退火（每个epoch调用一次）──────────────────────
        model.layer_router.anneal_tau()

        # val
        with torch.no_grad():
            model.eval()
            running_metrics_val.reset()

            val_loss_meter.reset()
            ################### val edit #######################
            for i, sample in enumerate(val_loader):
                depth = sample['depth'].to(device)
                image = sample['image'].to(device)
                label = sample['label'].to(device)

                # predict = model(torch.cat((image.unsqueeze(2), depth.unsqueeze(2)), dim=2))
                # loss = criterion(predict, label)
                predict = model(image,depth)[0]
                # predict = model(image)
                # print("label", label)
                # loss = criterion(predict[0], label)

                loss = criterion(predict, label)    #############################2

                val_loss_meter.update(loss.item())    ##########################

                # predict = predict[0].cpu().numpy()  # [1, h, w]     #############################3
                # print('predict',predict.shape)
                # predict = predict.max(1)[1].cpu().numpy()  # [1, h, w]
                predict = predict.max(1)[1].cpu().numpy()  # [1, h, w]
                # print('predict',predict.shape)``````````````````````````
                label = label.cpu().numpy()

            ###################edit end#########################
                running_metrics_val.update(label, predict)

        if args.local_rank == 0:
            # ── 打印辅助损失数值，方便监控是否异常 ──────────────────
            if ep >= aux_warmup_epochs:
                aux_log = model.get_aux_losses()
                aux_str = (
                    f' | nig={aux_log.get("nig_reg", torch.tensor(0)).item():.4f}'
                    f' r_rgb={aux_log.get("router_balance_rgb",   torch.tensor(0)).item():.4f}'
                    f' r_dep={aux_log.get("router_balance_depth", torch.tensor(0)).item():.4f}'
                    f' expert={aux_log.get("expert_balance", torch.tensor(0)).item():.4f}'
                )
            else:
                aux_str = f' | aux warmup ({ep+1}/{aux_warmup_epochs})'
            # ────────────────────────────────────────────────────────
            logger.info(
                 f'Iter | [{ep + 1:3d}/{cfg["epochs"]}] train/val loss={train_loss_meter.avg:.5f}/{val_loss_meter.avg:.5f} '
                # f', PA={running_metrics_val.get_scores()[0]["pixel_acc: " ]:.3f}'
                # f', CA={running_metrics_val.get_scores()[0]["class_acc: " ]:.3f}'
                f', mAcc={running_metrics_val.get_scores()[0]["mAcc: "]:.3f}'
                f', miou={running_metrics_val.get_scores()[0]["mIou: "]:.3f}'
                f', best_miou={miou:.3f}'
                f'{aux_str}')
            save_ckpt(logdir, model, kind='end')
            newmiou = running_metrics_val.get_scores()[0]["mIou: "]

            if newmiou > miou:
                save_ckpt(logdir, model, kind='best')  #消融可能不一样
                miou = newmiou

            if newmiou > 0.780:
                # save_ckpt(logdir, model, kind=f'{newmiou:.3f}')
                state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                torch.save(state, os.path.join(logdir, f'{newmiou:.3f}.ph'))

    save_ckpt(logdir, model, kind='end')  #保存最后一个模型参数

if __name__ == '__main__':


    import argparse

    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/SUIM.json",
        help="Configuration file to use",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--opt_level",
        type=str,
        default='O1',
    )

    args = parser.parse_args()

    run(args)
