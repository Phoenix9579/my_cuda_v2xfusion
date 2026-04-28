# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#训练入口，dist-run -np控制GPU数量
import argparse
import copy
import os
import random
import time

import numpy as np
import torch
from mmcv import Config #加载配置文件 读取YAML配置
from torchpack import distributed as dist #分布式训练
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs

from mmdet3d.apis import train_model #训练模型
from mmdet3d.datasets import build_dataset #构建数据集
from mmdet3d.models import build_model  #构建模型
from mmdet3d.utils import get_root_logger, convert_sync_batchnorm, recursive_eval
import tinyq

def main():
    tinyq.set_verbose() #启用稀疏性训练的详细日志
    use_dist = 1 #启用分布式训练 0表示单机训练
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    parser.add_argument("--run-dir", metavar="DIR", help="run directory")    
    parser.add_argument("--mode",  type=str, choices=["dense", "sparsity"], help="Training  use dense or sparsity mode.")
    args, opts = parser.parse_known_args()
    print(args)
    configs.load(args.config, recursive=True) #递归加载配置文件
    configs.update(opts) #更新命令行覆盖的配置项

    cfg = Config(recursive_eval(configs), filename=args.config)
    if use_dist:
        dist.init() #初始化分布式环境（设置 NCCL / GLOO 后端、端口、rank 等）
        torch.backends.cudnn.benchmark = cfg.cudnn_benchmark #开启 cuDNN 自动算法搜索；固定输入尺寸时加速卷积，可变尺寸时反而变慢。
        torch.cuda.set_device(dist.local_rank()) #把当前进程绑定到本机第 local_rank 号 GPU，避免多进程抢同一张卡
 
    if args.run_dir is None: #命令行没输出，自行设置输出路径
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)
    cfg.run_dir = args.run_dir

    # dump config
    cfg.dump(os.path.join(cfg.run_dir, "configs.yaml")) #保存当前配置到文件

    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime()) #日志名，年月——时分秒
    log_file = os.path.join(cfg.run_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file)

    # log some basic info
    logger.info(f"Config:\n{cfg.pretty_text}") #把复杂的配置对象（通常为 mmcv.Config）转成缩进对齐、易读的多行字符串
    # 把这段文本写到 终端 + 日志文件，方便以后查看

    # set random seeds
    if cfg.seed is not None:
        logger.info(
            f"Set random seed to {cfg.seed}, "
            f"deterministic mode: {cfg.deterministic}"
        )
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if cfg.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    datasets = [build_dataset(cfg.data.train)] #创建训练数据集

    model = build_model(cfg.model) #根据配置组装模型
    model.find_unused_parameters= False  #忽略这些未使用的参数，防止报错
    model.init_weights() #初始化模型权重 比如 kaiming_normal_ 或 xavier_normal_
    if cfg.get("sync_bn", None): #如果 sync_bn 是 None，就不会进入 BN 转换流程
        if not isinstance(cfg["sync_bn"], dict): #如果 cfg["sync_bn"] 不是字典，比如只是 True，那么这行代码会将其设置为一个默认字典：{"exclude": []}
            cfg["sync_bn"] = dict(exclude=[]) #exclude 是一个可选项，表示哪些层不需要转换为 SyncBN
        model = convert_sync_batchnorm(model, exclude=cfg["sync_bn"]["exclude"]) # PyTorch 提供的转换函数
        
    if args.mode == 'sparsity':
        print('use sparsity mode')
        tinyq.replace_sparsity_modules(model) #替换为稀疏模块
        
    logger.info(f"Model:\n{model}")
    train_model(
        model,
        datasets,
        cfg,
        distributed=use_dist,
        validate=True,
        timestamp=timestamp,
    ) #调用训练接口

if __name__ == "__main__":
    main()
