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

import argparse
import copy
import os
import warnings
import sys
import mmcv
import torch
from torchpack.utils.config import configs
from torchpack import distributed as dist
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.utils import recursive_eval
from mmdet3d.datasets.v2x_dataset import collate_fn
from functools import partial
from torch import nn
import quantize as quantize
import tinyq
# PTQ（Post-Training Quantization，训练后量化）处理，目的是压缩模型大小、加速推理速度，同时保持检测精度

def quantize_net(model): #对模型的 LiDAR 编码器主干网络、摄像头编码器 和 解码器 进行量化
    print("🔥 start quantization 🔥 ")
    quantize.quantize_encoders_lidar_branch(model.encoders.lidar.backbone)    
    quantize.quantize_encoders_camera_branch(model.encoders.camera)
    # quantize.replace_to_quantization_module(model.fuser)
    quantize.quantize_decoder(model.decoder)
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--mode",  type=str, choices=["dense", "sparsity"], help="The checkpoint is in dense or sparsity mode.")
    parser.add_argument("--out", help="output result file in pickle format")
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase"
        "the inference speed",
    )
    parser.add_argument(
        "--format-only",
        action="store_true",
        help="Format the output results without perform evaluation. It is"
        "useful when you want to format the result to a specific format and "
        "submit it to the test server",
    )
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC',
    )
    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument("--show-dir", help="directory where results will be saved")
    parser.add_argument(
        "--gpu-collect",
        action="store_true",
        help="whether to use gpu to collect results.",
    )
    parser.add_argument(
        "--tmpdir",
        help="tmp directory used for collecting results from multiple "
        "workers, available when gpu-collect is not specified",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function (deprecate), "
        "change to --eval-options instead.",
    )
    parser.add_argument(
        "--eval-options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            "--options and --eval-options cannot be both specified, "
            "--options is deprecated in favor of --eval-options"
        )
    if args.options:
        warnings.warn("--options is deprecated in favor of --eval-options")
        args.eval_options = args.options
    return args


def main():
    
    import sys
    import site
    print("sys.path:")
    for p in sys.path:
        print(p)

    print("\nUser site:", site.getusersitepackages())
    print("User dir exists?", __import__('os').path.exists('/root/.local'))
    
    args = parse_args() #参数解析，根据命令行参数得到解析后的参数对象 
    #dist.init()
    torch.backends.cudnn.benchmark = True #benchmark模式以优化性能
    torch.cuda.set_device(dist.local_rank()) #根据分布式训练环境中的本地rank设置当前使用的GPU设备
    
    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)
    print(cfg) #加载配置文件

    if args.cfg_options is not None: #如果有额外的配置选项，则将其合并到主配置中
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False): #根据配置决定是否开启cuDNN的benchmark模式
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None  #清除模型预训练权重设置，防止使用预训练模型
    # in case the test dataset is concatenated
    samples_per_gpu = 1 #设置每个GPU上的样本数，并调整数据加载器和数据管道的配置
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True #新添加的参数，用于控制测试数据集的测试模式.控制测试阶段的行为，无需修改原始YAML文件
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.test]
        )
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    distributed = 0 #定义是否启用分布式训练环境。这里设为0表示不启用。

    # set random seeds
    if args.seed is not None:  #如果设置了随机种子，则使用该种子初始化随机数生成器，确保实验可复现
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset_train  = build_dataset(cfg.data.train)
    data_loader_train = build_dataloader(
        dataset_train,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )
    data_loader_train.collate_fn = partial(collate_fn,is_return_depth=False) #collate_fn定义如何将多个样本（sample）组合成一个批次
    #  标准库 functools 中的函数，用于固定一个函数的部分参数，生成一个新的函数
    
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )
    data_loader.collate_fn = partial(collate_fn,is_return_depth=False) #手动将抽取出的样本堆叠起来的函数
    # build the model and load checkpoint
    cfg.model.train_cfg = None #清除train_cfg参数
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg")) #创建模型实例

    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, strict= True, map_location="cpu")
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model) #融合Conv与BN层

    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else: #优先从检查点提取类别标签（如object_classes配置的10类目标），否则从数据集获取，确保模型输出与标注数据一致
        model.CLASSES = dataset.classes
    
    model = model.cuda()
    if args.mode == 'sparsity':
        tinyq.replace_sparsity_modules(model) #稀疏性处理
        tinyq.slinker(model).recompute_mask
        tinyq.slinker(model).sparsify_weight
        model = tinyq.remove_sparsity_modules(model)
    #model = quantize_net(model) #模型量化 属于量化参数训练阶段，不是真正的完成量化。
    
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])  
        print("🔥 start calibrate 🔥 ")
        quantize.set_quantizer_fast(model)
        quantize.calibrate_model(model, data_loader_train, 0, None, 10)
        quantize.print_quantizer_status(model)
        outputs = single_gpu_test(model, data_loader) #已经得到了推理结果在outputs中
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)
    
    torch.save(model, 'ptq.pth') #保存经过量化的模型，并在主进程中评估模型的性能
    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f"\nwriting results to {args.out}",f"——from:{os.path.basename(__file__)}")
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:  #只做格式转换（如转 KITTI、COCO）而不算指标。
            dataset.format_results(outputs, **kwargs)
        if 1:
            eval_kwargs = cfg.get("evaluation", {}).copy() #真正的评估参数
            # hard-code way to remove EvalHook args
            for key in [
                "interval",
                "tmpdir",
                "start",
                "gpu_collect",
                "save_best",
                "rule", #去掉训练阶段用的钩子函数
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(f"——from:{os.path.basename(__file__)}")
            print(dataset.evaluate(outputs)) #计算指标并打印结果，传入推理结果开始评估

if __name__ == "__main__":
    main()