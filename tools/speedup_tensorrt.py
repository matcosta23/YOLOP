import argparse
import copy
import json
import os, sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import pprint
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import numpy as np
from lib.utils import DataLoaderX
from tensorboardX import SummaryWriter

import lib.dataset as dataset
from lib.config import cfg
from lib.config import update_config
from lib.core.loss import get_loss
from lib.core.function import validate, validate_with_speedup_engine
from lib.core.general import fitness
from lib.models import get_net
from lib.utils.utils import create_logger, select_device, get_optimizer

from nni.compression.pytorch.quantization_speedup import ModelSpeedupTensorRT

def parse_args():
    parser = argparse.ArgumentParser(description='Test Multitask Network with Post-Training Quantization '\
                                                 'and SpeedUp with TensorRT.')

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='runs/')
    parser.add_argument('--weights', nargs='+', type=str, default='/data2/zwt/wd/YOLOP/runs/BddDataset/detect_and_segbranch_whole/epoch-169.pth', help='model.pth path(s)')
    parser.add_argument('--conf_thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('-e', '--eval_original', action='store_true', help="If set, original model without optimization is also evaluated.")
    args = parser.parse_args()

    return args

def main():
    # set all the configurations
    args = parse_args()
    update_config(cfg, args)

    # TODO: handle distributed training logger
    # set the logger, tb_log_dir means tensorboard logdir

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, cfg.LOG_DIR, 'test')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    print("\n----- BEGIN TO BUILD UP THE MODEL -----")
    # start_time = time.time()
    # DP mode
    device = select_device(logger, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU* len(cfg.GPUS)) if not cfg.DEBUG \
        else select_device(logger, 'cpu')

    model = get_net(cfg)
    
    # define loss function (criterion) and optimizer
    criterion = get_loss(cfg, device=device)
    #optimizer = get_optimizer(cfg, model)

    ### Load checkpoint model

    model_dict = model.state_dict()
    # NOTE: Since 'weights' is a list, we need to read the element.
    checkpoint_file = args.weights[0]
    logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
    # NOTE: Adaption for CPU execution.
    checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
    checkpoint_dict = checkpoint['state_dict']
    model_dict.update(checkpoint_dict)
    model.load_state_dict(model_dict)
    logger.info("=> loaded checkpoint '{}' ".format(checkpoint_file))

    model = model.to(device)
    model.gr = 1.0
    model.nc = 1
    print("----- END OF THE MODEL BUILDING -----\n")

    """
    NOTE: Code to verify types of modules present in the model :
    modules_types = []
    modules_names = []
    for name, module in model.named_modules():
        modules_types.append(type(module).__name__)
        modules_names.append(name)
    modules_types = np.unique(modules_types)

    - Module types obtained:
        'BatchNorm2d', 'Bottleneck', 'BottleneckCSP', 'Concat', 'Conv',
        'Conv2d', 'Detect', 'Focus', 'Hardswish', 'LeakyReLU', 'MCnet',
        'MaxPool2d', 'ModuleList', 'SPP', 'Sequential', 'Upsample'
    """

    ### Create configure list.

    # Inspired by NNI's example, the following modules will be quantized.
    modules_to_quantize = ['Conv2d', 'LeakyReLU', 'MaxPool2d']

    # Create dictionaries with the same approach used above.
    conv_dict = {'weight_bits':8, 'input_bits':8}
    lr_mp_dict = {'output_bits':8}
    configure_list = {}
    for name, module in model.named_modules():
        if type(module).__name__ in modules_to_quantize[:1]:
            configure_list[name] = conv_dict
        elif type(module).__name__ in modules_to_quantize[2:]:
            configure_list[name] = lr_mp_dict

    ### Define dataloader.

    print("\n----- BEGIN TO LOAD DATA -----")
    # Data loading
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg=cfg,
        is_train=False,
        inputsize=cfg.MODEL.IMAGE_SIZE,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    valid_loader = DataLoaderX(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=False,
        collate_fn=dataset.AutoDriveDataset.collate_fn
    )
    print("----- DATA LOAD COMPLETE -----\n")

    ### Evaluate the original model if flag is set.
    if args.eval_original:
        print("\n----- EVALUATION OF THE ORIGINAL (NON OPTIMIZED) MODEL -----")

        epoch = 0 #special for test
        da_segment_results,ll_segment_results,detect_results, total_loss,maps, times = validate(
            epoch,cfg, valid_loader, valid_dataset, model, criterion,
            final_output_dir, tb_log_dir, writer_dict,
            logger, device
        )
        fi = fitness(np.array(detect_results).reshape(1, -1))
        msg =   'Test:    Loss({loss:.3f})\n' \
                'Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})\n' \
                        'Lane line Segment: Acc({ll_seg_acc:.3f})    IOU ({ll_seg_iou:.3f})  mIOU({ll_seg_miou:.3f})\n' \
                        'Detect: P({p:.3f})  R({r:.3f})  mAP@0.5({map50:.3f})  mAP@0.5:0.95({map:.3f})\n'\
                        'Time: inference({t_inf:.4f}s/frame)  nms({t_nms:.4f}s/frame)'.format(
                            loss=total_loss, da_seg_acc=da_segment_results[0],da_seg_iou=da_segment_results[1],da_seg_miou=da_segment_results[2],
                            ll_seg_acc=ll_segment_results[0],ll_seg_iou=ll_segment_results[1],ll_seg_miou=ll_segment_results[2],
                            p=detect_results[0],r=detect_results[1],map50=detect_results[2],map=detect_results[3],
                            t_inf=times[0], t_nms=times[1])
        logger.info(msg)

    ### Speed UP Model with TensorRT.
    print("\n----- BEGIN TO SPEED UP MODEL -----")
    input_shape = [valid_loader.batch_size, 3, *valid_loader.dataset.inputsize]

    # NOTE: The following lines only work in environments with GPU.
    # TODO: Verify if a 'DataLoaderX' type is accepted in the calib_data_loader.
    # TODO: Check if the batchsize should receive the total or per GPU.
    engine = ModelSpeedupTensorRT(model, input_shape, config=configure_list, calib_data_loader=valid_loader, batchsize=valid_loader.batch_size)
    engine.compress()
    print("----- SPEED UP FINISHED -----\n")

    ### Evaluate the model after quantization simulation.
    print("\n----- EVALUATION OF MODEL SPEEDED UP WITH TENSOR RT -----")

    if args.logDir[-1] == '/':
        args.logDir = os.path.dirname(args.logDir) + "_speedup/"
    else:
        args.logDir += "_speedup"
    update_config(cfg, args)
    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, cfg.LOG_DIR, 'test')

    # NOTE : The original 'validate' function has been updated.
    da_segment_results_spd,ll_segment_results_spd,detect_results_spd,total_loss_spd,maps_spd,times_spd = validate_with_speedup_engine(
        epoch, cfg, valid_loader, valid_dataset, model, engine, criterion,
        final_output_dir, tb_log_dir, writer_dict,
        logger, device
    )
    fi_spd  = fitness(np.array(detect_results_spd).reshape(1, -1))
    msg_spd = 'Test:    Loss({loss:.3f})\n' \
              'Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})\n' \
                      'Lane line Segment: Acc({ll_seg_acc:.3f})    IOU ({ll_seg_iou:.3f})  mIOU({ll_seg_miou:.3f})\n' \
                      'Detect: P({p:.3f})  R({r:.3f})  mAP@0.5({map50:.3f})  mAP@0.5:0.95({map:.3f})\n'\
                      'Time: inference({t_inf:.4f}s/frame)  nms({t_nms:.4f}s/frame)'.format(
                          loss=total_loss_spd, da_seg_acc=da_segment_results_spd[0],da_seg_iou=da_segment_results_spd[1],da_seg_miou=da_segment_results_spd[2],
                          ll_seg_acc=ll_segment_results_spd[0],ll_seg_iou=ll_segment_results_spd[1],ll_seg_miou=ll_segment_results_spd[2],
                          p=detect_results_spd[0],r=detect_results_spd[1],map50=detect_results_spd[2],map=detect_results_spd[3],
                          t_inf=times_spd[0], t_nms=times_spd[1])
    logger.info(msg_spd)

    # TODO : Inclusion exportation process.

    # Process done for Observers Quantizers.
    # os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # model_path = os.path.join(cfg.OUTPUT_DIR, "quantized_yolop.pth")
    # calibration_path = os.path.join(cfg.OUTPUT_DIR, cfg.DATASET.DATASET + '_calib.json')
    # calibration_config = quantizer.export_model(model_path, calibration_path)

    # calib_json = json.dumps(calibration_config)
    # json_file = open(calibration_path, 'w')
    # json_file.write(calib_json)
    # json_file.close()

    print("\n----- TEST FINISHED! -----")


if __name__ == '__main__':
    main()
    