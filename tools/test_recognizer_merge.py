import argparse

import torch
import mmcv
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict
from mmcv.parallel import scatter, collate, MMDataParallel

from mmaction import datasets
from mmaction.datasets import build_dataloader
from mmaction.models import build_recognizer, recognizers
from mmaction.core.evaluation.accuracy import (softmax, top_k_accuracy,
                                               mean_class_accuracy)


def parse_args():
    parser = argparse.ArgumentParser(description='Test an action recognizer')
    parser.add_argument('config', help='test config file path')    
    parser.add_argument('rgb', help='file containing rgb results')
    parser.add_argument('flow', help='file containing flow results')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--use_softmax', action='store_true',
                        help='whether to use softmax score')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)

    dataset = obj_from_dict(cfg.data.test, datasets, dict(test_mode=True))

    rgb_results = args.rgb # "/home/lixun/Desktop/HAR/mmaction/work_dirs/predictions_hmdb51_split1/rgb.pkl"
    flow_results = args.flow #  "/home/lixun/Desktop/HAR/mmaction/work_dirs/predictions_hmdb51_split1/flow.pkl"
    outputs_rgb = mmcv.load(rgb_results)
    outputs_flow = mmcv.load(flow_results)

    outputs = [outputs_rgb[i] + outputs_flow[i] for i in range(len(outputs_rgb))]

    if args.out:
        print('writing results to {}'.format(args.out))
        mmcv.dump(outputs, args.out)

    gt_labels = []
    for i in range(len(dataset)):
        ann = dataset.get_ann_info(i)
        gt_labels.append(ann['label'])

    if args.use_softmax:
        print("Averaging score over {} clips with softmax".format(
            outputs[0].shape[0]))
        results = [softmax(res, dim=1).mean(axis=0) for res in outputs]
    else:
        print("Averaging score over {} clips without softmax (ie, raw)".format(
            outputs[0].shape[0]))
        results = [res.mean(axis=0) for res in outputs]
    top1, top5 = top_k_accuracy(results, gt_labels, k=(1, 5))
    mean_acc = mean_class_accuracy(results, gt_labels)
 
    print("Mean Class Accuracy = {:.02f}".format(mean_acc * 100))
    print("Top-1 Accuracy = {:.02f}".format(top1 * 100))
    print("Top-5 Accuracy = {:.02f}".format(top5 * 100))


if __name__ == '__main__':
    main()
