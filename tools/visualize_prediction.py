# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import numpy as np
import os
from itertools import chain
import cv2
from PIL import Image
import torch

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.data.build import filter_images_with_few_keypoints
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from src import data
from src.engine import default_argument_parser, DefaultTrainer


def setup(args):
    cfg = get_cfg()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def parse_args(in_args=None):
    parser = argparse.ArgumentParser(description="Visualize ground-truth data")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--output-dir", default="./", help="path to output directory")
    parser.add_argument("--show", action="store_true", help="show output in a window")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args(in_args)


if __name__ == "__main__":
    args = parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup(args)
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    dirname = args.output_dir
    os.makedirs(dirname, exist_ok=True)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    # if args.eval_only:
    #     model = DefaultTrainer.build_model(cfg)
    #     DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
    #         cfg.MODEL.WEIGHTS, resume=args.resume
    #     )
    #     res = Trainer.test(cfg, model)
    #     if comm.is_main_process():
    #         verify_results(cfg, res)
    #     if cfg.TEST.AUG.ENABLED:
    #         res.update(Trainer.test_with_TTA(cfg, model))
    #     return res


    def output(vis, fname):
        if args.show:
            print(fname)
            cv2.imshow("window", vis.get_image()[:, :, ::-1])
            cv2.waitKey()
        else:
            filepath = os.path.join(dirname, fname)
            print("Saving to {} ...".format(filepath))
            vis.save(filepath)

    # model = DefaultTrainer.build_model(cfg)
    # DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
    #     cfg.MODEL.WEIGHTS, resume=args.resume
    # )
    predictor = DefaultPredictor(cfg)

    # scale = 2.0 if args.show else 1.0
    # train_data_loader = build_detection_train_loader(cfg)
    # for batch in train_data_loader:
    #     for per_image in batch:
    #         # Pytorch tensor is in (C, H, W) format
    #         img = per_image["image"].permute(1, 2, 0).numpy()
    #         prediction = predictor(img)
    #         if cfg.INPUT.FORMAT == "BGR":
    #             img = img[:, :, [2, 1, 0]]
    #         else:
    #             img = np.asarray(Image.fromarray(img, mode=cfg.INPUT.FORMAT).convert("RGB"))
    #
    #         vis = Visualizer(img, metadata=metadata, scale=scale)
    #         vis = vis.draw_instance_predictions(prediction['instances'].to('cpu'))
    #         output(vis, str(per_image["image_id"]) + ".jpg")
    cur_idx = 0
    correct = [0] * 8
    total_instance_num = [0] * 8

    scale = 2.0 if args.show else 1.0
    train_data_loader = build_detection_train_loader(cfg)
    for batch in train_data_loader:
        for per_image in batch:
            # Pytorch tensor is in (C, H, W) format
            img = per_image["image"].permute(1, 2, 0).numpy()
            gt_proposal = per_image['instances']
            gt_proposal.set('proposal_boxes', gt_proposal.gt_boxes)
            prediction = predictor(img, gt_proposal)
            # import pdb; pdb.set_trace()
            pred_classes = torch.argmax(prediction, dim=1)
            for idx, gt in enumerate(gt_proposal.gt_classes):
                total_instance_num[gt] += 1
                if pred_classes[idx] == gt:
                    correct[gt] += 1
                cur_idx += 1
        if cur_idx % 10 == 0:
            print(cur_idx)
        if cur_idx > 100000:
            break
    import pdb; pdb.set_trace()

    for c_idx, crr in enumerate(correct):
        print('{}: {} / {}, {:.2f}%'.format(c_idx, correct[c_idx], total_instance_num[c_idx], correct[c_idx] / total_instance_num[c_idx] * 100))