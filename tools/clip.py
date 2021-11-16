# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import numpy as np
import os
from itertools import chain
import cv2
from PIL import Image

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.data.build import filter_images_with_few_keypoints
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from src import data
import torch
import clip


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

    dirname = args.output_dir
    os.makedirs(dirname, exist_ok=True)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    def output(vis, fname):
        if args.show:
            print(fname)
            cv2.imshow("window", vis.get_image()[:, :, ::-1])
            cv2.waitKey()
        else:
            filepath = os.path.join(dirname, fname)
            print("Saving to {} ...".format(filepath))
            vis.save(filepath)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, preprocess = clip.load('ViT-B/32', device)
    from torchvision.datasets import CIFAR100
    import torchvision.models as models
    model = models.resnet18(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    import torchvision.transforms as transforms

    tf = transforms.ToTensor()
    pil = transforms.ToPILImage()

    transform_test = transforms.Compose([
        # transforms.Pad(16, fill=0, padding_mode='constant'),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    cls_names = ['person', 'car', 'train', 'rider', 'truck', 'motorcycle', 'bicycle', 'bus']
    colors = ['#476A2A', '#7851B8', '#BD3430', '#4A2D4E', '#875525', '#A83683', '#4E655E', '#853541', '#3A3120', '#535D8E']
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cls_names]).to(device)

    # Calculate features
    # with torch.no_grad():
    #     # image_features = model.encode_image(image_input)
    #     text_features = model.encode_text(text_inputs)
    scale = 2.0 if args.show else 1.0
    n = 10000
    image_features_all = torch.zeros(n, 512)
    gt_label = torch.zeros(n)
    gt_label_name = []
    cur_idx = 0
    correct = [0] * 8
    total_instance_num = [0] * 8
    train_data_loader = build_detection_train_loader(cfg)
    for idx_d, batch in enumerate(train_data_loader):
        if idx_d % 100 == 0:
            print('image {}'.format(idx_d))
        if cur_idx >= n:
            break
        for per_image in batch:
            if cur_idx >= n:
                break
            # Pytorch tensor is in (C, H, W) format
            img = per_image["image"].permute(1, 2, 0)
            if cfg.INPUT.FORMAT == "BGR":
                img = img[:, :, [2, 1, 0]]
            else:
                img = np.asarray(Image.fromarray(img, mode=cfg.INPUT.FORMAT).convert("RGB"))
            gt_bbox = per_image['instances'].gt_boxes.tensor
            for idx in range(gt_bbox.shape[0]):
                cur_bbox = gt_bbox[idx]

                cropped_img = img[cur_bbox[1].int():cur_bbox[3].int()+1, cur_bbox[0].int():cur_bbox[2].int()+1,:]
                cropped_img = pil(cropped_img.permute(2, 0, 1) / 256)
                # image_input = preprocess(cropped_img).unsqueeze(0).to(device)
                # with torch.no_grad():
                #     image_features = model.encode_image(image_input)
                image_input = transform_test(cropped_img)
                # import pdb; pdb.set_trace()
                with torch.no_grad():
                    image_features = model(image_input.unsqueeze(0))
                image_features = image_features.reshape(-1)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                # text_features /= text_features.norm(dim=-1, keepdim=True)
                # similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                # values, indices = similarity[0].topk(5)

                image_features_all[cur_idx] = image_features
                gt_label[cur_idx] = per_image['instances'].gt_classes[idx]
                gt_label_name.append(cls_names[gt_label[cur_idx].int()])
                total_instance_num[gt_label[cur_idx].int()] += 1
                # if indices[0] == gt_label[cur_idx]:
                #     correct[gt_label[cur_idx].int()] += 1
                cur_idx += 1
                if cur_idx >= n:
                    break
                print(cur_idx)
    for c_idx, c_name in enumerate(cls_names):
        print('{}: {} / {}, {:.2f}%'.format(c_name, correct[c_idx], total_instance_num[c_idx], correct[c_idx] / total_instance_num[c_idx] * 100))
    from sklearn.manifold import TSNE
    import time
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    import pdb; pdb.set_trace()

    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(image_features_all.numpy())
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))
    pdb.set_trace()
    # plot
    df = pd.DataFrame()
    df['tsne-2d-one'] = tsne_results[:, 0]
    df['tsne-2d-two'] = tsne_results[:, 1]
    df['class'] = gt_label_name
    plt.figure(figsize=(10, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="class",
        data=df,
        legend="full",
        alpha=0.5
    )
    plt.savefig('tsne.png')