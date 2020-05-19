from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse
import cv2
import numpy as np

from PIL import Image
from IPython.display import clear_output

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

def detect_image(img):
     # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms=transforms.Compose([transforms.Resize((imh,imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)), (128,128,128)), transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
    return detections[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("output", exist_ok=True)
    #f = open(sys.argv[1], 'w') 
    
    videopath = 'data/samples/test.mp4'
    vid = cv2.VideoCapture(videopath)

    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('A','V','C','1'), 1, mat.shape[0], mat.shape[1])
    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode

#    dataloader = DataLoader(
#        ImageFolder(opt.image_folder, img_size=opt.img_size),
#        batch_size=opt.batch_size,
#        shuffle=False,
#        num_workers=opt.n_cpu,
#    )
    img_size = opt.img_size
    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    for ii in range(40):
        ret, frame = vid.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pilimg = Image.fromarray(frame)
        detections = detect_image(pilimg)


        img = np.array(pilimg)
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
        unpad_h = img_size - pad_y
        unpad_w = img_size - pad_x

        if detections is not None:
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                # box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                # box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
                # y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                # x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])

                # color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # cls = classes[int(cls_pred)]
                # cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h),color, 4)
                # cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+60,y1), color, -1)
                # cv2.putText(frame, cls + "-" + str(cls), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)    
                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                box_w = x2 - x1
                box_h = y2 - y1

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(
                    x1,
                    y1,
                    s=classes[int(cls_pred)],
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                )
                out.write(ax)

        #fig=figure(figsize=(12, 8))
         # Save generated image with detections
        #plt.axis("off")
        #plt.gca().xaxis.set_major_locator(NullLocator())
        #plt.gca().yaxis.set_major_locator(NullLocator())
        #plt.close()
        cv2.destroyAllWindows()
        out.release()
        clear_output(wait=True)
