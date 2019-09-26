from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F

import torchvision
from torchvision import datasets

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
 

from utils.utils import rescale_boxes
from utils.datasets import pad_to_square, resize


 

from ALL_sign_data.model import Lenet5, my_resnt18, FashionCNN



os.environ['CUDA_VISIBLE_DEVICES']='3'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


sign_classes = 118
weights_path = "ALL_sign_data/model_acc_90__epoch_5.pt"



# os.makedirs("output", exist_ok=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/changshu_17_before", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/ALL_DATA.cfg", help="path to model definition file")
    # parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_13.pth", help="path to weights file")
    parser.add_argument("--weights_path", type=str, default="checkpoints_1/yolov3_ckpt_24.pth", help="path to weights file")
    # parser.add_argument("--class_path", type=str, default="data/ALL_DATA.names", help="path to class label file")
    parser.add_argument("--class_path", type=str, default="ALL_sign_data/ALL_data_in/names.txt", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=1216, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode

    # dataloader = DataLoader(
    #     ImageFolder(opt.image_folder, img_size=opt.img_size),
    #     batch_size=opt.batch_size,
    #     shuffle=False,
    #     num_workers=opt.n_cpu,
    # )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # imgs = []  # Stores image paths
    # img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    # prev_time = time.time()

    # to  class
    model_class = FashionCNN(sign_classes)
    model_class.load_state_dict(torch.load(weights_path))
    model_class.to(device)
    model_class.eval()
    # to  class






    # for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
    names = os.listdir(opt.image_folder)
    for name in names:
        img_path = os.path.join(opt.image_folder, name)


        # Extract image as PyTorch tensor
        img = torchvision.transforms.ToTensor()(Image.open(img_path))
      
        input_imgs, _ = pad_to_square(img, 0)
        # Resize
        input_imgs = resize(input_imgs, opt.img_size).unsqueeze(0)
 

 


        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs.to(device))
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)[0]


        # Log progress
        # current_time = time.time()
        # inference_time = datetime.timedelta(seconds=current_time - prev_time)
        # prev_time = current_time
        # print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections

        # print("detections = ", detections)
        # imgs.extend(img_paths)
        # img_detections.extend(detections)
        if  detections is not None:
            detections = rescale_boxes(detections, opt.img_size, img.shape[1:])
            
            unique_labels = detections[:, -1].cpu().unique()
            # n_cls_preds = len(unique_labels)
            # bbox_colors = random.sample(colors, n_cls_preds)
            # plt.figure()
            fig, ax = plt.subplots()
            img_copy =Image.open(img_path) 
            # ax.imshow(img_copy)
            for i, (x1, y1, x2, y2, conf, cls_conf, cls_pred) in enumerate(detections):
                
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                box_w = x2 - x1
                box_h = y2 - y1
                
                if box_w >=15 and box_h >= 15:
                    crop_sign = img_copy.crop((x1, y1, x2, y2))
                    # sign_type = 

                    # #### to class  ###############
                    test_transform = torchvision.transforms.Compose([ 
                        torchvision.transforms.Resize((28, 28), interpolation=2),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
                        ])

                    crop_sign_input = test_transform(crop_sign).unsqueeze(0)
                    # input_img = torch.autograd.Variable(input_img)

                    # print("input_img = ", input_img.size())
                    with torch.no_grad():
                        pred_class = model_class(crop_sign_input.to(device))
                    sign_type  = torch.max(pred_class, 1)[1].to("cpu").numpy()[0]
                    # #### to class  ###############
                    cls_pred = sign_type


                    print("cls_pred = ", cls_pred)




                   
                    # color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                    color = "r"
                    # Create a Rectangle patch
                    # plt.imshow()
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=1, edgecolor=color, facecolor="none")
                    # Add the bbox to the plot
                    ax.add_patch(bbox)
                    # Add label
                    plt.text(
                        x1,
                        y2 + 50,
                        s=classes[int(cls_pred)],
                        color="white",
                        verticalalignment="top",
                        bbox={"color": color, "pad": 0},
                    )
                    

                    pad_sign_path = "ALL_sign_data/pad-all/" + classes[int(cls_pred)] + ".png"
                    if  os.path.isfile(pad_sign_path):
                        pad_sign = Image.open(pad_sign_path)
                    else:
                        pad_sign = Image.new("RGB", (100, 100), (255, 255, 255))


                    img_copy.paste(crop_sign.resize((100, 100)), (0, i * 100) )
                    img_copy.paste(pad_sign.resize((100, 100)), (100, i * 100) )
                    
            # Save generated image with detections
            # plt.axis("off")
            # plt.gca().xaxis.set_major_locator(NullLocator())
            # plt.gca().yaxis.set_major_locator(NullLocator())
           

            ax.imshow(img_copy)
            plt.show()







        # filename = path.split("/")[-1].split(".")[0]
        # plt.savefig(f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0, dpi=400)
        # plt.close()



    # # Bounding-box colors
    # cmap = plt.get_cmap("tab20b")
    # colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    # print("\nSaving images:")



    # # Iterate through images and save plot of detections
    # for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

    #     # print("(%d) Image: '%s'" % (img_i, path))

    #     # Create plot
    #     img = np.array(Image.open(path))
    #     print("img.shape = ", img.shape) # ( 2048 2048 3 )

    #     plt.figure()
    #     fig, ax = plt.subplots(1)
    #     ax.imshow(img)
        
    #     # Draw bounding boxes and labels of detections
    #     if detections is not None:
    #         # Rescale boxes to original image
        #     detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
        #     unique_labels = detections[:, -1].cpu().unique()
        #     n_cls_preds = len(unique_labels)
        #     bbox_colors = random.sample(colors, n_cls_preds)
        #     for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

        #         print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

        #         box_w = x2 - x1
        #         box_h = y2 - y1

        #         color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
        #         color = "r"
        #         # Create a Rectangle patch
        #         bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=1, edgecolor=color, facecolor="none")
        #         # Add the bbox to the plot
        #         ax.add_patch(bbox)
        #         # Add label
        #         plt.text(
        #             x1,
        #             y2 + 50,
        #             s=classes[int(cls_pred)],
        #             color="white",
        #             verticalalignment="top",
        #             bbox={"color": color, "pad": 0},
        #         )

        # # Save generated image with detections
        # plt.axis("off")
        # plt.gca().xaxis.set_major_locator(NullLocator())
        # plt.gca().yaxis.set_major_locator(NullLocator())
        # filename = path.split("/")[-1].split(".")[0]
        # plt.savefig(f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0, dpi=400)
        # plt.close()

