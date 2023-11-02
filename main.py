import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F
from torchvision.models.detection import KeypointRCNN
import cv2
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
keypoints_classes_ids2names = {0: 'Carina' , 1: 'ETT'}

def get_model(num_keypoints=2, anchor_sizes = (64, 128, 256) , anchor_ratios= (0.5, 0.83, 1.2, 2), weights_path=None):
    
    backbone = torchvision.models.convnext_large(weights='DEFAULT').features
    backbone.out_channels = 1536

    anchor_generator = AnchorGenerator(sizes=(anchor_sizes,), aspect_ratios=(anchor_ratios,))
    
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)
    keypoint_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                             output_size=14,
                                                             sampling_ratio=2)

    model = KeypointRCNN(backbone,
                          num_classes=2,
                          rpn_anchor_generator=anchor_generator,
                          box_roi_pool=roi_pooler,
                          keypoint_roi_pool=keypoint_roi_pooler,
                          num_keypoints=num_keypoints)        
        
    return model


def load_model(model, weights_path):
    model_weights = torch.load(weights_path)
    model.load_state_dict(model_weights['model_state_dict'])
    return model

def get_best_keypoints(scores):
    avg_scores = []
    for score in scores:
        avg_scores.append(torch.mean(score))

    try:
        if len(avg_scores) > 0:
            return avg_scores.index(max(avg_scores))
        else:
            raise ValueError("avg_scores is empty. Make sure you have calculated the average scores before finding the maximum.")
    except ValueError as e:
        print(e)
        return []  # or some other default value
    
model = load_model(get_model(), "D:\ET_Tube\CheXpert-v1.0\checkpoints\checkpoint__1_5.pt")

def process_and_predict(image_file):
    raw_img = cv2.imread(image_file)
    aspect_ratio = raw_img.shape[0]/raw_img.shape[1]

    if aspect_ratio < 1.5 and aspect_ratio > 0.5:
        #raw_img = cv2.resize(raw_img, (456, 456))
        test_img = np.moveaxis(raw_img, -1, 0)
        test_img= np.expand_dims(test_img, axis=0)
        test_img = torch.from_numpy(test_img).float().to(device)

        with torch.no_grad():
            model.to(device)            
            model.eval()
            output = model(test_img)
        
        scores = output[0]['keypoints_scores']
        score_idx = get_best_keypoints(scores)

        keypoints = output[0]['keypoints'][score_idx].detach().cpu().numpy()
        for idx, kp in enumerate(keypoints):
            current_keypoint = kp[:2].astype(int)
            raw_img = cv2.circle(raw_img, current_keypoint, 1, (255,255,0), 10)
            image_original = cv2.putText(raw_img, " " + keypoints_classes_ids2names[idx], current_keypoint, cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2, cv2.LINE_AA)

        plt.figure(figsize=(10,10))
        plt.imshow(raw_img)
        plt.show()
    else:
        print(f"Image aspect ratio is {aspect_ratio:0.2F} it but be between 0.5 - 1.5")

test_img = r"C:\Users\nprim\Downloads\A-chest-x-ray-showing-correct-endotracheal-tube-placement-and-no-acute-lung-pathology.png"
process_and_predict(test_img) 