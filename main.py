import torch
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import KeypointRCNN
import cv2
#import albumentations as A
import numpy as np
import io
from flask import send_file
import requests

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
        return []

def load_model(model, weights_path):
    model_weights = torch.load(weights_path)
    model.load_state_dict(model_weights['model_state_dict'])
    return model

IMG_SIZE = 456


url = 'https://github.com/nicholasprimiano/et_tube/releases/download/v0.1/production.pth'
response = requests.get(url, allow_redirects=True)

# Save the content to a local file
with open('production.pth', 'wb') as f:
    f.write(response.content)


best_model = load_model(get_model(), 'production.pth')


def getPrediction(file_name):
    with torch.inference_mode():
        best_model.to(device)
        best_model.eval()

        test_img_file = rf"static/images/" + file_name
        print(test_img_file)
        raw_img = cv2.imread(test_img_file)
        raw_img = cv2.resize(raw_img, (IMG_SIZE, IMG_SIZE))
        #raw_img_processed = test_valid_transform()(image=raw_img)
        #raw_img_processed = raw_img_processed["image"]
        raw_img_processed = torch.from_numpy(raw_img).permute(2,0,1).float().to(device)
        raw_img_processed = raw_img_processed.float() / 255.0

        output = best_model([raw_img_processed])


        scores = output[0]['scores'].detach().cpu().numpy()
        high_scores_idxs = np.where(scores > 0.1)[0].tolist() # Indexes of boxes with scores > 0.1
        post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy() # Indexes of boxes left after applying NMS (iou_threshold=0.3)
        keypoint_scores = output[0]['keypoints_scores'][post_nms_idxs]
        score_idx = get_best_keypoints(keypoint_scores)

        keypoints = output[0]['keypoints'][score_idx].detach().cpu().numpy()

        
        for idx, kp in enumerate(keypoints):
                current_keypoint = kp[:2].astype(int)
                raw_img = cv2.circle(raw_img.copy(), current_keypoint, 1, (255,255,0), 10)
                cv2.putText(raw_img, " " + keypoints_classes_ids2names[idx], current_keypoint, cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 1, cv2.LINE_AA)
        
        is_success, buffer = cv2.imencode(".png", raw_img)
        #cv2.imshow('Image', raw_img)
        #cv2.waitKey(0)
        if is_success:
            img_io = io.BytesIO(buffer)
            return img_io
        else:
            return "Error processing image", 500

#getPrediction("cxr.png")