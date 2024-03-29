import torch
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import KeypointRCNN
import cv2
import numpy as np
import io
from flask import send_file
import requests
import os

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
keypoints_classes_ids2names = {0: 'Carina' , 1: 'ETT'}

def get_model(num_keypoints=2, anchor_sizes = (64, 128, 256) , anchor_ratios= (0.5, 1, 2)):
    
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
            raise ValueError("No scores: avg_scores is empty. Make sure you have calculated the average scores before finding the maximum.")
    except ValueError as e:
        print(e)
        return []

def load_model(model, weights_path):
    model_weights = torch.load(weights_path,  map_location=torch.device('cpu')) # CPU only for now
    model.load_state_dict(model_weights['model_state_dict'])
    return model

url = 'https://github.com/nicholasprimiano/et_tube/releases/download/v0.1/production.pth'
filename = 'production.pth'

# Check if the file already exists
if os.path.exists(filename):
    print(f"Model: {filename} already exists.")
else:
    response = requests.get(url, allow_redirects=True)
    # Check if the request was successful
    if response.status_code == 200:
        # Save the content to a local file
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {filename} successfully.")
    else:
        print(f"Failed to download {filename}. Status code: {response.status_code}")

print("Loading model...")
best_model = load_model(get_model(), 'production.pth')
print("Loaded model.")

def resize_image_aspect_ratio(img, max_size):

    # Get current dimensions
    height, width = img.shape[:2]

    # Only resize if the image is larger than the max size
    if max(height, width) > max_size:
        # Calculate the ratio
        ratio = min(max_size/height, max_size/width)

        # Calculate new dimensions
        new_dimensions = (int(width * ratio), int(height * ratio))

        # Resize the image
        resized_img = cv2.resize(img, new_dimensions, interpolation=cv2.INTER_AREA)
        print("Resizing Image")
        return resized_img
    else:
        # Return original image if no resizing is needed
        return img


IMG_SIZE = 1080

def getPrediction(file_name):
    with torch.inference_mode():
        best_model.to(device)
        best_model.eval()

        test_img_file = file_name
        print(test_img_file)
        raw_img = cv2.imread(test_img_file)
        #raw_img = cv2.resize(raw_img, (IMG_SIZE, IMG_SIZE))
        raw_img = resize_image_aspect_ratio(raw_img, IMG_SIZE)
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
        
        is_success, buffer =  cv2.imencode(".jpg", raw_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

        if is_success:
            img_io = io.BytesIO(buffer)
            return img_io
        else:
            return "Error processing image", 500