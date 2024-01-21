import torch
from transformers import SamModel, SamProcessor
import gc
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import SamModel, SamProcessor
import cv2
from scipy import ndimage

from GroundingDINO.groundingdino.util.inference import Model
import supervision as sv
import cv2

from typing import List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py" #unix path
CHECK_POINT_PATH = "GroundingDINO/weights/groundingdino_swint_ogc.pth"

def load_model():
    GD_model = Model(model_config_path=CONFIG_PATH,model_checkpoint_path=CHECK_POINT_PATH)
    model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    model_dict = dict(sam_model=model, sam_processor=processor, GD_model=GD_model)
    return model_dict


def enhance_class_name(class_names: List[str]) -> List[str]:
    return [
        f"all {class_name}s"
        for class_name
        in class_names
    ]

def get_box(model_dict, image, class_name, box_threshold=0.35, text_threshold=0.25):
    CLASSES = class_name
    BOX_THRESHOLD = box_threshold
    TEXT_THRESHOLD = text_threshold
    GD_model = model_dict['GD_model']
    detections = GD_model.predict_with_classes(
        image=image,
        classes=enhance_class_name(class_names=CLASSES),
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )
    boxes = detections.xyxy.tolist()
    del GD_model
    torch.cuda.empty_cache()
    return boxes

def sam(model_dict, image, input_points=None, input_boxes=None, target_mask_shape=None, return_numpy=True):
    """target_mask_shape: (h, w)"""
    sam_model, sam_processor = model_dict['sam_model'], model_dict['sam_processor']
    
    with torch.no_grad():
        inputs = sam_processor(image, input_points=input_points, input_boxes=[input_boxes], return_tensors="pt").to(device)
        outputs = sam_model(**inputs)
        masks = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu().float(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
        )
        conf_scores = outputs.iou_scores.cpu().numpy()[0]
        del inputs, outputs
    # utils.free_memory()
    
    if return_numpy:
        masks = [F.interpolate(masks_item.type(torch.float), target_mask_shape, mode='bilinear').type(torch.bool).numpy() for masks_item in masks]
    else:
        masks = [F.interpolate(masks_item.type(torch.float), target_mask_shape, mode='bilinear').type(torch.bool) for masks_item in masks]

    return masks, conf_scores

def sam_point_input(sam_model_dict, image, input_points, **kwargs):
    return sam(sam_model_dict, image, input_points=input_points, **kwargs)
    
def sam_box_input(sam_model_dict, image, input_boxes, **kwargs):
    
    return sam(sam_model_dict, image, input_boxes=input_boxes, **kwargs)

def select_mask(masks, conf_scores, discourage_mask_below_confidence=0.85):

    mask_sizes = masks.sum(axis=(1, 2))
    max_mask_size = np.max(mask_sizes)
    scores = mask_sizes - (conf_scores < discourage_mask_below_confidence) * max_mask_size

    print(f"mask_sizes: {mask_sizes}, scores: {scores}")

    mask_id = np.argmax(scores)
    mask = masks[mask_id]
    
    selection_conf = conf_scores[mask_id]

    print(f"Selected a mask with confidence: {selection_conf}") 
    return mask, selection_conf

def get_mask(image, prompt,labels):
    print("get mask")
    model_dict = load_model()
    
    image = image.squeeze() # (512,512,3)
    image = (image * 255).astype(np.uint8)
    cv2.imwrite('output_image.png', image)
    
    class_name = labels
    boxes=get_box(model_dict, image, class_name, box_threshold=0.35, text_threshold=0.25)
    # check if boxes=[] then assign boxes = image resolution
    if boxes==[]:
        boxes=[[0,0,512,512]]
    target_mask_shape=(512,512)
    masks, conf_scores = sam_box_input(model_dict, image, input_boxes=boxes, target_mask_shape=target_mask_shape, return_numpy=True)
    
    print("masks, conf_score", masks[0].shape, conf_scores.shape)
    
    selected_mask_list=[]
    #selected_scores_list=[]
    for idx in range(len(masks[0])):
        selected_mask, selected_scores = select_mask(masks[0][idx],conf_scores[idx])
        selected_mask_list.append(selected_mask)
        #selected_scores_list.append(selected_scores)
    combined_mask = np.logical_or.reduce(selected_mask_list) #only 1 mask
    print("combine mask")
    has_different_value = ((combined_mask != 0) & (combined_mask != 1)).any()
    print(has_different_value.item())
    combined_mask = (combined_mask * 255).astype(np.uint8)
    print("combine mask1")
    has_different_value = ((combined_mask != 0) & (combined_mask != 1)).any()
    print(has_different_value.item())
    cv2.imwrite('mask1.png', combined_mask)
    print(combined_mask.shape)
    #plot the combined_mask
    plt.figure(figsize=(10, 8))
    plt.imshow(combined_mask)
    plt.savefig('combined_mask.png')

    return combined_mask
