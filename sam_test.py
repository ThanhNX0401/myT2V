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

from gpt import get_label

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py" #unix path
CHECK_POINT_PATH = "GroundingDINO/weights/groundingdino_swint_ogc.pth"


def enhance_class_name(class_names: List[str]) -> List[str]:
    return [
        f"all {class_name}s"
        for class_name
        in class_names
    ]


def load_model():
    GD_model = Model(model_config_path=CONFIG_PATH,model_checkpoint_path=CHECK_POINT_PATH)
    model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    model_dict = dict(sam_model=model, sam_processor=processor, GD_model=GD_model)
    return model_dict

def get_box(model_dict, image, class_name, box_threshold=0.35, text_threshold=0.25):
    #IMAGE_PATH = "GroundingDINO/Image/spiderman.jpg"
    #CLASSES = ["spiderman", "surf board"]
    image = (image * 255).astype(np.uint8) # BGR //need ndarray (512,512,3)
    CLASSES = class_name
    BOX_THRESHOLD = box_threshold
    TEXT_THRESHOLD = text_threshold
    # load image
    #image = cv2.imread(IMAGE_PATH) # BGR //need ndarray (512,512,3)
    # detect objects
    
    GD_model = model_dict['GD_model']
    detections = GD_model.predict_with_classes(
        image=image,
        classes=enhance_class_name(class_names=CLASSES),
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )
    # print(detections)

    # # annotate image with detections
    # box_annotator = sv.BoxAnnotator()
    # labels = [
    #     f"{CLASSES[class_id]} {confidence:0.2f}" 
    #     for boxes, mask, confidence, class_id, _ 
    #     in detections]

    # annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
    # annotated_frame = annotated_frame[...,::-1] # BGR to RGB image with boxes and labels
    # #annotated_frame= cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    boxes = detections.xyxy.tolist()
    return boxes

def sam(model_dict, image, input_points=None, input_boxes=None, target_mask_shape=None, return_numpy=True):
    """target_mask_shape: (h, w)"""
    sam_model, sam_processor = model_dict['sam_model'], model_dict['sam_processor']
    
    with torch.no_grad():
        #with torch.autocast(device):
        inputs = sam_processor(image, input_points=input_points, input_boxes=[input_boxes], return_tensors="pt").to(device)
        outputs = sam_model(**inputs)
        masks = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu().float(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
        )
        conf_scores = outputs.iou_scores.cpu().numpy()[0]
        del inputs, outputs
    
    # Uncomment if experiencing out-of-memory error:
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

def select_mask(masks, conf_scores, coarse_ious=None, rule="largest_over_conf", discourage_mask_below_confidence=0.85, discourage_mask_below_coarse_iou=0.2, verbose=3):
    """masks: numpy bool array"""
    mask_sizes = masks.sum(axis=(1, 2))
    
    # Another possible rule: iou with the attention mask
    if rule == "largest_over_conf":
        # Use the largest segmentation
        # Discourage selecting masks with conf too low or coarse iou is too low
        max_mask_size = np.max(mask_sizes)
        if coarse_ious is not None:
            scores = mask_sizes - (conf_scores < discourage_mask_below_confidence) * max_mask_size - (coarse_ious < discourage_mask_below_coarse_iou) * max_mask_size
        else:
            scores = mask_sizes - (conf_scores < discourage_mask_below_confidence) * max_mask_size
        if verbose:
            print(f"mask_sizes: {mask_sizes}, scores: {scores}")
    else:
        raise ValueError(f"Unknown rule: {rule}")

    mask_id = np.argmax(scores)
    mask = masks[mask_id]
    
    selection_conf = conf_scores[mask_id]
    
    if coarse_ious is not None:
        selection_coarse_iou = coarse_ious[mask_id]
    else:
        selection_coarse_iou = None

    if verbose:
        # print(f"Confidences: {conf_scores}")
        print(f"Selected a mask with confidence: {selection_conf}") #, coarse_iou: {selection_coarse_iou}")

    if verbose >= 2:
        plt.figure(figsize=(10, 8))
        # plt.suptitle("After SAM")
        for ind in range(3):
            plt.subplot(1, 3, ind+1)
            # This is obtained before resize.
            plt.title(f"Mask {ind}, score {scores[ind]}, conf {conf_scores[ind]:.2f}") #, iou {coarse_ious[ind] if coarse_ious is not None else None:.2f}")
            plt.imshow(masks[ind])
        plt.tight_layout()
        plt.show()
        plt.close()

    return mask, selection_conf

def get_mask(image, prompt):
    print("get mask")
    model_dict = load_model()
    image = image.squeeze()
    class_name = get_label(prompt)
    boxes=get_box(model_dict, image, class_name, box_threshold=0.35, text_threshold=0.25)
    target_mask_shape=(512,512)
    masks, conf_scores = sam_box_input(model_dict, image, input_boxes=boxes, target_mask_shape=target_mask_shape, return_numpy=True)
    
    selected_mask_list=[]
    selected_scores_list=[]
    for idx in range(len(masks[0])):
        selected_mask, selected_scores = select_mask(masks[0][idx],conf_scores[idx])
        selected_mask_list.append(selected_mask)
        selected_scores_list.append(selected_scores)
    
    combined_mask = np.logical_or.reduce(selected_mask_list)
    print(combined_mask.shape)
    #plot the combined_mask
    plt.figure(figsize=(10, 8))
    plt.imshow(combined_mask)
    plt.tight_layout()
    plt.show()
    # plt.close()
    
    return combined_mask
