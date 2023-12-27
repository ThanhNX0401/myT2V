from GroundingDINO.groundingdino.util.inference import Model
import supervision as sv
import cv2

from typing import List

def enhance_class_name(class_names: List[str]) -> List[str]:
    return [
        f"all {class_name}s"
        for class_name
        in class_names
    ]

CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py" #unix path
CHECK_POINT_PATH = "GroundingDINO/weights/groundingdino_swint_ogc.pth"

GD_model = Model(model_config_path=CONFIG_PATH,model_checkpoint_path=CHECK_POINT_PATH)


IMAGE_PATH = "GroundingDINO/Image/spiderman.jpg"
CLASSES = ["spiderman", "surf board"]
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

# load image
image = cv2.imread(IMAGE_PATH) # BGR 

# detect objects
detections = GD_model.predict_with_classes(
    image=image,
    classes=enhance_class_name(class_names=CLASSES),
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD
)
print(detections)

# annotate image with detections
box_annotator = sv.BoxAnnotator()
labels = [
    f"{CLASSES[class_id]} {confidence:0.2f}" 
    for boxes, mask, confidence, class_id, _ 
    in detections]

annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
annotated_frame = annotated_frame[...,::-1] # BGR to RGB image with boxes and labels
#annotated_frame= cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
