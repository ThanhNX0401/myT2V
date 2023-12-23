from GroundingDINO.groundingdino.util.inference import load_model,load_image, predict, annotate
import supervision
import cv2

CONFIG_PATH = "GroundingDINO\groundingdino\config\GroundingDINO_SwinT_OGC.py" #unix path

CHECK_POINT_PATH = "GroundingDINO\weights\groundingdino_swint_ogc.pth"
GD_model = load_model(CONFIG_PATH, CHECK_POINT_PATH)

IMAGE_PATH = "GroundingDINO\Image\spiderman.jpg"
TEXT_PROMPT = "surf board"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

image_sourcee, myImage = load_image(IMAGE_PATH)

boxes, accuracy, obj_name = predict(
    model= GD_model, 
    image= myImage, 
    caption= TEXT_PROMPT, 
    box_threshold = BOX_THRESHOLD, 
    text_threshold= TEXT_THRESHOLD)
print(boxes)

# annotated_image = annotate(image_source = image_source,boxes = detected_boxes, logits = accuracy, phrases = obj_name)
# print(annotated_image.shape)
# cv.plot_imgae(annotated_image, (16,16))