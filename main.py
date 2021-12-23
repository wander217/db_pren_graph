from models.det.predictor import DBPredictor
from models.rec.predictor import PRENPredictor
import cv2 as cv
from PIL import Image, ImageFont, ImageDraw

db_pred = DBPredictor(config_path=r'data/det/config/eb0-config.yaml',
                      pretrained=r'data/det/pretrained/checkpoint466.pth')
pren_pred = PRENPredictor(config_path=r"data/rec/config/pc_eb3.yaml",
                          pretrained=r"data/rec/pretrained/checkpoint406000.pth")
image = cv.imread(r"data/image/img1.jpg")
bbox = db_pred.predict(image)
bbox_list = []
x1, x2, y1, y2 = image.shape[1], 0, image.shape[0], 0
for i in range(len(bbox)):
    x_min, x_max, y_min, y_max = bbox[i]
    x1 = min(x_min, x1)
    x2 = max(x_max, x2)
    y1 = min(y_min, y1)
    y2 = max(y_max, y2)
image = image[y1:y2 + 1, x1:x2 + 1, :]
if image.shape[0] < image.shape[1]:
    image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
    for i in range(len(bbox)):
        x_min, x_max, y_min, y_max = bbox[i]
        bbox[i] = [y_min - y1, y_max - y1, x_min - x1, x_max - x1]
else:
    for i in range(len(bbox)):
        x_min, x_max, y_min, y_max = bbox[i]
        bbox[i] = [x_min - x1, x_max - x1, y_min - y1, y_max - y1]
for i in range(len(bbox)):
    x_min, x_max, y_min, y_max = bbox[i]
    crop_image = image[y_min:y_max + 1, x_min:x_max + 1, :]
    bbox_list.append([
        x_min, x_max, y_min, y_max,
        pren_pred.predict(crop_image)
    ])
for item in bbox_list:
    x_min, x_max, y_min, y_max, text = item
    cv.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
image = Image.fromarray(image)
image_draw = ImageDraw.Draw(image)
font = ImageFont.truetype("data/font/font33.tff", size=15)
for item in bbox_list:
    x_min, x_max, y_min, y_max, text = item
    image_draw.text((x_min, y_min), text, fill=(255, 0, 0), font=font)
image.show()
