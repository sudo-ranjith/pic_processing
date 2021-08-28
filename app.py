import io
from tkinter import Image
import numpy as np
import cv2
import torch
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision import transforms
from PIL import Image
from flask import Flask, request, make_response, send_file
import matplotlib.pyplot as plt


app = Flask(__name__)


@app.route('/ping', methods=['GET'])
def ping():
    result = {}
    result['message'] = "welcome to image processing"
    return result

@app.route('/', methods=['POST'])
def upload_file():
    images = request.files.getlist('images')
    height = int(request.form.get("height", 0))
    width = int(request.form.get("width", 0))
    background_image = cv2.imdecode(np.fromstring(request.files['background_image'].read(), np.uint8), cv2.IMREAD_COLOR)

    if height or width:
        background_image = Image.fromarray(background_image).resize(find_aspect_radio(background_image, height, width))
    else:
        background_image = Image.fromarray(background_image)
    for image in images:
        if image.filename.endswith("jpg") or image.filename.endswith("svg") or image.filename.endswith("png"):
            npimg = np.fromstring(image.read(), np.uint8)
            original_img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            result = remove_bg(original_img, background_image)
            img_io = io.BytesIO()
            result.save(img_io, 'JPEG', quality=70)
            img_io.seek(0)
            return send_file(img_io, mimetype='image/jpeg')
            # import pdb;pdb.set_trace()


def make_deeplab(device):
    deeplab = deeplabv3_resnet101(pretrained=True).to(device)
    deeplab.eval()
    return deeplab


deeplab_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def apply_deeplab(deeplab, img, device):
    input_tensor = deeplab_preprocess(img)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = deeplab(input_batch.to(device))['out'][0]
    output_predictions = output.argmax(0).cpu().numpy()
    return output_predictions == 15


device = torch.device("cpu")
deeplab = make_deeplab(device)

deeplab_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def apply_deeplab(deeplab, img, device):
    input_tensor = deeplab_preprocess(img)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = deeplab(input_batch.to(device))['out'][0]
    output_predictions = output.argmax(0).cpu().numpy()
    return output_predictions == 15


def calculate_center_of_image(foregorund, background):
    img_w, img_h = foregorund.size
    bg_w, bg_h = background.size
    offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
    return offset


def remove_bg(img_orig, background_img):
    k = min(1.0, 1024 / max(img_orig.shape[0], img_orig.shape[1]))
    img = cv2.resize(img_orig, None, fx=k, fy=k, interpolation=cv2.INTER_LANCZOS4)
    mask = apply_deeplab(deeplab, img, device).astype(np.uint8)
    masked = cv2.bitwise_and(img_orig[:, :, ::-1], img_orig[:, :, ::-1], mask=mask)
    tmp = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(masked)
    rgba = [b, g, r, alpha]
    dst = cv2.merge(rgba, 4)
    width, height = background_img.size
    img1 = Image.fromarray(image_resize(dst, height=height))
    offset = calculate_center_of_image(img1, background_img)
    background_img.paste(img1, box=offset, mask=img1)
    return background_img


def find_aspect_radio(image, width=None, height=None):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]
    # if both the width and height are None, then return the
    # original image
    if not width and not height:
        return image
    # check to see if the width is None
    if not width:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    return dim


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = find_aspect_radio(image, width, height)
    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

# f_name = "img3.jpg"
# original_img = cv2.imread(f_name)
# background_image = cv2.imread(r"background2.jpg")
# height = 1000
# width = 1000
# if height or width:
#     width, height = find_aspect_radio(background_image, height, width)
#     background_image = Image.fromarray(background_image).resize((width, height))
# result = remove_bg(original_img, background_image)
# print(original_img, type(background_image))
# result.show()
