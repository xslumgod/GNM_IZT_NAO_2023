from easyocr import easyocr
from matplotlib import pyplot as plt

from algorithm.object_detector import YOLOv7
from utils.detections import draw
import json
import cv2

yolov7 = YOLOv7()
yolov7.load('best.weights', classes='classes.yaml', device='cpu') # use 'gpu' for CUDA GPU inference
yolov7.set(ocr_classes=['ocr_classes'])
image = cv2.imread('8.jpg')


def recognize_text(img_path):
    '''loads an image and recognizes text.'''

    reader = easyocr.Reader(['en'])
    return reader.readtext(img_path)

result = recognize_text(image)


def overlay_ocr_text(img_path):
    '''loads an image, recognizes text, and overlays the text on the image.'''

    # loads image
    img = cv2.imread('8.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    dpi = 80
    fig_width, fig_height = int(img.shape[0] / dpi), int(img.shape[1] / dpi)
    plt.figure()
    f, axarr = plt.subplots(1, 2, figsize=(fig_width, fig_height))
    axarr[0].imshow(img)

    # recognize text
    result = recognize_text(img_path)

    # if OCR prob is over 0.5, overlay bounding box and text
    for (bbox, text, prob) in result:
        if prob >= 0.35:
            # display
            print(f'Detected text: {text} (Probability: {prob:.2f})')

            # get top-left and bottom-right bbox vertices
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = (int(top_left[0]), int(top_left[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))

            # create a rectangle for bbox display
            cv2.rectangle(img=img, pt1=top_left, pt2=bottom_right, color=(255, 0, 0), thickness=2)

            # put recognized text
            cv2.putText(img=img, text=text, org=(top_left[0], top_left[1] - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(102, 255, 178), thickness=2)

    # show and save image
    axarr[1].imshow(img)
    plt.savefig('books_read.jpg', bbox_inches='tight')

detections = yolov7.detect(image)
detected_image = draw(image, detections)
overlay_ocr_text(image)
cv2.imwrite('detected.jpg', detected_image)
print(json.dumps(detections, indent=4))