import cv2
import numpy as np
import base64

def decode_image(base64_string):
    img_data = base64.b64decode(base64_string.split(',')[1])
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def encode_image(image):
    _, buffer = cv2.imencode('.png', image)
    img_str = base64.b64encode(buffer).decode()
    return f'data:image/png;base64,{img_str}'

def apply_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def apply_canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def process_image(image_data, operation):
    img = decode_image(image_data)
    
    processors = {
        'hsv': apply_hsv,
        'canny': apply_canny
    }
    
    processed = processors[operation](img)
    return encode_image(processed)