from app_args import FLAGS

import numpy as np
import cv2

class Params(object):
    """
    Container for the default parameters for the image processor
    """
    INPUT_SHAPE = (160, 320, 3)
    CLIP = [60, 25]
    RESIZE = (200, 66)
    BLUR = FLAGS.blur
    # Contrast Limited Adaptive Histogram Equalization
    CLAHE = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8)) if FLAGS.clahe else None

Params = Params()

def output_shape(clip = Params.CLIP, resize = Params.RESIZE):
    """
    Returns the final shape after image processing that can be used as the input of the model
    without hardcoding the shape.

    Parameters
        clip: 2 elements array containing the amount of pixels that are removed from the top/bottom of
              the image
        resize: 2 values shape if resizing was part of the processing, this is the final shape
                (plus the original number of channels) if present
    Returns
        The shape of the image after processing
    """
    shape = Params.INPUT_SHAPE

    if resize is not None:
        shape = (resize[1], resize[0], Params.INPUT_SHAPE[2])
    elif clip is not None:
        shape = (Params.INPUT_SHAPE[0] - clip[0] - clip[1], Params.INPUT_SHAPE[1], Params.INPUT_SHAPE[2])

    return shape

def process_image(img, 
                  rgb = False,
                  blur = Params.BLUR,
                  clip = Params.CLIP,
                  resize = Params.RESIZE,
                  clahe = Params.CLAHE):
    """
    Process the given image so that it can be fed to correctly to the model.

    Parameters
        img: The input image to be processed
        rgb: True if the image is RGB, False for BGR
        clip: optional, 2 elements array containing the number of pixels to clip from the
              top and bottom of the input image
        resize: optional, final size (excluding channels) of the image, (width, height)
        clahe: optional, apply contrast limited adaptive histogram equalization to the first channel
    
        Note: Clipping is applied before resizing
    
    Returns
        The processed image
    """

    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV if rgb else cv2.COLOR_BGR2YUV)

    if clip is not None:
        img = img[clip[0]:img.shape[0] - clip[1], :, :]
    if resize is not None:
        img = cv2.resize(img, resize, interpolation = cv2.INTER_AREA)
    if clahe is not None:
        img[:, :, 0] = clahe.apply(img[:, :, 0])
    if blur:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    
    return img

def adjust_image_brightness(img, factor = 1.2):
    
    # Assume YUV color space
    y = img[:, :, 0]
    y = np.where(y * factor <= 255, y * factor, 255)
    adjusted_img = np.copy(img)
    adjusted_img[:, :, 0] = y

    return adjusted_img

def translate_image(img, x = 5):

    border = abs(x)
    adjusted_img = cv2.copyMakeBorder(img, 0, 0, border, border, cv2.BORDER_REPLICATE)
    row, col, _ = adjusted_img.shape
    trans_m = np.float32([[1, 0, x],[0, 1, 0]])
    adjusted_img = cv2.warpAffine(adjusted_img, trans_m, (col, row))
    
    return adjusted_img[:, border:col - border]