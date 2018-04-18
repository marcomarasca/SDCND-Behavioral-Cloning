import cv2

class Params(object):
    """
    Container for the default parameters for the image processor
    """
    INPUT_SHAPE = (160, 320, 3)
    CS_CONV = cv2.COLOR_BGR2YUV
    CLIP = [60, 25]
    RESIZE = (200, 66)
    # Contrast Limited Adaptive Histogram Equalization
    CLAHE = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (11, 11))

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
                  cs_conv = Params.CS_CONV,
                  clahe = Params.CLAHE,
                  clahe_ch = 0,
                  clip = Params.CLIP,
                  resize = Params.RESIZE):
    """
    Process the given image so that it can be fed to correctly to the model.

    Parameters
        img: The input image to be processed
        cs_conv: optional, The color space conversion to apply (e.g. cv2.COLOR_BGR2YUV)
        clahe: optional, apply contrast limited adaptive histogram equalization to the first channel
        clahe_ch: The channel to which the clahe is applied to
        clip: optional, 2 elements array containing the number of pixels to clip from the
              top and bottom of the input image
        resize: optional, final size (excluding channels) of the image, (width, height)
    
        Note: Clipping is applied before resizing
    
    Returns
        The processed image
    """
    if cs_conv is not None:
        img = cv2.cvtColor(img, cs_conv)
    if clahe is not None:
        img[:,:,clahe_ch] = clahe.apply(img[:,:,clahe_ch])
    if clip is not None:
        img = img[clip[0]:img.shape[0] - clip[1],:,:]
    if resize is not None:
        img = cv2.resize(img, resize, interpolation = cv2.INTER_AREA)
    
    return img
