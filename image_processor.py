import cv2

class Params(object):
    INPUT_SHAPE = (160, 320, 3)
    CS_CONV = cv2.COLOR_BGR2YUV
    CLIP = [60, 20]
    RESIZE = (200, 66)

Params = Params()

def output_shape(clip = Params.CLIP, resize = Params.RESIZE):
    
    shape = Params.INPUT_SHAPE

    if resize is not None:
        shape = (resize[1], resize[0], Params.INPUT_SHAPE[2])
    elif clip is not None:
        shape = (Params.INPUT_SHAPE[0] - clip[0] - clip[1], Params.INPUT_SHAPE[1], Params.INPUT_SHAPE[2])

    return shape

def process_image(img, 
                  cs_conv = Params.CS_CONV,
                  clip = Params.CLIP,
                  resize = Params.RESIZE):
    if cs_conv is not None:
        img = cv2.cvtColor(img, cs_conv)
    if clip is not None:
        img = img[clip[0]:img.shape[0] - clip[1],:,:]
    if resize is not None:
        img = cv2.resize(img, resize, interpolation = cv2.INTER_AREA)

    return img
