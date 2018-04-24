import argparse
import os
import cv2

import numpy as np
import image_processor as ip

from tqdm import tqdm
from skimage import img_as_ubyte
from keras import backend as K
from keras.models import load_model
from moviepy.editor import ImageSequenceClip

IMAGE_EXT = ['jpeg', 'gif', 'png', 'jpg']

def process_layer(model, layer_name, image_list, out, fmap = 'max', scale_factor = 2):

    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layer = layer_dict[layer_name]
    
    print('Preprocessing {} images...'.format(len(image_list)))
    images = list(map(lambda img_file: ip.process_image(cv2.imread(img_file)), image_list))

    processed = []
    inputs = [K.learning_phase()] + model.inputs

    print('Running model...')
    output = K.function(inputs, [layer.output])([0] + [images])
    output = np.squeeze(output)
    
    for activation in tqdm(output, unit=' images', desc='fmaps processing'):
        
        if fmap == 'max':
            a_count = np.count_nonzero(activation, axis = (0,1))
            a_idx = a_count.argmax()
        else:
            a_idx = int(fmap)

        a_img = (np.clip(activation[:, :, a_idx] * 255.0 * 10.0, a_min = 0, a_max = 255)).astype('uint8')
        a_img = cv2.cvtColor(a_img, cv2.COLOR_GRAY2RGB)
        if not scale_factor == 1:
            a_img = cv2.resize(a_img, None, fx = scale_factor, fy = scale_factor, interpolation = cv2.INTER_CUBIC)

        processed.append(a_img)
        
    return processed

def create_video(images, file_path, fps = 60):
    print("Creating video {}, FPS={}".format(file_path, fps))
    clip = ImageSequenceClip(images, fps=fps)
    clip.write_videofile(file_path)

def main():
    parser = argparse.ArgumentParser(description='Processing model feature maps')
    parser.add_argument(
        'model',
        type=str,
        default='model.h5',
        help='Path to the model file.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        default='',
        help='Path to image folder. The video will be created from these images.'
    )
    parser.add_argument(
        '--layer_name',
        type=str,
        default='convolution2d_1',
        help='The name of the layer to use for visualization'
    )
    parser.add_argument(
        '--fmap',
        type=str,
        default='max',
        help='Index for the fmap to use, "max" for using the one whose values that are > 0 is higher'
    )
    parser.add_argument(
        '--out',
        type=str,
        default='fmaps',
        help='The output path the generated video (will append .mp4)'
    )
    parser.add_argument(
        '--scale',
        type=float,
        default=2,
        help='How much to scale the output'
    )
    args = parser.parse_args()

    print('Loading model...')
    model = load_model(args.model)
    model.summary()
    
    image_list = sorted([os.path.join(args.image_folder, image_file) for image_file in os.listdir(args.image_folder)])
    image_list = [image_file for image_file in image_list if os.path.splitext(image_file)[1][1:].lower() in IMAGE_EXT]
    
    images = process_layer(model, args.layer_name, image_list, args.out, fmap = args.fmap, scale_factor = args.scale)
    video_file = args.out + '.mp4'
    create_video(images, video_file)

if __name__ == '__main__':
    main()