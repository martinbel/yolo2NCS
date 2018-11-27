#! /usr/bin/env python3
from tqdm import tqdm
from utils import draw_boxes, get_session
from frontend import YOLO
from utils import list_images
import tensorflow as tf
import numpy as np
import argparse
import os
import cv2
import keras
import json
import time

from imutils.video import FileVideoStream, VideoStream, FPS

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    default='config.json',
    help='path to configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    default='',
    help='path to pretrained weights')

argparser.add_argument(
    '-i',
    '--input',
    help='path to an image or an video (mp4 format)')

def _main_(args):
    config_path  = args.conf
    weights_path = args.weights
    image_path   = args.input

    keras.backend.tensorflow_backend.set_session(get_session())

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    if weights_path == '':
        weights_path = config['train']['pretrained_weights"']

    ###############################
    #   Make the model
    ###############################

    input_size = (config['model']['input_size_h'],config['model']['input_size_w'])

    yolo = YOLO(backend             = config['model']['backend'],
                input_size          = (config['model']['input_size_h'],config['model']['input_size_w']),
                labels              = config['model']['labels'],
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'],
                gray_mode           = config['model']['gray_mode'])

    if config['model']['gray_mode']:
        depth = 1
    else:
        depth = 3

    yolo.load_weights(weights_path)

    if image_path[-4:] == '.mp4':
        video_out = image_path[:-4] + '_detected' + image_path[-4:]
        #cap = FileVideoStream(image_path).start()
        cap = cv2.VideoCapture(image_path)
        time.sleep(1.0)
#        fps = FPS().start()
        fps_img = 0.0
        counter = 0
        while True:
            start = time.time()
            ret, image = cap.read()

            if depth == 1:
                # convert video to gray
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, input_size, interpolation = cv2.INTER_CUBIC)
                image = np.expand_dims(image, 2)
                #image = np.array(image, dtype='f')
            else:
                if counter == 1:
                    print("Color image")
                image = cv2.resize(image, input_size, interpolation = cv2.INTER_CUBIC)
                #image = np.array(image, dtype='f')

            #image = np.divide(image, 255.)
            tm_inf = time.time()
            boxes = yolo.predict(image)
            fps_img  = ( fps_img + ( 1 / (time.time() - start) ) ) / 2

            print( "Inference time: {:.4f}".format(time.time() - tm_inf) )
            image = draw_boxes(image, boxes, config['model']['labels'])
            image = cv2.putText(image, "fps: {:.2f}".format(fps_img), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, 4 )
            cv2.imshow("Press q to quit", image)
#            fps.update()

            #if counter == 10:
                #print(image.sum(), boxes)
            #    time.sleep(1)
            counter += 1

            if cv2.getWindowProperty( "Press q to quit", cv2.WND_PROP_ASPECT_RATIO ) < 0.0:
                print("Window closed" )
                break
            elif cv2.waitKey( 1 ) & 0xFF == ord( 'q' ):
                print( "Q pressed" )
                break
#        fps.stop()
#        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
#        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        cap.release()

    else:
        images = list(list_images(image_path))
        for fname in images[100:]:
            image = cv2.imread(fname)
            tm_inf = time.time()
            boxes = yolo.predict(image)
            print( "Inference time: {:.4f}".format(time.time() - tm_inf) )
            image = draw_boxes(image, boxes, config['model']['labels'])
            cv2.imshow("Press q to quit", image)

            if cv2.waitKey( 1 ) & 0xFF == ord( 'q' ):
                break
            time.sleep(2)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
