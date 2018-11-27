#! /usr/bin/env python3
from mvnc import mvncapi as mvnc
import cv2
import argparse
import time
import numpy as np
from imutils.video import FileVideoStream, VideoStream
from imutils.video import FPS

from utils import draw_boxes, get_session, decode_netout
from frontend import YOLO
from utils import list_images
import tensorflow as tf
import argparse
import os
import cv2
import keras
import json
import time

from imutils.video import FileVideoStream, VideoStream, FPS

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


def _main_(args):
    # Set up device
    device = mvnc.Device(args['device'])
    device.open()
    fifoIn, fifoOut = graph.allocate_with_fifos(device, graphfile)

    # Configuration stuff needed for postprocessing the predictions
    config_path = args['config']
    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    backend = config['model']['backend']
    input_size = (config['model']['input_size_h'],config['model']['input_size_w'])
    labels = config['model']['labels']
    max_box_per_image = config['model']['max_box_per_image']
    anchors = config['model']['anchors']
    gray_mode = config['model']['gray_mode']
    nb_class = 1


    #   Predict bounding boxes
    if video_mode:
        cap = FileVideoStream(source).start()
        time.sleep(1.0)
        fps = FPS().start()
        fps_img = 0.0
        counter = 0
        while True:
            start = time.time()
            image = cap.read()

            # Preprocessing
            if depth == 1:
                # convert video to gray
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, input_size, interpolation = cv2.INTER_CUBIC)
                image = np.expand_dims(image, 2)
                fimage = np.array(image, dtype=np.float32)
            else:
                image = cv2.resize(image, input_size, interpolation = cv2.INTER_CUBIC)
                fimage = np.array(image, dtype=np.float32)

            #image = np.expand_dims(image, 0)
            fimage = np.divide(fimage, 255.)

            tm_inf = time.time()
            graph.queue_inference_with_fifo_elem(fifoIn, fifoOut, fimage, 'user object')
            prediction, _ = fifoOut.read_elem()

            #prediction = np.add(fimage, 128)
            grid = 8
            prediction = np.reshape(prediction, (grid, grid, 5, 4 + 1 + nb_class))
            # prediction = np.multiply(prediction, 255)

            fps_img  = ( fps_img + ( 1 / (time.time() - start) ) ) / 2
            print( "Inference time: {:.4f}".format(time.time() - tm_inf) )

            # predictions decoding
            boxes  = decode_netout(prediction, anchors, nb_class)
            image = draw_boxes(image, boxes, config['model']['labels'])

            image = cv2.putText(image, "fps: {:.2f}".format(fps_img), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, 4 )
            cv2.imshow("Press q to quit", image)
            fps.update()

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
        fps.stop()
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        cap.release()

    else:
        #detected_images_path = os.path.join(image_path, "detected")
        #if not os.path.exists(detected_images_path):
        #    os.mkdir(detected_images_path)
        images = list(list_images(source))
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
        #    fname = os.path.basename(fname)
        #    cv2.imwrite(os.path.join(image_path, "detected", fname), image)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    source = 'videos/uba.mp4'
    video_mode = True
    config_path  = 'ncs_darknetreference/config2.json'
    graph_file = 'ncs_darknetreference/graph'
    depth = 1
    mvnc.global_set_option(mvnc.GlobalOption.RW_LOG_LEVEL, 2)
    devices = mvnc.enumerate_devices()
    if len(devices) == 0:
        print( "No devices found..." )
        quit()

    with open(graph_file, mode='rb' ) as f:
        graphfile = f.read()
        graph = mvnc.Graph('graph')

    args = dict(source=source, video_mode=video_mode,
                device=devices[0], graph=graph,
                depth=depth, input_size=256, config=config_path)
    _main_(args)
