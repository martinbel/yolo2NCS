#! /usr/bin/env python3
from tqdm import tqdm
from utils import draw_boxes, get_session
from frontend import YOLO
from utils import *
import tensorflow as tf
import numpy as np
import argparse
import os
import cv2
import keras
import json
import time
from ncsmodel.DarknetReferenceNet import Net

from imutils.video import FileVideoStream, VideoStream, FPS

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"




def _main_():
    config_path = 'ncsmodel/config.json'
    image_path = 'videos/uba.mp4'
    keras.backend.tensorflow_backend.set_session(get_session())

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    ###############################
    #   Make the model
    ###############################

    input_size = (config['model']['input_size_h'],config['model']['input_size_w'])
    labels = config['model']['labels']
    max_box_per_image   = config['model']['max_box_per_image']
    anchors = config['model']['anchors']
    gray_mode = config['model']['gray_mode']
    nb_class = 1

    # Net Config
    NN = Net(load_weights=True)
    cv2_image = cv2.imread("images/person.jpg", 0)
    image = NN.image

    if config['model']['gray_mode']:
        depth = 1
    else:
        depth = 3

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if image_path[-4:] == '.mp4':
            video_out = image_path[:-4] + '_detected' + image_path[-4:]
            cap = FileVideoStream(image_path).start()
            time.sleep(1.0)
            fps = FPS().start()
            fps_img = 0.0
            counter = 0
            tf_image = NN.image

            while True:
                start = time.time()
                image = cap.read()
                print(image.min(), image.max(), image.mean(), image.shape)

                if depth == 1:
                    # convert video to gray
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    # resize for plotting
                    resized_image = cv2.resize(gray_image, input_size, interpolation = cv2.INTER_CUBIC)
                    # convert to float
                    image_data = np.array(resized_image, dtype=np.float16)
                    image_data /= 255.
                    # dimensions for inference (1, w, h, c)
                    image_data = np.expand_dims(np.expand_dims(image_data, 0), 3) # 1, 256, 256, 1
                    # dimension for plot (w, h, c)
                    plot_image = np.expand_dims(resized_image, 2)
                else:
                    if counter == 1:
                        print("Color image")
                    image_data = cv2.resize(image, input_size, interpolation = cv2.INTER_CUBIC)
                    #image = np.array(image, dtype='f')
                #image = np.divide(image, 255.)

                tm_inf = time.time()
                netout = sess.run(NN.predict(), feed_dict={tf_image: image_data})
                netout = np.reshape(np.squeeze(netout, axis=0), (8,8,5,6))
                boxes  = decode_netout(netout, anchors, nb_class)

                fps_img  = ( fps_img + ( 1 / (time.time() - start) ) ) / 2
                #print( "Inference time: {:.4f}".format(time.time() - tm_inf) )
                print(plot_image.shape)
                image = draw_boxes(plot_image, boxes, labels)
                #image = cv2.putText(image, "fps: {:.2f}".format(fps_img), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, 4 )
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
    _main_()
