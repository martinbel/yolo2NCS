import numpy as np
import tensorflow as tf
import cv2
import os
import time
from frontend import YOLO
import json
import sys
from keras import backend as K
from keras.models import model_from_json
from keras.models import Model, Sequential
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from frontend import YOLO

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Net:
    classes = ["person"]
    colors = {"person": (56.4, 42.3, 0)}

    image = tf.placeholder(tf.float32, shape=[1, 256, 256, 1], name="Input")
    actual_width = 0.0
    actual_height = 0.0
    input_height = 256
    input_width = 256
    video = False
    cv_window_name = "Result - press q to quit"
    fps = 0.0
    seconds = 0

    def get_keras_model(self):
        config_path = 'ncsmodel/config.json'
        weights_path = 'ncsmodel/openimages_best3.h5'

        def weights_name(layer_list, name):
            for l in layer_list:
                if l.name == name:
                    return l.get_weights()

        with open(config_path) as config_buffer:
            config = json.load(config_buffer)

        yolo = YOLO(backend             = config['model']['backend'],
                    input_size          = (config['model']['input_size_h'], config['model']['input_size_w']),
                    labels              = config['model']['labels'],
                    max_box_per_image   = config['model']['max_box_per_image'],
                    anchors             = config['model']['anchors'],
                    gray_mode           = config['model']['gray_mode'])

        # Loading weights:
        yolo.load_weights(weights_path)
        ymodel = yolo.model

        ylayers = ymodel.layers[1].layers
        for l in ymodel.layers[2:]:
            ylayers.append(l)

        print("Layers:")
        for i,l in enumerate(ylayers): print(i, l.name)

        # Save weights to numpy - 9 conv layers
        # 1 based index
        q_conv_bn = 6
        layer_weights =  []
        for i in range(q_conv_bn):
            weights = weights_name(ylayers, 'conv_{}'.format(i+1))[0]
            biases = weights_name(ylayers, 'norm_{}'.format(i+1))
            layer_weights.append((biases, weights))

        weights, biases = weights_name(ylayers, 'DetectionLayer')
        layer_weights.append((biases, weights))
        return layer_weights


    def leaky_relu(self, x, alpha=0.1):
        return tf.nn.leaky_relu(x, alpha=alpha)

    def sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        out = e_x / e_x.sum()
        return out

    def max_pool_layer(self, input_tensor, kernel=2, stride=2, padding='VALID'):
        pooling_result = tf.nn.max_pool(input_tensor, ksize=[1,kernel, kernel, 1], strides=[1, stride, stride, 1], padding=padding)
        return pooling_result

    def load_keras_conv_bn(self, idx):
        if idx < 6:
            norm, weights = self.layer_weights[idx]
        #   0 = gamma (if scale == True)
        #   1 = beta (if center == True)
        #   2 = moving mean
        #   3 = moving variance
        gamma, beta, mean, var = norm
        for i in range(mean.shape[0]):
            scale = gamma[i] / np.sqrt(var[i] + 0.001)
            weights[:,:,:,i] = weights[:,:,:,i] * scale
            beta[i] = beta[i] - mean[i] * scale
        return beta, weights

    def load_keras_conv(self):
        biases, weights = self.layer_weights[6]
        return biases, weights

    def preprocess_image(self, input):
        #self.actual_height, self.actual_width, _ = input.shape
        resized_image = cv2.resize(input, (256, 256), interpolation = cv2.INTER_CUBIC)
        image_data = np.array(resized_image, dtype='f')
        image_data /= 255.
        image_array = np.expand_dims(image_data, 0)

        return resized_image, image_array


    def iou(self, boxA, boxB):
        # boxA = boxB = [x1,y1,x2,y2]

        # Determine the coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # Compute the area of intersection
        intersection_area = (xB - xA + 1) * (yB - yA + 1)

        # Compute the area of both rectangles
        boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # Compute the IOU
        iou = intersection_area / float(boxA_area + boxB_area - intersection_area)

        return iou



    def non_maximal_suppression(self, thresholded_predictions, iou_threshold):
        nms_predictions = []
        # Add the best B-Box because it will never be deleted
        nms_predictions.append(thresholded_predictions[0])
        # For each B-Box (starting from the 2nd) check its iou with the higher score B-Boxes
        # thresholded_predictions[i][0] = [x1,y1,x2,y2]
        i = 1
        while i < len(thresholded_predictions):
            n_boxes_to_check = len(nms_predictions)
            to_delete = False

            j = 0
            while j < n_boxes_to_check:
                curr_iou = self.iou(thresholded_predictions[i][0],nms_predictions[j][0])
                if(curr_iou > iou_threshold ):
                    to_delete = True

                j = j+1

            if to_delete == False:
                nms_predictions.append(thresholded_predictions[i])
            i = i+1

        return nms_predictions

    def postprocess(self, predictions, input, score_threshold, iou_threshold):

        anchors = [1,1, 1,5, 2,5, 2,4, 3,7]
        thresholded_predictions = []
        grid = 8
        nb_anchors = 5
        nb_classes = 1
        print(predictions.shape)
        predictions = np.reshape(predictions, (grid, grid, nb_anchors, 1+4+nb_classes))  # 13, 13, 5, 25
        print(predictions.shape)

        # IMPORTANT: Compute the coordinates and score of the B-Boxes by considering the parametrization of YOLOv2
        for row in range(grid):
            for col in range(grid):
                for b in range(nb_anchors):
                    tx, ty, tw, th, tc = predictions[row, col, b, :5]
                    center_x = (float(col) + self.sigmoid(tx)) * 32.0
                    center_y = (float(row) + self.sigmoid(ty)) * 32.0

                    roi_w = np.exp(tw) * anchors[2*b + 0] * 32.0
                    roi_h = np.exp(th) * anchors[2*b + 1] * 32.0

                    final_confidence = self.sigmoid(tc)

                    # Find best class
                    class_predictions = predictions[row, col, b, 5:]
                    class_predictions = self.softmax(class_predictions)
                    class_predictions = tuple(class_predictions)
                    best_class = class_predictions.index(max(class_predictions))
                    best_class_score = class_predictions[best_class]

                    # Compute the final coordinates on both axes
                    left   = int(center_x - (roi_w/2.))
                    right  = int(center_x + (roi_w/2.))
                    top    = int(center_y - (roi_h/2.))
                    bottom = int(center_y + (roi_h/2.))

                    if( (final_confidence * best_class_score) > score_threshold):
                        thresholded_predictions.append([ [left,top,right,bottom], final_confidence * best_class_score, self.classes[best_class] ])

        nms_predictions = []
        if len(thresholded_predictions) != 0:
            thresholded_predictions.sort(key=lambda tup: tup[1], reverse=True)
            nms_predictions = self.non_maximal_suppression(thresholded_predictions, iou_threshold)

            # Draw boxes with texts
            for i in range(len(nms_predictions)):
                color = self.colors[nms_predictions[i][2]]
                best_class_name = nms_predictions[i][2]
                textWidth = cv2.getTextSize(best_class_name, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0][0] + 10

                input = cv2.rectangle(input, (nms_predictions[i][0][0], nms_predictions[i][0][1]), (nms_predictions[i][0][2],nms_predictions[i][0][3]), color)
                input = cv2.rectangle(input, (nms_predictions[i][0][0], nms_predictions[i][0][1]), (nms_predictions[i][0][0]+textWidth, nms_predictions[i][0][1]+20), color, -1)
#                input_image = cv2.putText(input_image, best_class_name, (nms_predictions[i][0][0]+5, nms_predictions[i][0][1]+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, 4)
        return input, nms_predictions

    def __init__(self, load_weights=False):
        self.bn_epsilon = 1e-3
        self.layer_weights = self.get_keras_model()

        if load_weights:
            print("Loading weights and biases...")
            # 0 based index
            # Conv1 , 3x3, 3->16
            self.biases1, self.weights1 = self.load_keras_conv_bn(0)
            self.biases1 = tf.Variable(self.biases1, dtype=tf.float32)
            self.weights1 = tf.Variable(self.weights1, dtype=tf.float32)

            # Conv2 , 3x3, 16->32
            self.biases2, self.weights2 = self.load_keras_conv_bn(1)
            self.biases2 = tf.Variable(self.biases2, dtype=tf.float32)
            self.weights2 = tf.Variable(self.weights2, dtype=tf.float32)

            # Conv3 , 3x3, 32->64
            self.biases3, self.weights3 = self.load_keras_conv_bn(2)
            self.biases3 = tf.Variable(self.biases3, dtype=tf.float32)
            self.weights3 = tf.Variable(self.weights3, dtype=tf.float32)

            # Conv4 , 3x3, 64->128
            self.biases4, self.weights4 = self.load_keras_conv_bn(3)
            self.biases4 = tf.Variable(self.biases4, dtype=tf.float32)
            self.weights4 = tf.Variable(self.weights4, dtype=tf.float32)

            # Conv5 , 3x3, 128->256
            self.biases5, self.weights5 = self.load_keras_conv_bn(4)
            self.biases5 = tf.Variable(self.biases5, dtype=tf.float32)
            self.weights5 = tf.Variable(self.weights5, dtype=tf.float32)

            # Conv6 , 3x3, 256->512
            self.biases6, self.weights6 = self.load_keras_conv_bn(5)
            self.biases6 = tf.Variable(self.biases6, dtype=tf.float32)
            self.weights6 = tf.Variable(self.weights6, dtype=tf.float32)

            # Conv7 , 3x3, 512->1024
            self.biases7, self.weights7 = self.load_keras_conv()
            self.biases7 = tf.Variable(self.biases7, dtype=tf.float32)
            self.weights7 = tf.Variable(self.weights7, dtype=tf.float32)

            print("Biases and weights are loaded!!")

    def predict(self):
        conv1 = tf.add(tf.nn.conv2d(self.image, self.weights1, strides=[1, 1, 1, 1], padding='SAME'), self.biases1)
        conv1 = self.leaky_relu( conv1 )
        max1 = self.max_pool_layer( conv1 )

        conv2 = tf.add(tf.nn.conv2d(max1, self.weights2, strides=[1, 1, 1, 1], padding='SAME'), self.biases2)
        conv2 = self.leaky_relu( conv2 )
        max2 = self.max_pool_layer( conv2 )

        conv3 = tf.add(tf.nn.conv2d(max2, self.weights3, strides=[1, 1, 1, 1], padding='SAME'), self.biases3)
        conv3 = self.leaky_relu( conv3 )
        max3 = self.max_pool_layer( conv3 )

        conv4 = tf.add(tf.nn.conv2d(max3, self.weights4, strides=[1, 1, 1, 1], padding='SAME'), self.biases4)
        conv4 = self.leaky_relu( conv4 )
        max4 = self.max_pool_layer( conv4 )

        conv5 = tf.add(tf.nn.conv2d(max4, self.weights5, strides=[1, 1, 1, 1], padding='SAME'), self.biases5)
        conv5 = self.leaky_relu( conv5 )
        max5 = self.max_pool_layer( conv5 )

        conv6 = tf.add(tf.nn.conv2d(max5, self.weights6, strides=[1, 1, 1, 1], padding='SAME'), self.biases6)
        conv6 = self.leaky_relu( conv6 )
        max6 = self.max_pool_layer( conv6, stride=1, padding='SAME' )

        conv7 = tf.add(tf.nn.conv2d(max6, self.weights7, strides=[1, 1, 1, 1], padding='SAME'), self.biases7, name='Output')
        return conv7
