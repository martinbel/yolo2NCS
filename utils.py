from backend import (TinyYoloFeature, FullYoloFeature, MobileNetFeature, SqueezeNetFeature,
                    Inception3Feature, VGG16Feature, ResNet50Feature, BaseFeatureExtractor,
                    SuperTinyYoloFeature, DarknetReferenceFeature)
from datetime import datetime
import xml.etree.ElementTree as ET
import numpy as np
import os
import tensorflow as tf
import copy
import cv2
import sys
import shutil
from scipy.special import expit

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.c     = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score


class WeightReader:
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file, dtype='float32')

    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size:self.offset]

    def reset(self):
        self.offset = 4


def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin,     box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin

    union = w1*h1 + w2*h2 - intersect

    return float(intersect) / union


def draw_boxes(image, boxes, labels):
    image_h, image_w, _ = image.shape

    color_levels = [0,255,128,64,32]
    colors = []
    for r in color_levels:
        for g in color_levels:
            for b in color_levels:
                if r==g and r==b: #prevent grayscale colors
                    continue
                colors.append((b,g,r))

    for box in boxes:
        xmin = int(box.xmin*image_w)
        ymin = int(box.ymin*image_h)
        xmax = int(box.xmax*image_w)
        ymax = int(box.ymax*image_h)

        line_width_factor = int(min(image_h,image_w) * 0.005)
        cv2.rectangle(image, (xmin,ymin), (xmax,ymax), colors[box.get_label()], line_width_factor * 2)
        cv2.putText(image,
                    "{:.2f} {}".format(box.get_score(), labels[box.get_label()]),
                    (xmin, ymin - line_width_factor * 3),
                    cv2.FONT_HERSHEY_PLAIN,
                    3e-3 * min(image_h, image_w),
                    (255,255,0), line_width_factor)

    return image

# Malisiewicz et al.
def non_max_suppression(boxes, overlapThresh=0.5):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    boxes_array = np.array([(x.xmin, x.ymin, x.xmax, x.ymax) for x in boxes])

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes_array.dtype.kind == "i":
        boxes_array = boxes_array.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes_array[:,0]
    y1 = boxes_array[:,1]
    x2 = boxes_array[:,2]
    y2 = boxes_array[:,3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    selected_bbox = [boxes[p] for p in pick]

    return selected_bbox


def decode_netout(netout, anchors, nb_class, obj_threshold=0.50, nms_threshold=0.5):
    grid_h, grid_w, nb_box = netout.shape[:3]

    boxes = []
    ### decode the output by the network
    # 4th index has the object probability, we take the sigmoid of it.
    netout[..., 4]  = expit(netout[..., 4])
    # Prob object exists (index 4) * Sofmax(each of presence of class, index 5:)
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * _softmax(netout[..., 5:])
    netout[..., 5:] *= netout[..., 5:] > obj_threshold

    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):
                # from 4th element onwards are confidence and class classes
                classes = netout[row, col, b, 5:]
                if np.sum(classes) > 0:
                    # first 4 elements are x, y, w, and h
                    x, y, w, h = netout[row,col,b,:4]
                    x = (col + expit(x)) / grid_w # center position, unit: image width
                    y = (row + expit(y)) / grid_h # center position, unit: image height
                    w = anchors[2 * b + 0] * np.exp(w) / grid_w # unit: image width
                    h = anchors[2 * b + 1] * np.exp(h) / grid_h # unit: image height
                    confidence = netout[row,col,b,4]
                    box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, confidence, classes)
                    boxes.append(box)

    # remove the boxes which are less likely than a obj_threshold
    boxes = [box for box in boxes if box.get_score() > obj_threshold]
    # TODO: add support for more classes
    # suppress non-maximal boxes
#    for c in range(nb_class):
        #sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))
#        box =
    selected = non_max_suppression(boxes, overlapThresh=nms_threshold)

    if len(selected) == 0:
        selected = []

    return selected


def compute_overlap(a, b):
    """
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def _softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)

    if np.min(x) < t:
        x = x/np.min(x)*t

    e_x = np.exp(x)

    return e_x / e_x.sum(axis, keepdims=True)

def import_dynamically(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod



def import_classifier(backend, input_size):
    if backend == 'Darknet Reference':
        classifier = DarknetReferenceFeature(input_size)
    return classifier


def import_feature_extractor(backend, input_size):
    if backend == 'Inception3':
        feature_extractor = Inception3Feature(input_size)
    elif backend == 'SqueezeNet':
        feature_extractor = SqueezeNetFeature(input_size)
    elif backend == 'MobileNet':
        feature_extractor = MobileNetFeature(input_size)
    elif backend == 'Full Yolo':
        feature_extractor = FullYoloFeature(input_size)
    elif backend == 'Tiny Yolo Voc':
        feature_extractor = TinyYoloVocFeature(input_size)
    elif backend == 'Tiny Yolo':
        feature_extractor = TinyYoloFeature(input_size)
    elif backend == 'VGG16':
        feature_extractor = VGG16Feature(input_size)
    elif backend == 'ResNet50':
        feature_extractor = ResNet50Feature(input_size)
    elif backend == 'Super Tiny Yolo':
        feature_extractor = SuperTinyYoloFeature(input_size)
    elif backend == 'Darknet Reference':
        feature_extractor = DarknetReferenceFeature(input_size)
    elif os.path.dirname(backend) != "":
        basePath = os.path.dirname(backend)
        sys.path.append(basePath)
        custom_backend_name = os.path.basename(backend)
        custom_backend = import_dynamically(custom_backend_name)
        feature_extractor = custom_backend(input_size)
        if not issubclass(custom_backend, BaseFeatureExtractor):
            raise RuntimeError('You are trying to import a custom backend, your backend must'
            ' be in inherited from "backend.BaseFeatureExtractor".')
        print('Using a custom backend called {}.'.format(custom_backend_name))
    else:
        raise RuntimeError('Architecture not supported! Only support Full Yolo, Tiny Yolo, MobileNet,'
            'SqueezeNet, VGG16, ResNet50, or Inception3 at the moment!')

    return feature_extractor

#these funcition are from imutils, you can check this library here: https://github.com/jrosebr1/imutils
#just added this function to have less dependencies
def list_images(basePath, validExts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"), contains=None):
    # return the set of files that are valid
    return list_files(basePath, validExts, contains=contains)

def list_files(basePath, validExts=(""), contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if ext.endswith(validExts):
                # construct the path to the image and yield it
                imagePath = os.path.join(rootDir, filename)
                yield imagePath

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def create_backup(config):
    backup_folder = config['backup']['backup_path']
    prefix = config['backup']['backup_prefix']
    backup_id = datetime.now().strftime('%d%H%M')
    train_folder_name = "_".join([prefix,backup_id])
    path = os.path.join(backup_folder,train_folder_name)
    if os.path.isdir(path) :
        shutil.rmtree(path)
    os.makedirs(path)

    shutil.copytree(os.path.dirname(os.path.realpath(__file__)), os.path.join(path,"Keras-yolo2"), ignore=shutil.ignore_patterns(".git"))
    readme_message = ""
    #while(readme_message == ""):
    #    readme_message = input("Insert a comment about this training: ")
    with open(os.path.join(path,"readme.txt"),'w') as readme_file:
        readme_file.write(readme_message)

    if config['backup']['redirect_model']:
        model_name = ".".join([train_folder_name,"h5"])
        model_name = os.path.join(path, model_name)
        log_name = os.path.join(path,"logs")
        print('\n\nRedirecting {} file name to {}.'.format(config['train']['saved_weights_name'],model_name))
        print('Redirecting {} tensorborad log to {}.'.format(config['train']['tensorboard_log_dir'],log_name))
        config['train']['saved_weights_name'] = model_name
        config['train']['tensorboard_log_dir'] = log_name

    return config





def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
    """Filters YOLO boxes by thresholding on object and class confidence.

    Arguments:
    box_confidence -- tensor of shape (grid, grid, nb_anchor, class)
    boxes -- tensor of shape (grid, grid, nb_anchor, nb_box)
    box_class_probs -- tensor of shape (grid, grid, nb_anchor, nb_classes)
    threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box

    Returns:
    scores -- tensor of shape (None,), containing the class probability score for selected boxes
    boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
    classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes

    Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold.
    For example, the actual output size of scores would be (10,) if there are 10 boxes.
    """

    # Step 1: Compute box scores
    box_scores = box_confidence * box_class_probs # 19x19x80

    # Step 2: Find the box_classes thanks to the max box_scores, keep track of the corresponding score
    box_classes = np.argmax(box_scores, axis = -1)  # 19x19x5x1  (1 class index)
    box_class_scores = np.max(box_scores, axis = -1)  # 19x19x5x1 (1 class score)

    # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
    # same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
    filtering_mask = box_class_scores >= threshold

    # Step 4: Apply the mask to scores, boxes and classes
    scores = box_class_scores[filtering_mask]
    boxes = boxes[filtering_mask]
    classes = box_classes[filtering_mask]

    return scores, boxes, classes

def iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2

    Arguments:
    box1 -- first box, list object with coordinates (x1, y1, x2, y2)
    box2 -- second box, list object with coordinates (x1, y1, x2, y2)
    """

    # Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
    ### START CODE HERE ### (≈ 5 lines)
    xi1 = np.max([box1[0], box2[0]])
    yi1 = np.max([box1[1], box2[1]])
    xi2 = np.min([box1[2], box2[2]])
    yi2 = np.min([box1[3],box2[3]])
    inter_area = max((yi2-yi1),0) * max((xi2-xi1),0)
    ### END CODE HERE ###

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    ### START CODE HERE ### (≈ 3 lines)
    box1_area = (box1[3]-box1[1]) * (box1[2] - box1[0])
    box2_area = (box2[3]-box2[1]) * (box2[2] - box2[0])
    union_area = box1_area + box2_area - inter_area
    ### END CODE HERE ###

    # compute the IoU
    ### START CODE HERE ### (≈ 1 line)
    iou = inter_area/union_area
    ### END CODE HERE ###

    return iou
