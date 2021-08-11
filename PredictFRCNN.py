from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import sys
import time
from PyQt5 import QtCore, QtWidgets
import numpy as np
import pickle
import math
import cv2
import tensorflow as tf
from natsort import natsorted
import os

from keras import backend as K
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import TimeDistributed
from keras.models import Model
import RoiPoolingConv
from config import Config


class AI:
    def __init__(self,GUI):
        self.GUI = GUI
        self.real_Position = []
        

    def setTemp(self,temp):
        self.temp = temp
    def rpn_to_roi(self,rpn_layer, regr_layer, C, dim_ordering, use_regr=True, max_boxes=300,overlap_thresh=0.9):
        """Convert rpn layer to roi bboxes

        Args: (num_anchors = 9)
            rpn_layer: output layer for rpn classification 
                shape (1, feature_map.height, feature_map.width, num_anchors)
                Might be (1, 18, 25, 18) if resized image is 400 width and 300
            regr_layer: output layer for rpn regression
                shape (1, feature_map.height, feature_map.width, num_anchors)
                Might be (1, 18, 25, 72) if resized image is 400 width and 300
            C: config
            use_regr: Wether to use bboxes regression in rpn
            max_boxes: max bboxes number for non-max-suppression (NMS)
            overlap_thresh: If iou in NMS is larger than this threshold, drop the box

        Returns:
            result: boxes from non-max-suppression (shape=(300, 4))
                boxes: coordinates for bboxes (on the feature map)
        """
        regr_layer = regr_layer / C.std_scaling

        anchor_sizes = C.anchor_box_scales   # (3 in here)
        anchor_ratios = C.anchor_box_ratios  # (3 in here)

        assert rpn_layer.shape[0] == 1

        (rows, cols) = rpn_layer.shape[1:3]

        curr_layer = 0

        # A.shape = (4, feature_map.height, feature_map.width, num_anchors) 
        # Might be (4, 18, 25, 18) if resized image is 400 width and 300
        # A is the coordinates for 9 anchors for every point in the feature map 
        # => all 18x25x9=4050 anchors cooridnates
        A = np.zeros((4, rpn_layer.shape[1], rpn_layer.shape[2], rpn_layer.shape[3]))

        for anchor_size in anchor_sizes:
            for anchor_ratio in anchor_ratios:
                # anchor_x = (128 * 1) / 16 = 8  => width of current anchor
                # anchor_y = (128 * 2) / 16 = 16 => height of current anchor
                anchor_x = (anchor_size * anchor_ratio[0])/C.rpn_stride
                anchor_y = (anchor_size * anchor_ratio[1])/C.rpn_stride
                
                # curr_layer: 0~8 (9 anchors)
                # the Kth anchor of all position in the feature map (9th in total)
                regr = regr_layer[0, :, :, 4 * curr_layer:4 * curr_layer + 4] # shape => (18, 25, 4)
                regr = np.transpose(regr, (2, 0, 1)) # shape => (4, 18, 25)

                # Create 18x25 mesh grid
                # For every point in x, there are all the y points and vice versa
                # X.shape = (18, 25)
                # Y.shape = (18, 25)
                X, Y = np.meshgrid(np.arange(cols),np. arange(rows))

                # Calculate anchor position and size for each feature map point
                A[0, :, :, curr_layer] = X - anchor_x/2 # Top left x coordinate
                A[1, :, :, curr_layer] = Y - anchor_y/2 # Top left y coordinate
                A[2, :, :, curr_layer] = anchor_x       # width of current anchor
                A[3, :, :, curr_layer] = anchor_y       # height of current anchor

                # Apply regression to x, y, w and h if there is rpn regression layer
                if use_regr:
                    A[:, :, :, curr_layer] = self.apply_regr_np(A[:, :, :, curr_layer], regr)

                # Avoid width and height exceeding 1
                A[2, :, :, curr_layer] = np.maximum(1, A[2, :, :, curr_layer])
                A[3, :, :, curr_layer] = np.maximum(1, A[3, :, :, curr_layer])

                # Convert (x, y , w, h) to (x1, y1, x2, y2)
                # x1, y1 is top left coordinate
                # x2, y2 is bottom right coordinate
                A[2, :, :, curr_layer] += A[0, :, :, curr_layer]
                A[3, :, :, curr_layer] += A[1, :, :, curr_layer]

                # Avoid bboxes drawn outside the feature map
                A[0, :, :, curr_layer] = np.maximum(0, A[0, :, :, curr_layer])
                A[1, :, :, curr_layer] = np.maximum(0, A[1, :, :, curr_layer])
                A[2, :, :, curr_layer] = np.minimum(cols-1, A[2, :, :, curr_layer])
                A[3, :, :, curr_layer] = np.minimum(rows-1, A[3, :, :, curr_layer])

                curr_layer += 1

        all_boxes = np.reshape(A.transpose((0, 3, 1, 2)), (4, -1)).transpose((1, 0))  # shape=(4050, 4)
        all_probs = rpn_layer.transpose((0, 3, 1, 2)).reshape((-1))                   # shape=(4050,)

        x1 = all_boxes[:, 0]
        y1 = all_boxes[:, 1]
        x2 = all_boxes[:, 2]
        y2 = all_boxes[:, 3]

        # Find out the bboxes which is illegal and delete them from bboxes list
        idxs = np.where((x1 - x2 >= 0) | (y1 - y2 >= 0))

        all_boxes = np.delete(all_boxes, idxs, 0)
        all_probs = np.delete(all_probs, idxs, 0)

        # Apply non_max_suppression
        # Only extract the bboxes. Don't need rpn probs in the later process
        result = self.non_max_suppression_fast(all_boxes, all_probs, overlap_thresh=overlap_thresh, max_boxes=max_boxes)[0]

        return result

    def apply_regr_np(self,X, T):
        """Apply regression layer to all anchors in one feature map

        Args:
            X: shape=(4, 18, 25) the current anchor type for all points in the feature map
            T: regression layer shape=(4, 18, 25)

        Returns:
            X: regressed position and size for current anchor
        """
        try:
            x = X[0, :, :]
            y = X[1, :, :]
            w = X[2, :, :]
            h = X[3, :, :]

            tx = T[0, :, :]
            ty = T[1, :, :]
            tw = T[2, :, :]
            th = T[3, :, :]

            cx = x + w/2.
            cy = y + h/2.
            cx1 = tx * w + cx
            cy1 = ty * h + cy

            w1 = np.exp(tw.astype(np.float32)) * w
            h1 = np.exp(th.astype(np.float32)) * h
            x1 = cx1 - w1/2.
            y1 = cy1 - h1/2.

            x1 = np.round(x1)
            y1 = np.round(y1)
            w1 = np.round(w1)
            h1 = np.round(h1)
            return np.stack([x1, y1, w1, h1])
        except Exception as e:
            print(e)
            return X

    def non_max_suppression_fast(self,boxes, probs, overlap_thresh=0.9, max_boxes=300):
        # code used from here: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
        # if there are no boxes, return an empty list

        # Process explanation:
        #   Step 1: Sort the probs list
        #   Step 2: Find the larget prob 'Last' in the list and save it to the pick list
        #   Step 3: Calculate the IoU with 'Last' box and other boxes in the list. If the IoU is larger than overlap_threshold, delete the box from list
        #   Step 4: Repeat step 2 and step 3 until there is no item in the probs list 
        if len(boxes) == 0:
            return []

        # grab the coordinates of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        np.testing.assert_array_less(x1, x2)
        np.testing.assert_array_less(y1, y2)

        # if the bounding boxes integers, convert them to floats --
        # this is important since we'll be doing a bunch of divisions
        if boxes.dtype.kind == "i":
            boxes = boxes.astype(np.float32)

        # initialize the list of picked indexes	
        pick = []

        # calculate the areas
        area = (x2 - x1) * (y2 - y1)

        # sort the bounding boxes 
        idxs = np.argsort(probs)

        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # find the intersection

            xx1_int = np.maximum(x1[i], x1[idxs[:last]])
            yy1_int = np.maximum(y1[i], y1[idxs[:last]])
            xx2_int = np.minimum(x2[i], x2[idxs[:last]])
            yy2_int = np.minimum(y2[i], y2[idxs[:last]])

            ww_int = np.maximum(0, xx2_int - xx1_int)
            hh_int = np.maximum(0, yy2_int - yy1_int)

            area_int = ww_int * hh_int

            # find the union
            area_union = area[i] + area[idxs[:last]] - area_int

            # compute the ratio of overlap
            overlap = area_int/(area_union + 1e-6)

            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last],
                np.where(overlap > overlap_thresh)[0])))

            if len(pick) >= max_boxes:
                break

        # return only the bounding boxes that were picked using the integer data type
        boxes = boxes[pick].astype("int")
        probs = probs[pick]
        return boxes, probs

    def nn_base(self,input_tensor=None, trainable=False):

        input_shape = (None, None, 3)
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

        bn_axis = 3

        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        return x

    def rpn_layer(self,base_layers, num_anchors):
        """Create a rpn layer
            Step1: Pass through the feature map from base layer to a 3x3 512 channels convolutional layer
                    Keep the padding 'same' to preserve the feature map's size
            Step2: Pass the step1 to two (1,1) convolutional layer to replace the fully connected layer
                    classification layer: num_anchors (9 in here) channels for 0, 1 sigmoid activation output
                    regression layer: num_anchors*4 (36 in here) channels for computing the regression of bboxes with linear activation
        Args:
            base_layers: vgg in here
            num_anchors: 9 in here

        Returns:
            [x_class, x_regr, base_layers]
            x_class: classification for whether it's an object
            x_regr: bboxes regression
            base_layers: vgg in here
        """
        x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)

        x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
        x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

        return [x_class, x_regr, base_layers]

    def classifier_layer(self,base_layers, input_rois, num_rois, nb_classes = 4, trainable=False):
        """Create a classifier layer
        
        Args:
            base_layers: vgg
            input_rois: `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
            num_rois: number of rois to be processed in one time (4 in here)

        Returns:
            list(out_class, out_regr)
            out_class: classifier layer output
            out_regr: regression layer output
        """

        input_shape = (num_rois,7,7,512)

        pooling_regions = 7

        # out_roi_pool.shape = (1, num_rois, channels, pool_size, pool_size)
        # num_rois (4) 7x7 roi pooling
        out_roi_pool = RoiPoolingConv.RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])

        # Flatten the convlutional layer and connected to 2 FC and 2 dropout
        out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
        out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
        out = TimeDistributed(Dropout(0.5))(out)
        out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)
        out = TimeDistributed(Dropout(0.5))(out)

        # There are two output layer
        # out_class: softmax acivation function for classify the class name of the object
        # out_regr: linear activation function for bboxes coordinates regression
        out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
        # note: no regression target for bg class
        out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)

        return [out_class, out_regr]

    def format_img_size(self,img, C):
        """ formats the image size based on config """
        img_min_side = float(C.im_size)
        (height,width,_) = img.shape
            
        if width <= height:
            ratio = img_min_side/width
            new_height = int(ratio * height)
            new_width = int(img_min_side)
        else:
            ratio = img_min_side/height
            new_width = int(ratio * width)
            new_height = int(img_min_side)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        return img, ratio	

    def format_img_channels(self,img, C):
        """ formats the image channels based on config """
        img = img[:, :, (2, 1, 0)]
        img = img.astype(np.float32)
        img[:, :, 0] -= C.img_channel_mean[0]
        img[:, :, 1] -= C.img_channel_mean[1]
        img[:, :, 2] -= C.img_channel_mean[2]
        img /= C.img_scaling_factor
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def format_img(self,img, C):
        """ formats an image for model prediction based on config """
        img, ratio = self.format_img_size(img, C)
        img = self.format_img_channels(img, C)
        return img, ratio

    def get_real_coordinates(self,ratio, x1, y1, x2, y2):

        real_x1 = int(round(x1 // ratio))
        real_y1 = int(round(y1 // ratio))
        real_x2 = int(round(x2 // ratio))
        real_y2 = int(round(y2 // ratio))

        return (real_x1, real_y1, real_x2 ,real_y2)

    def apply_regr(self,x, y, w, h, tx, ty, tw, th):
        # Apply regression to x, y, w and h
        try:
            cx = x + w/2.
            cy = y + h/2.
            cx1 = tx * w + cx
            cy1 = ty * h + cy
            w1 = math.exp(tw) * w
            h1 = math.exp(th) * h
            x1 = cx1 - w1/2.
            y1 = cy1 - h1/2.
            x1 = int(round(x1))
            y1 = int(round(y1))
            w1 = int(round(w1))
            h1 = int(round(h1))

            return x1, y1, w1, h1

        except ValueError:
            return x, y, w, h
        except OverflowError:
            return x, y, w, h
        except Exception as e:
            print(e)
            return x, y, w, h

    def TEMP_DIRS_POHOTO(self,_VIDEO_FILE):
        dirname = "temp"
        dirname2 = "tempVideo"
        try:
            os.makedirs(dirname)
            os.makedirs(dirname2)
        except OSError:
            if os.path.exists(dirname) or os.path.exists(dirname2):
                pass
            else:
                raise
        i = 0
        j = 0
        while True:
            cap = cv2.VideoCapture(_VIDEO_FILE,0) 
            cap.set(1,i); 
            eat, img = cap.read()
            cv2.imwrite("temp/0{}.jpg".format(j),img)
            print(i)
            i = i+5
            j = j + 1
            if i > 800:
                break
        return j

    def CONFIG_SESSION(self):
        sys.setrecursionlimit(40000)
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True,inter_op_parallelism_threads=1,intra_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        sess = K.set_session(tf.compat.v1.Session(config=config))
        K.set_session(sess)

    def PREDECT_PHOTO(self):
        
        self.GUI.READ_FILE_TEMP_TEXT()

        filename = 'model/model_vgg_config.pickle'
        with open(filename, 'rb') as f_in:
            C = pickle.load(f_in)

        if C.network == 'resnet50':
            print(1)
        elif C.network == 'vgg':
            print(2)

        C.use_horizontal_flips = False
        C.use_vertical_flips = False
        C.rot_90 = False


        img_path =  "temp"
        class_mapping = C.class_mapping
        num_rois = 4

        if 'bg' not in class_mapping:
            class_mapping['bg'] = len(class_mapping)

        class_mapping = {v: k for k, v in class_mapping.items()}
        print(class_mapping)
        class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
        C.num_rois = int(num_rois)

        if C.network == 'resnet50':
            num_features = 1024
        elif C.network == 'vgg':
            num_features = 512

        if K.image_data_format() == 'th':
            input_shape_img = (3, None, None)
            input_shape_features = (num_features, None, None)
        else:
            input_shape_img = (None, None, 3)
            input_shape_features = (None, None, num_features)

        

        img_input = Input(shape=input_shape_img)
        roi_input = Input(shape=(C.num_rois, 4))
        feature_map_input = Input(shape=input_shape_features)

        # define the base network (resnet here, can be VGG, Inception, etc)
        shared_layers = self.nn_base(img_input, trainable=True)

        # define the RPN, built on the base layers
        num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
        rpn_layers = self.rpn_layer(shared_layers, num_anchors)

        classifier = self.classifier_layer(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

        model_rpn = Model(img_input, rpn_layers)
        model_classifier_only = Model([feature_map_input, roi_input], classifier)

        model_classifier = Model([feature_map_input, roi_input], classifier)

        C.model_path = "model/model_frcnn_vgg.hdf5"
        print(f'Loading weights from {C.model_path}')
        model_rpn.load_weights(C.model_path, by_name=True)
        model_classifier.load_weights(C.model_path, by_name=True)

        model_rpn.compile(optimizer='sgd', loss='mse')
        model_classifier.compile(optimizer='sgd', loss='mse')

        all_imgs = []
        all_forVideo = []

        classes = {}

        bbox_threshold = 0.8

        visualise = True
        
        for idx, img_name in enumerate(natsorted(os.listdir(img_path))):
            if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
                continue
            print(img_name)
            st = time.time()
            filepath = os.path.join(img_path,img_name)
            img = cv2.imread(filepath)
            X, ratio = self.format_img(img, C)
            img_scaled = (np.transpose(X[0,:,:,:],(1,2,0)) + 127.5).astype('uint8')
            # if K.image_data_format() == 'tf':
            X = np.transpose(X, (0, 2, 3, 1))

            # get the feature maps and output from the RPN
            [Y1, Y2, F] = model_rpn.predict(X)
            
            R = self.rpn_to_roi(Y1, Y2, C, K.image_data_format(), overlap_thresh=0.7)
            print(R.shape)
                
                # convert from (x1,y1,x2,y2) to (x,y,w,h)
            R[:, 2] -= R[:, 0]
            R[:, 3] -= R[:, 1]

                # apply the spatial pyramid pooling to the proposed regions
            bboxes = {}
            probs = {}
            
            for jk in range(R.shape[0]//num_rois + 1):
                ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
                if ROIs.shape[1] == 0:
                    break
                    
                if jk == R.shape[0]//num_rois:
                        #pad R
                    curr_shape = ROIs.shape
                    target_shape = (curr_shape[0],num_rois,curr_shape[2])
                    ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                    ROIs_padded[:,:curr_shape[1],:] = ROIs
                    ROIs_padded[0,curr_shape[1]:,:] = ROIs[0,0,:]
                    ROIs = ROIs_padded
                    
                [P_cls,P_regr] = model_classifier.predict([F, ROIs])
                
                for ii in range(P_cls.shape[1]):
                    if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                        continue

                    cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

                    if cls_name not in bboxes:
                        bboxes[cls_name] = []
                        probs[cls_name] = []
                    (x,y,w,h) = ROIs[0,ii,:]

                    bboxes[cls_name].append([16*x,16*y,16*(x+w),16*(y+h)])
                    probs[cls_name].append(np.max(P_cls[0,ii,:]))
                    cls_num = np.argmax(P_cls[0, ii, :])
                    try:
                        (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                        tx /= C.classifier_regr_std[0]
                        ty /= C.classifier_regr_std[1]
                        tw /= C.classifier_regr_std[2]
                        th /= C.classifier_regr_std[3]
                        x, y, w, h = self.apply_regr(x, y, w, h, tx, ty, tw, th)
                    except:
                        pass
                    bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
                    probs[cls_name].append(np.max(P_cls[0, ii, :]))

            all_dets = []
            temp_real_Position = []
            for key in bboxes:
                
                bbox = np.array(bboxes[key])
                
                new_boxes, new_probs = self.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh = 0.3)
                for jk in range(new_boxes.shape[0]):
                    (x1, y1, x2, y2) = new_boxes[jk,:]
                    (real_x1, real_y1, real_x2, real_y2) = self.get_real_coordinates(ratio, x1, y1, x2, y2)

                    imgforGUI = img[real_y1:real_y2, real_x1:real_x2]
                    cv2.imwrite('detectOut/{}.png'.format(idx),imgforGUI)
                    temp_real_Position.append([real_x1, real_y1, real_x2, real_y2])
                    
                    
                    cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)
                    
                    textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
                    all_dets.append((key,100*new_probs[jk]))
                    
                    (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
                    textOrg = (real_x1, real_y1-0)
                    
                    cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
                    cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
                    cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)



            self.real_Position.append(temp_real_Position)
            cv2.imwrite('tempVideo/{}.png'.format(idx),img)
            all_forVideo.append(img)
            self.GUI.SET_IMG_OR()
            self.GUI.SET_IMG_DETECT(img)
            self.GUI.canvas_1.SetArrayPlotUpdate(self.real_Position)
            self.GUI.canvas.SetArrayPlotUpdate(self.real_Position)
        self.GUI.canvas_1.SetArrayPlotUpdate(self.real_Position)
        self.GUI.canvas.SetArrayPlotUpdate(self.real_Position)

        

