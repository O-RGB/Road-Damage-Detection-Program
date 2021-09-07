import time
import numpy as np
import pickle
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import backend as K
from keras.layers import Input

from keras.models import Model
from source.frcnn.keras_frcnn.roi_helpers import rpn_to_roi
from source.frcnn.keras_frcnn.vgg import classifier_layer, nn_base, rpn_layer
from source.frcnn.keras_frcnn.roi_helpers import non_max_suppression_fast,apply_regr
from source.frcnn.keras_frcnn.get_dataset import get_data,get_real_coordinates
from source.frcnn.keras_frcnn.data_generators import iou as ious
from source.frcnn.keras_frcnn import config


def PercisionAndRecall(obj,gt):
        print("Images\t\tDetections\tConfidences\t\tTP or FP\tIOU\t\tTP\t\tFP\t\tAcc TP\t\tAcc FP\t\tPercision\tRecall")
        AccTP,AccFP = 0,0
        TP,FP = 0,0
        Percision = []
        Recall = []
        for i in obj:
                i["TP"] = 0
                i["FP"] = 0
                i["AccTP"] = AccTP
                i["AccFP"] = AccFP
                if i["Result"] == "TP":
                        i["TP"] = 1
                        i["FP"] = 0
                        TP = TP  +  1
                        AccTP = AccTP + 1
                        i["AccTP"] = AccTP    
                elif i["Result"] == "FP":
                        i["TP"] = 0
                        i["FP"] = 1
                        FP = FP + 1
                        AccFP = AccFP + 1
                        i["AccFP"] = AccFP
                
                i["Percision"] = round(i["AccTP"]/(i["AccTP"]+i["AccFP"]),5)
                i["Recall"] = round(i["AccTP"]/gt,5)

                Percision.append(i["Percision"])
                Recall.append(i["Recall"])

                
                for j in i.keys():
                        print(i[j],end="\t\t")
                print()
        print()
        print("ตรวจจับได้ ",len(obj)," จุด")
        print("(TP + FN) มี GT จำนวน",gt," จุด")
        print("(TP) ทำนายถูก ",TP," จุด")
        print("(FP) ทำนายผิด ",FP," จุด")
        return Percision,Recall

def interp(Percision):
        interpolation = Percision.copy()

        for id,i in enumerate(interpolation):
                array =  interpolation[id:len(interpolation)]
                if interpolation[id] < np.max(array):
                        interpolation[id] = np.max(array)
        return interpolation

def averagePrecision(interpolation,Recall):
        points = set([x for x in interpolation  if interpolation .count(x) > 1])
        ap_all = 0
        for i in list(points):
                search = np.where(np.array(interpolation) == i)
                indexAll = np.array(search[0]).tolist()
                r2 = Recall[indexAll[len(indexAll)-1]]
                r1 = Recall[indexAll[0]]
                p = interpolation[indexAll[0]]
                ap = (r2-r1)*p
                ap_all = ap_all + ap
        return ap_all

def evolution(obj,gt,classNumber):

        print(obj)
        splitPredict=[[],[],[]]
        for i in obj:
                if i["Predict"] == "pothole":
                        splitPredict[0].append(i)
                elif i["Predict"] == "repai":
                        splitPredict[1].append(i)
                elif i["Predict"] == "crack":
                        splitPredict[2].append(i)
                
        AP = {}
        _, axs = plt.subplots(1, 3)
        for idx,data in enumerate(splitPredict):
                Percision,Recall = PercisionAndRecall(data,classNumber[data[0]["Predict"]])
                interpolation = interp(Percision)
                AP[data[0]["Predict"]] = averagePrecision(interpolation,Recall)
                
                axs[idx].plot(Recall,Percision, "-o")
                axs[idx].plot(Recall,interpolation ,".", ls='--')
                axs[idx].set_title(AP[data[0]["Predict"]])
                axs[idx].set(xlabel='Recall', ylabel='Percision')
        plt.show()

        print("mAP = ",sum(AP.values())/3)


def get_map(pred, gt, f,imgName):
        T = {}
        P = {}
        
        fx, fy = f
        
        for bbox in gt:
                bbox['bbox_matched'] = False
        
        pred_probs = np.array([s['prob'] for s in pred])
        box_idx_sorted_by_prob = np.argsort(pred_probs)[::-1]
        ACC = []
        BBOXPER = len(box_idx_sorted_by_prob)
        
        
        for idx,box_idx in enumerate(box_idx_sorted_by_prob):
                temp = {}
                temp["Images"] = imgName
                pred_box = pred[box_idx]
                pred_class = pred_box['class']
                pred_x1 = pred_box['x1']
                pred_x2 = pred_box['x2']
                pred_y1 = pred_box['y1']
                pred_y2 = pred_box['y2']
                pred_prob = pred_box['prob']
                if pred_class not in P:
                        P[pred_class] = []
                        T[pred_class] = []
                P[pred_class].append(pred_prob)
                temp["Predict"] = pred_class
                temp["confidences"] = (pred_prob)
                found_match = 0
                for id,gt_box in enumerate(gt):
                        gt_class = gt_box['class']
                        gt_x1 = gt_box['x1']/fx
                        gt_x2 = gt_box['x2']/fx
                        gt_y1 = gt_box['y1']/fy
                        gt_y2 = gt_box['y2']/fy
                        gt_seen = gt_box['bbox_matched']

                        if (gt_class) == (pred_class):
                                temp["Result"] = "TP"
                        else:
                                temp["Result"] = "FP"
                        
                        if gt_seen:
                                continue
                        iou = ious((pred_x1, pred_y1, pred_x2, pred_y2), (gt_x1, gt_y1, gt_x2, gt_y2))
                        if iou >= 0.01:
                                found_match = iou
                                gt_box['bbox_matched'] = True
                                break
                        else:
                                continue
                temp["IOU"] = round(found_match, 5)
                ACC.append(temp)
                T[pred_class].append((found_match))
  
        return ACC, len(gt)

config = tf.compat.v1.ConfigProto(allow_soft_placement=True,inter_op_parallelism_threads=1,intra_op_parallelism_threads=1)
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = K.set_session(tf.compat.v1.Session(config=config))
K.set_session(sess)

filename = 'Road-Damage-Detection-Program/source/model/model_vgg_config.pickle'
with open(filename, 'rb') as f_in:
        C = pickle.load(f_in)

C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False


def getRatio(img, C):
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
        return ratio	


def format_img(img, C):
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape
	
	if width <= height:
		f = img_min_side/width
		new_height = int(f * height)
		new_width = int(img_min_side)
	else:
		f = img_min_side/height
		new_width = int(f * width)
		new_height = int(img_min_side)
	fx = width/float(new_width)
	fy = height/float(new_height)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img /= C.img_scaling_factor
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img, fx, fy





num_rois = 4
test_path = "Road-Damage-Detection-Program/source/DATASET/test/annotate.txt"
class_mapping = C.class_mapping

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
shared_layers = nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = rpn_layer(shared_layers, num_anchors)

classifier = classifier_layer(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)

model_classifier = Model([feature_map_input, roi_input], classifier)
temp = ('Road-Damage-Detection-Program/'+C.model_path)
print(f'Loading weights from {temp}')
model_rpn.load_weights(temp, by_name=True)


model_classifier.load_weights(temp, by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')
model_classifier.summary()
test_imgs, classNumber, _ = get_data(test_path)

T = {}
P = {}

ArrayTable = []
NumberGT = []

for idx, img_data in enumerate(test_imgs):
        print('{}/{}'.format(idx,len(test_imgs)))
        st = time.time()
        filepath = img_data['filepath']
        imgName = filepath.split("/")
        imgName = imgName[len(imgName)-1]
        print()
        print(filepath)
        
        img = cv2.imread(filepath)
        
        X, fx, fy = format_img(img, C)
        ratio = getRatio(img, C)

        imgSave = img.copy()

        X = np.transpose(X, (0, 2, 3, 1))

        # get the feature maps and output from the RPN
        [Y1, Y2, F] = model_rpn.predict(X)

        R = rpn_to_roi(Y1, Y2, C, K.image_data_format(), overlap_thresh=0.7)

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}

        for jk in range(R.shape[0] // C.num_rois + 1):
                ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
                if ROIs.shape[1] == 0:
                        break

                if jk == R.shape[0] // C.num_rois:
                        # pad R
                        curr_shape = ROIs.shape
                        target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
                        ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                        ROIs_padded[:, :curr_shape[1], :] = ROIs
                        ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                        ROIs = ROIs_padded

                [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

                for ii in range(P_cls.shape[1]):

                        if np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                                continue

                        cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

                        if cls_name not in bboxes:
                                bboxes[cls_name] = []
                                probs[cls_name] = []

                        (x, y, w, h) = ROIs[0, ii, :]

                        cls_num = np.argmax(P_cls[0, ii, :])
                        try:
                                (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                                tx /= C.classifier_regr_std[0]
                                ty /= C.classifier_regr_std[1]
                                tw /= C.classifier_regr_std[2]
                                th /= C.classifier_regr_std[3]
                                x, y, w, h = apply_regr(x, y, w, h, tx, ty, tw, th)
                        except:
                                pass
                        bboxes[cls_name].append([16 * x, 16 * y, 16 * (x + w), 16 * (y + h)])
                        probs[cls_name].append(np.max(P_cls[0, ii, :]))

        all_dets = []

        for key in bboxes:
                bbox = np.array(bboxes[key])

                new_boxes, new_probs = non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
                for jk in range(new_boxes.shape[0]):
                        (x1, y1, x2, y2) = new_boxes[jk, :]
                        det = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': key, 'prob': new_probs[jk]}
                        all_dets.append(det)
              
                        (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
                        cv2.rectangle(imgSave,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)
                        
                        textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
                        
                        (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
                        textOrg = (real_x1, real_y1-0)
                        
                        cv2.rectangle(imgSave, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
                        cv2.rectangle(imgSave, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
                        cv2.putText(imgSave, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)


        print('Elapsed time = {}'.format(time.time() - st))
        ACC, gt = get_map(all_dets, img_data['bboxes'], (fx, fy),imgName)
        ArrayTable.append(ACC)
        NumberGT.append(gt)
        # for key in t.keys():
        #         if key not in T:
        #                 T[key] = []
        #                 P[key] = []
        #         T[key].extend(t[key])
        #         P[key].extend(p[key])
        # all_aps = []
        # for key in T.keys():  
        #         ap = average_precision_score(T[key], P[key])
        #         print('{} AP: {}'.format(key, ap))
        #         all_aps.append(ap)
        print("\n\n\n\n",idx)
        # print('mAP = {}'.format(np.mean(np.array(all_aps))))
        cv2.imwrite('test{}.png'.format(idx),imgSave)

ArrayTemp = []
for i in ArrayTable:
        for data in i:
                if data["IOU"] > 0.01:
                        ArrayTemp.append(data)
                        
ArrayTemp = sorted(ArrayTemp, key=lambda k: k['confidences'] ,reverse=True) 
evolution(ArrayTemp,sum(NumberGT),classNumber)