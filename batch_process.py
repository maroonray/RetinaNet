import os
import shutil
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

import glob
import time
import random
import sys

import cv2
import keras
import numpy as np
import tensorflow as tf

from keras_retinanet.models.resnet import custom_objects
from keras_retinanet.utils.image import preprocess_image, resize_image
from keras_retinanet.utils.tracker_old import iou
from utils.write_xml import make_xml


def drop_large_overlaped_boxes(detect_result, drop_iou=0.5):
    drop_index = []  # record which boxes to drop

    scores = detect_result[:, 4]
    labels = detect_result[:, 5]
    boxes = detect_result[:, :4]

    num_boxes = len(boxes)
    for i in range(num_boxes - 1):
        for j in range(i + 1, num_boxes):
            if labels[i] == labels[j]:
                continue
            else:
                iou_ij = iou(boxes[i], boxes[j])
                if iou_ij >= drop_iou:

                    idx = np.argmin((scores[i], scores[j]))
                    if idx:
                        drop_index.append(j)
                    else:
                        drop_index.append(i)

    if not len(drop_index):
        return detect_result
    else:
        detect_result = np.delete(detect_result, drop_index, axis=0)

    return detect_result


def vis_detections(im, results):
    """Draw detected bounding boxes."""
    hat_thresh = 0.5
    nohat_thresh = 0.5
    results = drop_large_overlaped_boxes(results)
    num_head = len(results)
    if num_head == 0:
        return im
    else:
        img = im.copy()
        for i in range(num_head):
            bbox = results[i, :4]
            score = results[i, 4]
            label = results[i, 5]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

            if label == labels_to_names['hathead']:
                if score < hat_thresh:
                    continue
                print("class: hat  ", "score: ", score, "left-top: ", int(bbox[0]), ", ", int(bbox[1]), 'area: ', area)
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
            else:
                if score < nohat_thresh:
                    continue
                print("class: nohat", "score: ", score, "left-top: ", int(bbox[0]), ", ", int(bbox[1]), 'area: ', area)
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 1)
                img = cv2.putText(img, str(round(area, 2)), (bbox[0], int(bbox[3] + 30)), font, 0.5, (0, 0, 0), 2)

            img = cv2.putText(img, str(score)[:6], (bbox[0], bbox[1]), font, 0.5, (0, 0, 0), 2)
        return img


def vis_detections_track(im, tracker, frame):
    """Draw detected bounding boxes."""
    if type(tracker[0]) == int:
        tracker = tracker[1:]

    if len(tracker) == 0:
        return im
    else:
        img = im.copy()
        img = cv2.putText(img, str(frame), (10, 30), font, 2, (255, 255, 255), 2)
        for obj in tracker:
            bbox = obj["position"]
            score = np.round(obj["track_score"], 4)
            if not obj["show"]:
                continue
            else:

                if obj["class"] == -1:  # 无安全帽
                    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
                    img = cv2.putText(img, str(obj["contains"]),
                                      (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)),
                                      font, 0.5, (255, 255, 255), 2)
                    img = cv2.putText(img, str(score)[:6], (int(bbox[0]), int(bbox[1])), font, 0.5, (255, 255, 255), 2)
                else:  # 有安全帽
                    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                    img = cv2.putText(img, str(obj["contains"]),
                                      (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)),
                                      font, 0.5, (255, 255, 255), 2)
                    img = cv2.putText(img, str(score)[:6], (int(bbox[0]), int(bbox[1])), font, 0.5, (255, 255, 255), 2)

        print("--------------------------------------------------------------------------")
        return img


def iou_for_debug(tracker, idx1, idx2):
    box1 = tracker[idx1 + 1]["pred_position"]
    box2 = tracker[idx2 + 1]["pred_position"]
    return iou(box1, box2)


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def main(model_path, im_path, dst_dir):
    im_list = []
    for file in glob.glob(im_path):
        if file.split('.')[-1] == 'jpg' and '_r' not in file:
            im_list.append(file)
        elif os.path.isdir(file):
            file = file + '/*'
            file_2 = glob.glob(file)
            count = 0
            for file_3 in file_2:
                if file_3.split('.')[-1] == 'jpg' and '_r' not in file_3:
                    count += 1
                    im_list.append(file_3)
                # if count > 300:
                #     break

    # random.shuffle(im_list)
    im_list.sort()
    # im_list = im_list[::-1]

    # set the modified tf session as backend in keras
    keras.backend.tensorflow_backend.set_session(get_session())

    # load retinanet model
    model = models.load_model(model_path, backbone='resnet50', convert=False)

    # load label to names mapping for visualization purposes
    names_to_labels = {'nohathead': 1, 'hathead': 0}
    labels_to_colors = {1: (0, 0, 255), 0: (0, 255, 0)}
    labels_to_names = {0: 'hathead', 1: 'nohathead'}

    label_info = []
    area_info = []
    score_info = []
    name_info = []

    count = 0
    for im_path in im_list:
        im_name = im_path.split('/')[-1]
        count += 1
        image = cv2.imread(im_path)
        if image is None:
            continue
        ori_shape = image.shape

        # image = preprocess_image(image)
        image, scale = resize_image(image, 800, 1430)

        start = time.time()
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        end = time.time()
        print('{}/{}, {}'.format(count, len(im_list), end - start))
        # img_idx += 1
        boxes /= scale
        over_threshold = np.where(scores[0] > 0.5)[0]

        label_info.extend(labels[0][over_threshold])
        if np.any(labels[0][over_threshold] == 1):
            xmin = boxes[0][over_threshold, 0]
            xmax = boxes[0][over_threshold, 2]
            ymin = boxes[0][over_threshold, 1]
            ymax = boxes[0][over_threshold, 3]
            label_list = [labels_to_names[ii] for ii in labels[0][over_threshold]]
            dom = make_xml(xmin, ymin, xmax, ymax, label_list, im_name, ori_shape[1], ori_shape[0])
            xml_name = im_name.split('.')[0] + '.xml'
            xml_path = os.path.join(dst_dir, 'Annotations', xml_name)
            with open(xml_path, 'wb') as f:
                f.write(dom.toprettyxml(indent='\t', encoding='utf-8'))
            im_path_dst = os.path.join(dst_dir, 'JPEGImages', im_name)
            shutil.move(im_path, im_path_dst)

    #     score_info.extend(scores[0][over_threshold])
    #     area = (boxes[0][over_threshold, 2] - boxes[0][over_threshold, 0]) * (
    #                 boxes[0][over_threshold, 3] - boxes[0][over_threshold, 1])
    #     area_info.extend(area)
    #     name_info.extend([im_name] * len(over_threshold))
    #
    # label_info = np.asarray(label_info)
    # score_info = np.asarray(score_info)
    # area_info = np.asarray(area_info)
    # name_info = np.asarray(name_info)
    # print(len(area_info))


if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    import models

    random.seed(2018)

    snapshots_fld = '/home/yul-j/Desktop/Project/Safetyhat/Retinanet/logs/0928_800_lr_mult/snapshots/'
    model_subpath = 'resnet50_pascal_50_inference.h5'
    model_path = os.path.join(snapshots_fld, model_subpath)
    im_source1 = '/media/yul-j/ssd_disk/Datasets/Data/safetyhat/data/oss/for_dist_analyze/e917d20a-e13a-472a-ac63-844fe2add939/*'
    dst_dir = '/media/yul-j/ssd_disk/Datasets/Data/safetyhat/data/oss/negative_img/pool'
    main(model_path, im_source1, dst_dir)
