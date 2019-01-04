import os

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

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    # import keras_retinanet.bin
    #
    # __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
import models

font = cv2.FONT_HERSHEY_SIMPLEX


def drop_large_overlaped_boxes(detect_result, drop_iou=0.1):
    drop_index = []  # record which boxes to drop

    scores = detect_result[:, 4]
    labels = detect_result[:, 5]
    boxes = detect_result[:, :4]

    num_boxes = len(boxes)
    for i in range(num_boxes - 1):
        for j in range(i + 1, num_boxes):

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
    hat_thresh = 0.7
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
                if score < 0.75:
                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 0), 1)
                else:
                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 1)
                # elif 400 < area < 750:
                #     cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 1)
                # else:
                #     cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 0), 1)

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


def detect_and_draw(model, image):
    image, scale = resize_image(image, 800, 1430)
    print("scale:", scale)
    start = time.time()
    boxes1, scores1, labels1 = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("Process Time: ", time.time() - start)
    # img_idx += 1
    # boxes /= scale

    hat_idx = np.where(labels1[0] == labels_to_names['hathead'])[0]
    nohat_idx = np.where(labels1[0] == labels_to_names['nohathead'])[0]
    det_hat = np.hstack((boxes1[0][hat_idx],
                         scores1[0][hat_idx][:, np.newaxis],
                         labels1[0][hat_idx][:, np.newaxis])).astype(np.float32)
    det_nohat = np.hstack((boxes1[0][nohat_idx],
                           scores1[0][nohat_idx][:, np.newaxis],
                           labels1[0][nohat_idx][:, np.newaxis])).astype(np.float32)
    dets = np.vstack((det_hat, det_nohat))
    frame_no_track = vis_detections(image, dets)
    return frame_no_track


save_path = '/media/yul-j/ssd_disk/Datasets/Data/safetyhat/data/oss/difficult'

# snapshots_fld = "/home/yul-j/Desktop/Smoke/Detection/logs/0824_v3/snapshots/"
model_path1 = '/home/yul-j/Desktop/Project/Safetyhat/Retinanet/logs/0928_800_lr_mult/snapshots/resnet50_pascal_50_inference.h5'
model_path2 = '/home/yul-j/Desktop/Project/Safetyhat/Retinanet/logs/1223_dataAug/snapshots/resnet50_pascal_50_inference.h5'

backbone = 'resnet50'

# im_source1 = '/media/yul-j/ssd_disk/Datasets/Data/safetyhat/data/VOC2007_Simplified/test/*'
# im_source1 = '/media/yul-j/ssd_disk/Datasets/Data/safetyhat/data/VOC2007/test/*'
# im_source1 = '/media/yul-j/ssd_disk/Datasets/Data/safetyhat/data/oss/for_dist_analyze/b73722d0-b7b0-4532-8d74-b8ddb8964fa4/*'
im_source1 = '/media/yul-j/ssd_disk/Datasets/Data/safetyhat/data/oss/difficult/*'

# im_source2 = '/home/yul-j/Desktcddcop/Safetyhat/data/test/test.txt'
# source2_path = '/home/yul-j/Desktop/Safetyhat/data/VOC2007/JPEGImages'
# im_source1 = '/home/yul-j/Desktop/*'

im_list = []
for file in glob.glob(im_source1):
    if file.split('.')[-1] == 'jpg' and '_r' not in file:
        im_list.append(file)
    elif os.path.isdir(file):
        file = file + '/*'
        file_2 = glob.glob(file)
        for file_3 in file_2:
            if file_3.split('.')[-1] == 'jpg' and '_r' not in file_3:
                im_list.append(file_3)

if 'im_source2' in locals().keys():
    f = open(im_source2)
    line = f.readline()
    while line:
        im_name = line.split('\n')[0] + '.jpg'
        im_path = os.path.join(source2_path, im_name)
        im_list.append(im_path)
        line = f.readline()

random.shuffle(im_list)
# im_list.sort()
# im_list = im_list[::-1]


# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

# load retinanet model
model1 = models.load_model(model_path1, backbone=backbone, convert=False)
model2 = models.load_model(model_path2, backbone=backbone, convert=False)
# print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names = {'nohathead': 1, 'hathead': 0}
labels_to_colors = {1: (0, 0, 255), 0: (0, 255, 0)}

img_idx = 0
length = len(im_list)

while img_idx < length:
    im_path = im_list[img_idx]
    imName = im_path.split('/')[-1]
    imNum, imFormat = imName.split('.')
    if not imFormat in ['jpg', 'jpeg', 'JPG']:
        del (im_list[img_idx])
        length -= 1
        continue
    print(im_path)
    image = cv2.imread(im_path)

    if image is None:
        del (im_list[img_idx])
        length -= 1
        continue

    img_save = image.copy()
    im_name = im_path.split('/')[-1]
    im_save_path = os.path.join(save_path, im_name)
    if '_r' not in im_name:
        frame_no_track = detect_and_draw(model1, image)
    else:
        frame_no_track = image
    cv2.imshow(im_name + 'model1', frame_no_track)
    op = cv2.waitKey(0)
    cv2.destroyAllWindows()

    if op == ord('s'):
        cv2.imwrite(im_save_path, img_save)
        print('image saved as:', save_path)
    elif op == ord('c'):
        frame_no_track = detect_and_draw(model2, image)
        cv2.namedWindow(im_name)  # Create a named window
        cv2.moveWindow(im_name, 500, 30)  # Move it to (40,30)
        cv2.imshow(im_name + 'model2', frame_no_track)
        op2 = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if op2 == ord('c'):
            continue
        elif op2 == ord('d'):
            img_idx += 1
            # old_result = im_path.split('.')[0] + '_r.jpg'
            # old_result_im = cv2.imread(old_result)
            # cv2.imshow(im_name, old_result_im)
            # cv2.waitKey(0)
        elif op2 == ord('a'):
            img_idx -= 1
            img_idx = max(0, img_idx)
        elif op == ord('q'):
            break
    elif op == ord('a'):
        img_idx -= 1
        img_idx = max(0, img_idx)
    elif op == ord('d'):
        img_idx += 1
    elif op == ord('r'):
        continue
    elif op == ord('q'):
        break
print('done')
