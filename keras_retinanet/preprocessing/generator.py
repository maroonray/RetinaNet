"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import sys
import numpy as np
import random
import threading
import time
import warnings
import cv2

import keras

sys.path.append("..")

from utils.anchors import anchor_targets_bbox, bbox_transform
from utils.image import (
    TransformParameters,
    adjust_transform_for_image,
    apply_transform,
    preprocess_image,
    resize_image,
)
from utils.transform import transform_aabb


class Generator(object):
    def __init__(
            self,
            transform_generator=None,
            batch_size=1,
            group_method='ratio',  # one of 'none', 'random', 'ratio'
            shuffle_groups=True,
            image_min_side=800,
            image_max_side=1430,
            transform_parameters=None,
            compute_anchor_targets=anchor_targets_bbox,
            anchor_strides=None,
            anchor_sizes=None,
            anchor_ratios=None,
            anchor_scales=None
    ):
        self.transform_generator = transform_generator
        self.batch_size = int(batch_size)
        self.group_method = group_method
        self.shuffle_groups = shuffle_groups
        self.image_min_side = image_min_side
        self.image_max_side = image_max_side
        self.transform_parameters = transform_parameters or TransformParameters()

        if anchor_strides is None:
            self.anchor_strides = [8, 16, 32, 64, 128]
        else:
            self.anchor_strides = anchor_strides
        if anchor_sizes is None:
            self.anchor_sizes = [16, 32, 64, 128, 256]
        else:
            self.anchor_sizes = anchor_sizes
        if anchor_ratios is None:
            self.anchor_ratios = [0.5, 1, 2]
        else:
            self.anchor_ratios = anchor_ratios
        if anchor_scales is None:
            self.anchor_scales = [0, 1.0 / 3, 2.0 / 3]
        else:
            self.anchor_scales = anchor_scales

        self.compute_anchor_targets = compute_anchor_targets

        self.group_index = 0
        self.lock = threading.Lock()

        self.group_images()

    def size(self):
        raise NotImplementedError('size method not implemented')

    def num_classes(self):
        raise NotImplementedError('num_classes method not implemented')

    def name_to_label(self, name):
        raise NotImplementedError('name_to_label method not implemented')

    def label_to_name(self, label):
        raise NotImplementedError('label_to_name method not implemented')

    def image_aspect_ratio(self, image_index):
        raise NotImplementedError('image_aspect_ratio method not implemented')

    def load_image(self, image_index):
        raise NotImplementedError('load_image method not implemented')

    def load_annotations(self, image_index):
        raise NotImplementedError('load_annotations method not implemented')

    def load_annotations_group(self, group):
        return [self.load_annotations(image_index) for image_index in group]

    def filter_annotations(self, image_group, annotations_group, group):
        # test all annotations
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            assert (isinstance(annotations,
                               np.ndarray)), '\'load_annotations\' should return a list of numpy arrays, received: {}'.format(
                type(annotations))

            # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
            invalid_indices = np.where(
                (annotations[:, 2] <= annotations[:, 0]) |
                (annotations[:, 3] <= annotations[:, 1]) |
                (annotations[:, 0] < 0) |
                (annotations[:, 1] < 0) |
                (annotations[:, 2] > image.shape[1]) |
                (annotations[:, 3] > image.shape[0])
            )[0]

            # delete invalid indices
            if len(invalid_indices):
                warnings.warn('Image with id {} (shape {}) contains the following invalid boxes: {}.'.format(
                    group[index],
                    image.shape,
                    [annotations[invalid_index, :] for invalid_index in invalid_indices]
                ))
                annotations_group[index] = np.delete(annotations, invalid_indices, axis=0)

        return image_group, annotations_group

    def load_image_group(self, group):
        return [self.load_image(image_index) for image_index in group]

    def grayscale_inversesolor(self, gray, inverse_color, image):
        if gray and inverse_color:
            r = random.randint(1, 4)  # with probability of 1/4 to convert color or gray scale
            if r == 1:
                m = np.mean(image, axis=2)
                for k in range(3):
                    image[:, :, k] = m
            elif r == 2:
                image = 255 - image
        elif gray and not inverse_color:
            if random.random() > 0.5:
                m = np.mean(image, axis=2)
                for k in range(3):
                    image[:, :, k] = m
        elif not gray and inverse_color:
            if random.random() > 0.5:
                image = 255 - image

        return image

    def random_transform_group_entry(self, image, annotations):
        # randomly transform both image and annotations
        if self.transform_generator:
            gray, inverse_color, transform_mat = next(self.transform_generator)
            transform = adjust_transform_for_image(transform_mat, image,
                                                   self.transform_parameters.relative_translation)
            if gray or inverse_color:
                image = self.grayscale_inversesolor(gray, inverse_color, image)

            image = apply_transform(transform, image, self.transform_parameters)

            # Transform the bounding boxes in the annotations.
            annotations = annotations.copy()
            for index in range(annotations.shape[0]):
                annotations[index, :4] = transform_aabb(transform, annotations[index, :4])

        return image, annotations

    def resize_image(self, image):
        return resize_image(image, min_side=self.image_min_side, max_side=self.image_max_side)

    def preprocess_image(self, image):
        return preprocess_image(image)

    def _show_annotated_image_debug(self, image, annotations):
        draw = image.copy()
        for annotation in annotations:
            annotation = [int(a) for a in annotation]
            if annotation[-1] == 0:
                cv2.rectangle(draw, (annotation[0], annotation[1]), (annotation[2], annotation[3]), (0, 0, 255), 2)
            else:
                cv2.rectangle(draw, (annotation[0], annotation[1]), (annotation[2], annotation[3]), (0, 255, 0), 2)
        cv2.imshow("a", draw)
        cv2.waitKey(0)

    def preprocess_group_entry(self, image, annotations):
        # preprocess the image
        # image = self.preprocess_image(image)

        # self._show_annotated_image_debug(image, annotations)

        # randomly transform image and annotations
        image, annotations = self.random_transform_group_entry(image, annotations)

        # resize image
        image, image_scale = self.resize_image(image)

        # apply resizing to annotations too
        annotations[:, :4] *= image_scale

        # self._show_annotated_image_debug(image, annotations)

        return image, annotations

    def preprocess_group(self, image_group, annotations_group):
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            # preprocess a single group entry
            image, annotations = self.preprocess_group_entry(image, annotations)

            # copy processed data back to group
            image_group[index] = image
            annotations_group[index] = annotations

        return image_group, annotations_group

    def group_images(self):
        # determine the order of the images
        order = list(range(self.size()))
        if self.group_method == 'random':
            random.shuffle(order)
        elif self.group_method == 'ratio':
            order.sort(key=lambda x: self.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in
                       range(0, len(order), self.batch_size)]

    def compute_inputs(self, image_group):
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        # construct an image batch object
        image_batch = np.zeros((self.batch_size,) + max_shape, dtype=keras.backend.floatx())

        # copy all images to the upper left part of the image batch object
        for image_index, image in enumerate(image_group):
            image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

        return image_batch

    def compute_targets(self, image_group, annotations_group):
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        # compute labels and regression targets
        labels_group = [None] * self.batch_size
        regression_group = [None] * self.batch_size
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            # compute regression targets
            labels_group[index], annotations, anchors = self.compute_anchor_targets(
                max_shape,
                annotations,
                self.num_classes(),
                mask_shape=image.shape,
                ratios=self.anchor_ratios,
                scales=self.anchor_scales,
                strides=self.anchor_strides,
                sizes=self.anchor_sizes
            )
            regression_group[index] = bbox_transform(anchors, annotations)

            # append anchor states to regression targets (necessary for filtering 'ignore', 'positive' and 'negative' anchors)
            anchor_states = np.max(labels_group[index], axis=1, keepdims=True)
            regression_group[index] = np.append(regression_group[index], anchor_states, axis=1)

        labels_batch = np.zeros((self.batch_size,) + labels_group[0].shape, dtype=keras.backend.floatx())
        regression_batch = np.zeros((self.batch_size,) + regression_group[0].shape, dtype=keras.backend.floatx())

        # copy all labels and regression values to the batch blob
        for index, (labels, regression) in enumerate(zip(labels_group, regression_group)):
            labels_batch[index, ...] = labels
            regression_batch[index, ...] = regression

        return [regression_batch, labels_batch]

    def compute_input_output(self, group):
        # load images and annotations
        image_group = self.load_image_group(group)
        annotations_group = self.load_annotations_group(group)

        # check validity of annotations
        image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        # perform preprocessing steps
        image_group, annotations_group = self.preprocess_group(image_group, annotations_group)

        # compute network inputs
        inputs = self.compute_inputs(image_group)

        # compute network targets
        targets = self.compute_targets(image_group, annotations_group)

        return inputs, targets

    def __next__(self):
        return self.next()

    def next(self):
        # advance the group index
        with self.lock:
            if self.group_index == 0 and self.shuffle_groups:
                # shuffle groups at start of epoch
                random.shuffle(self.groups)
            group = self.groups[self.group_index]
            self.group_index = (self.group_index + 1) % len(self.groups)

        return self.compute_input_output(group)
