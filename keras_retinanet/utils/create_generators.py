from preprocessing.csv_generator import CSVGenerator
from preprocessing.kitti import KittiGenerator
from preprocessing.open_images import OpenImagesGenerator
from preprocessing.pascal_voc import PascalVocGenerator
from utils.transform import random_transform_generator


def create_generators(configs):
    """create a input data generator"""
    # create random transform generator for augmenting training data
    if not configs["Data_Augmentation"]['only_x_flip']:
        transform_generator = random_transform_generator(
            min_rotation=configs['Data_Augmentation']['rotation'][0],
            max_rotation=configs['Data_Augmentation']['rotation'][1],
            min_translation=configs['Data_Augmentation']['min_translation'],
            max_translation=configs['Data_Augmentation']['max_translation'],
            min_shear=configs['Data_Augmentation']['shear'][0],
            max_shear=configs['Data_Augmentation']['shear'][1],
            min_scaling=configs['Data_Augmentation']['min_scaling'],
            max_scaling=configs['Data_Augmentation']['max_scaling'],
            flip_x_chance=0.5,
            gray=configs['Data_Augmentation']['gray'],
            inverse_color=configs['Data_Augmentation']['inverse_color'],
        )
    else:
        transform_generator = random_transform_generator(flip_x_chance=0.5)

    if configs['Dataset']['dataset_type'] == 'coco':
        # import here to prevent unnecessary dependency on cocoapi
        from preprocessing.coco import CocoGenerator

        train_generator = CocoGenerator(
            configs['Dataset']['dataset_path'],
            'train2017',
            transform_generator=transform_generator,
            batch_size=configs['Train']['batch_size'],
            image_min_side=configs['Train']['image_min_side'],
            image_max_side=configs['Train']['image_max_side']
        )

        validation_generator = CocoGenerator(
            configs['Dataset']['dataset_path'],
            'val2017',
            batch_size=configs['Train']['batch_size'],
            image_min_side=configs['Train']['image_min_side'],
            image_max_side=configs['Train']['image_max_side']
        )
    elif configs['Dataset']['dataset_type'] == 'pascal':
        train_generator = PascalVocGenerator(
            configs['Dataset']['dataset_path'],
            'trainval',
            classes=configs['Dataset']['classes'],
            transform_generator=transform_generator,
            batch_size=configs['Train']['batch_size'],
            image_min_side=configs['Train']['image_min_side'],
            image_max_side=configs['Train']['image_max_side'],
            anchor_ratios=configs['Anchors']['ratios'],
            anchor_scales=configs['Anchors']['scales'],
            anchor_sizes=configs['Anchors']['sizes'],
            anchor_strides=configs['Anchors']['strides']
        )

        validation_generator = PascalVocGenerator(
            configs['Dataset']['dataset_path'],
            'test',
            classes=configs['Dataset']['classes'],
            batch_size=configs['Train']['batch_size'],
            image_min_side=configs['Train']['image_min_side'],
            image_max_side=configs['Train']['image_max_side'],
            anchor_ratios=configs['Anchors']['ratios'],
            anchor_scales=configs['Anchors']['scales'],
            anchor_sizes=configs['Anchors']['sizes'],
            anchor_strides=configs['Anchors']['strides']
        )
    elif configs['Dataset']['dataset_type'] == 'csv':
        train_generator = CSVGenerator(
            configs['Dataset']['csv_data_file'],
            configs['Dataset']['csv_classes_file'],
            transform_generator=transform_generator,
            batch_size=configs['Train']['batch_size'],
            image_min_side=configs['Train']['image_min_side'],
            image_max_side=configs['Train']['image_max_side']
        )

        if configs['Dataset']['csv_val_annotations']:
            validation_generator = CSVGenerator(
                configs['Dataset']['csv_val_annotations'],
                configs['Dataset']['csv_classes_file'],
                batch_size=configs['Train']['batch_size'],
                image_min_side=configs['Train']['image_min_side'],
                image_max_side=configs['Train']['image_max_side']
            )
        else:
            validation_generator = None
    elif configs['Dataset']['dataset_type'] == 'oid':
        train_generator = OpenImagesGenerator(
            configs['Dataset']['dataset_path'],
            subset='train',
            version=configs['Dataset']['version'],
            labels_filter=configs['Dataset']['oid_labels_filter'],
            annotation_cache_dir=configs['Dataset']['oid_annotation_cache_dir'],
            fixed_labels=configs['Dataset']['fixed_labels'],
            transform_generator=transform_generator,
            batch_size=configs['Train']['batch_size'],
            image_min_side=configs['Train']['image_min_side'],
            image_max_side=configs['Train']['image_max_side']
        )

        validation_generator = OpenImagesGenerator(
            configs['Dataset']['dataset_path'],
            subset='validation',
            version=configs['Dataset']['version'],
            labels_filter=configs['Dataset']['oid_labels_filter'],
            annotation_cache_dir=configs['Dataset']['oid_annotation_cache_dir'],
            fixed_labels=configs['Dataset']['fixed_labels'],
            batch_size=configs['Train']['batch_size'],
            image_min_side=configs['Train']['image_min_side'],
            image_max_side=configs['Train']['image_max_side']
        )
    elif configs['Dataset']['dataset_type'] == 'kitti':
        train_generator = KittiGenerator(
            configs['Dataset']['dataset_path'],
            subset='train',
            transform_generator=transform_generator,
            batch_size=configs['Train']['batch_size'],
            image_min_side=configs['Train']['image_min_side'],
            image_max_side=configs['Train']['image_max_side']
        )

        validation_generator = KittiGenerator(
            configs['Dataset']['dataset_path'],
            subset='val',
            batch_size=configs['Train']['batch_size'],
            image_min_side=configs['Train']['image_min_side'],
            image_max_side=configs['Train']['image_max_side']
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(configs['Dataset']['dataset_type']))

    return train_generator, validation_generator
