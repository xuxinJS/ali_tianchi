import os
import sys

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from argparse import ArgumentParser
from keras.callbacks import TensorBoard
from keras import backend as K
import tensorflow as tf
from multiprocessing import cpu_count

from model_lib import *

# control CUDA/tensorflow log level
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument('-m', required=True, type=str, choices=['mobilenet', 'resnet50', 'xception'])
    parser.add_argument('-pw', help='pretrained_weights path', type=str, default=None)
    parser.add_argument('-b', help='batch size', type=int, default=16)
    parser.add_argument('-e', help='epoch', type=int, default=30)
    parser.add_argument('-log', help='directory to save log', type=str, default='./log_dir')
    parser.add_argument('-dst', help='Path to the models to save.', required=True, type=str)
    parser.add_argument('--train', '-t', help='Path to the folder of the training data', required=True,
                        type=str)
    parser.add_argument('--validataion', '-v', help='Path to the folder of the validation data', required=True,
                        type=str)
    parser.add_argument('-gpu', default='0', type=str)

    return parser

def steps_per_epoch_num(folder, batch_size):
    data_number = 0
    for folder_name in os.listdir(folder):
        data_number += len(os.listdir(os.path.join(folder, folder_name)))
    steps = data_number // batch_size
    return steps

def main():
    args = build_argparser().parse_args()
    train_path = os.path.abspath(args.train)
    validation_path = os.path.abspath(args.validataion)
    dst_path = os.path.abspath(args.dst)
    log_dir = args.log
    model_name = args.m
    pre_weights = args.pw
    batch_size = args.b
    epochs = args.e
    gpu = args.gpu

    # prepare data
    num_classes = len(os.listdir(train_path))

    # keras config
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    keras_config = tf.ConfigProto()
    keras_config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
    with tf.Session(config=keras_config) as sess:
        K.set_session(sess)
        if model_name == 'mobilenet':
            image_size = 224
            model = mobilenet(pretrained_weights=pre_weights, input_size=(image_size, image_size, 3),
                              num_classes=num_classes)
        elif model_name == 'resnet50':
            image_size = 224
            model = resnet50(pretrained_weights=pre_weights, input_size=(image_size, image_size, 3),
                             num_classes=num_classes)
        elif model_name == 'xception':
            image_size = 299
            model = xception(pretrained_weights=pre_weights, input_size=(image_size, image_size, 3),
                             num_classes=num_classes)

        # model.summary()
        train_steps = steps_per_epoch_num(train_path, batch_size)
        validation_step = steps_per_epoch_num(validation_path, batch_size)

        # generate training data
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           brightness_range=(0, 1.0),
                                           horizontal_flip=True,
                                           vertical_flip=True,
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           zoom_range=0.2)

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(train_path,
                                                            target_size=(image_size, image_size),
                                                            batch_size=batch_size)
                                                            # save_to_dir='/home/xuxin/model/0627/gen')
        validation_generator = test_datagen.flow_from_directory(validation_path,
                                                                target_size=(image_size, image_size),
                                                                batch_size=batch_size)

        os.makedirs(dst_path, exist_ok=True)
        h5_path = os.path.join(dst_path, model_name + '.h5')

        # training
        callbacks_list = [EarlyStopping(monitor='val_acc', patience=20, verbose=0),
                          TensorBoard(log_dir=log_dir),
                          ModelCheckpoint(h5_path, monitor='val_acc', verbose=0,
                                          save_best_only=True, save_weights_only=False)]

        cpu_workers = cpu_count() - 1 if (cpu_count() - 1) >= 1 else cpu_count()

        model.fit_generator(train_generator,
                            steps_per_epoch=train_steps,
                            epochs=epochs,
                            validation_data=validation_generator,
                            validation_steps=validation_step,
                            verbose=0,
                            callbacks=callbacks_list,
                            max_queue_size=50,
                            workers=cpu_workers,
                            use_multiprocessing=True)  # may cause exception

    K.clear_session()


if __name__ == '__main__':
    sys.exit(main() or 0)
