import os
import sys
import cv2
import math

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger, LearningRateScheduler
from argparse import ArgumentParser
from keras.utils import to_categorical, Sequence
from albumentations import *
from model_lib import *
from keras.optimizers import Adam
from glob import glob

from keras import backend as K
import tensorflow as tf
import numpy as np
import datetime as dt

# control CUDA/tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument('-m', required=True, type=str, choices=['mobilenet', 'resnet50', 'xception'])
    parser.add_argument('-pw', help='pretrained_weights path', type=str, default=None)
    parser.add_argument('-b', help='batch size', type=int, default=16)
    parser.add_argument('-e', help='epoch', type=int, default=30)
    parser.add_argument('-dst', help='Path to the models to save.', type=str, default='.')
    parser.add_argument('--train', '-t', help='folder of the training data', required=True, type=str)
    parser.add_argument('--validataion', '-v', help='folder of the validation data', type=str, default=None)
    parser.add_argument('--learning_rate', '-lr', help='init learning rate', type=float, default=1e-3)
    parser.add_argument('--epoch_drop', '-ed', help='epochs learning rate drop', type=int, default=10)
    parser.add_argument('--aug_prob', '-aug', type=float, default=0.5)
    parser.add_argument('--process_num', '-pn', type=int, default=1)
    parser.add_argument('-gpu', default='0', type=str)
    return parser


def strong_aug(p=.5):
    return Compose([
        Flip(),
        OneOf([
            CLAHE(clip_limit=4, tile_grid_size=(8, 8), p=1),
            RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1),
            HueSaturationValue(hue_shift_limit=15, sat_shift_limit=15, val_shift_limit=15, p=1),
        ], p=1),
    ], p=p)


def lr_decay(epoch):
    """ Learning rate decay function

        Drop learning rate by [cfg.lr_sched_drop] every
        [cfg.epochs_drop] number of epochs

        Input
                epoch : Current epoch count
        Output
                lrate : New learning rate
    """
    initial_lrate = lr
    drop_rate = 0.6
    epochs_until_drop = lr_epochs_drop
    lrate = initial_lrate * math.pow(drop_rate, math.floor((1 + epoch) / epochs_until_drop))
    return lrate


class DataGenerator(Sequence):
    """ Generates data for training and validation
    """

    def __init__(self, img_names, labels, batch_size, n_classes, dim, preprocess_input,
                 shuffle=True, aug=True, save_folder=None):
        """ Initialization
        img_names:full path of image
        labels:0 ~ n_classes-1
        dim:(width, heigth)
        """
        self.img_names = img_names
        self.labels = labels
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.dim = dim
        self.preprocess_input = preprocess_input
        self.shuffle = shuffle
        self.aug = aug
        self.save_folder = save_folder
        self.image_nums = len(self.img_names)
        self.all_index = np.arange(self.image_nums)

        self.on_epoch_end()

    def __len__(self):
        """ Returns number of batches in dataset
        """
        return int(np.floor(self.image_nums / self.batch_size))

    def __getitem__(self, index):
        """ Returns one batch of data
        """
        # Generate start and end indices of the batch
        idxmin = index * self.batch_size
        idxmax = min((index + 1) * self.batch_size, self.image_nums)
        temp_index = self.all_index[idxmin:idxmax]
        # Find IDs and labels for the selected indices
        img_names = [self.img_names[i] for i in temp_index]
        list_labels = [self.labels[i] for i in temp_index]
        # Generate data for selected IDs and labels
        x, y = self.__data_generation(img_names, list_labels)
        return x, y

    def on_epoch_end(self):
        """ Updates all_index after each epoch
        """
        # Shuffle array of indices after each epoch end
        if self.shuffle:
            np.random.seed(42)
            np.random.shuffle(self.all_index)

    def __data_generation(self, img_names, list_labels):
        """ Generates data containing batch_size samples
        """
        x_batch = []
        for image_name in img_names:
            image = cv2.imread(image_name)
            image = cv2.resize(image, self.dim)
            if self.aug == True:
                data = {"image": image}
                augmented = global_aug(**data)
                image = augmented["image"]
            if self.save_folder:
                cv2.imwrite(os.path.join(self.save_folder, os.path.basename(image_name)), image)
            x_batch.append(image)
        x_array = self.preprocess_input(np.array(x_batch))
        y_array = to_categorical(np.array(list_labels), num_classes=self.n_classes)
        return x_array, y_array


def main():
    global lr
    global lr_epochs_drop
    global global_aug

    args = build_argparser().parse_args()
    train_path = os.path.abspath(args.train)
    start_time = dt.datetime.now().strftime('%Y%m%d_%H%M')
    dst_path = os.path.join(os.path.abspath(args.dst), start_time)
    model_name = args.m
    pre_weights = args.pw
    batch_size = args.b
    epochs = args.e
    gpu = args.gpu
    lr = args.learning_rate
    lr_epochs_drop = args.epoch_drop
    global_aug = strong_aug(p=args.aug_prob)
    process_num = args.process_num

    log_dir = os.path.join(dst_path, 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # prepare data
    classes = sorted(os.listdir(train_path))
    num_classes = len(classes)
    class_ids = dict()  # {'label1':0, ...}
    train_image_names = []
    train_image_indexs = []
    val_image_names = []
    val_image_indexs = []

    with open(os.path.join(dst_path, 'class.txt'), 'w') as f:
        for index, label in enumerate(classes):
            f.write("%s\n" % label)
            class_ids[label] = index

    for label in classes:
        tmp_names = glob(os.path.join(train_path, label, '*'))
        tmp_indexs = [class_ids[label]] * len(tmp_names)
        train_image_names.extend(tmp_names)
        train_image_indexs.extend(tmp_indexs)

    valid_flag = False
    if validation_path is not None:
        validation_path = os.path.abspath(args.validataion)
        valid_flag = True
        for label in classes:
            tmp_names = glob(os.path.join(validation_path, label, '*'))
            tmp_indexs = [class_ids[label]] * len(tmp_names)
            val_image_names.extend(tmp_names)
            val_image_indexs.extend(tmp_indexs)

    save_name = model_name + '_ep{epoch:02d}_' + 'vloss{val_loss:.4f}.h5' if valid_flag else 'loss{loss:.4f}.h5'
    full_save_name = os.path.join(dst_path, save_name)

    # keras config
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    keras_config = tf.ConfigProto()
    keras_config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
    with tf.Session(config=keras_config) as sess:
        K.set_session(sess)
        if model_name == 'mobilenet':
            image_height = 224
            image_width = 224
            model, process_input = mobilenet(input_size=(image_height, image_width, 3),
                                             num_classes=num_classes)
        elif model_name == 'resnet50':
            image_height = 224
            image_width = 224
            model, process_input = resnet50(input_size=(image_height, image_width, 3),
                                            num_classes=num_classes)
        elif model_name == 'xception':
            image_height = 299
            image_width = 299
            model, process_input = xception(input_size=(image_height, image_width, 3),
                                            num_classes=num_classes)

        if pre_weights:
            model.load_weights(pre_weights)
        model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])

        # training
        callbacks_list = [
            LearningRateScheduler(lr_decay),
            EarlyStopping(monitor='val_loss', patience=20, verbose=0),
            TensorBoard(log_dir=log_dir),
            ModelCheckpoint(full_save_name, monitor='val_loss' if valid_flag else 'loss',
                            verbose=0, save_best_only=True, save_weights_only=True),
            CSVLogger(os.path.join(dst_path, "%s_%s.csv" % (model_name, start_time)))
        ]

        # Initialize train data generator
        training_generator = DataGenerator(img_names=train_image_names,
                                           labels=train_image_indexs,
                                           batch_size=batch_size,
                                           n_classes=num_classes,
                                           dim=(image_width, image_height),
                                           preprocess_input=process_input,
                                           shuffle=True,
                                           aug=True,
                                           save_folder=None)

        # Initialize validation generator
        if valid_flag:
            validation_generator = DataGenerator(img_names=val_image_names,
                                                 labels=val_image_indexs,
                                                 batch_size=batch_size,
                                                 n_classes=num_classes,
                                                 dim=(image_width, image_height),
                                                 preprocess_input=process_input,
                                                 shuffle=False,
                                                 aug=False)

        model.fit_generator(generator=training_generator,
                            epochs=epochs,
                            callbacks=callbacks_list,
                            validation_data=validation_generator if valid_flag else None,
                            max_queue_size=20,
                            workers=process_num,
                            use_multiprocessing=True)
        # class_weight)

    K.clear_session()

if __name__ == '__main__':
    sys.exit(main() or 0)
