# coding=utf-8
from keras.layers import *
from keras.models import Model
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input as mobilenet_process
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as resnet50_process
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input as xception_process
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input as inresv2_process


def mobilenet(input_size=(224, 224, 3), num_classes=2):
    # If imagenet weights are being loaded, input must have a static square shape
    # (one of (128, 128), (160, 160), (192, 192), or (224, 224))
    base_model = MobileNet(input_shape=input_size, dropout=0.5, include_top=False,
                           weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(base_model.input, predictions)
    process = mobilenet_process
    return model, process


def resnet50(input_size=(224, 224, 3), num_classes=2):
    base_model = ResNet50(input_shape=input_size, include_top=False, weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(base_model.input, predictions)
    process = resnet50_process
    return model, process


def xception(input_size=(299, 299, 3), num_classes=2):
    base_model = Xception(input_shape=input_size, include_top=False, weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(base_model.input, predictions)
    process = xception_process
    return model, process


def inresv2(input_size=(299, 299, 3), num_classes=2):
    base_model = InceptionResNetV2(input_shape=input_size, include_top=False, weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # x = Dropout(rate=0.25)(x)  # rate随机断开的概率
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(base_model.input, predictions)
    process = inresv2_process
    return model, process
