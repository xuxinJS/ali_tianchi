from keras.layers import *
from keras.optimizers import *
from keras.models import Model
from keras.applications.mobilenet import MobileNet
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.applications.nasnet import NASNetMobile

def mobilenet(pretrained_weights=None, input_size=(224, 224, 3), num_classes=2):
    # If imagenet weights are being loaded, input must have a static square shape
    # (one of (128, 128), (160, 160), (192, 192), or (224, 224))
    base_model = MobileNet(input_shape=input_size, dropout=0.5, include_top=False,
                           weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(base_model.input, predictions)
    if pretrained_weights:
        base_model.load_weights(pretrained_weights, by_name=True)
    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    # model.summary()

    return model


def resnet50(pretrained_weights=None, input_size=(224, 224, 3), num_classes=2):
    base_model = ResNet50(input_shape=input_size, include_top=False,
                          weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(base_model.input, predictions)
    if pretrained_weights:
        base_model.load_weights(pretrained_weights, by_name=True)
    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    # model.summary()

    return model


def xception(pretrained_weights=None, input_size=(299, 299, 3), num_classes=2):
    # (one of (150, 150), (299, 299))
    base_model = Xception(input_shape=input_size, include_top=False,
                          weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(base_model.input, predictions)
    if pretrained_weights:
        base_model.load_weights(pretrained_weights, by_name=True)
    #     for layer in base_model.layers[:-2]:
    #         layer.trainable = False
    model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['acc'])
    # model.summary()

    return model


