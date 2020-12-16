import sys
import time
from random import shuffle
import numpy as np
import argparse
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.utils import multi_gpu_model
from keras.models import Model
from keras.optimizers import SGD,Adam
from keras.regularizers import l2
from keras.layers import Input, Dense, Layer, Dropout, Conv2D, MaxPooling2D, Flatten, multiply
from .metrics import bag_accuracy, bag_loss
from .custom_layers import Mil_Attention, Last_Sigmoid

def cell_net(input_dim, args, useMulGpu=False):

    lr = args.init_lr
    weight_decay = args.init_lr
    momentum = args.momentum

    data_input = Input(shape=input_dim, dtype='float32', name='input')
    # conv1 = Conv2D(36, kernel_size=(4,4), kernel_regularizer=l2(weight_decay), activation='relu')(data_input)
    # conv1 = MaxPooling2D((2,2))(conv1)

    # conv2 = Conv2D(48, kernel_size=(3,3),  kernel_regularizer=l2(weight_decay), activation='relu')(conv1)
    # conv2 = MaxPooling2D((2,2))(conv2)
    conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=input_dim)(data_input)
    x = Flatten()(conv_base)
    fc1 = Dense(512, activation='relu',kernel_regularizer=l2(weight_decay), name='fc1')(x)
    fc1 = Dropout(0.5)(fc1)

    # softmax over N  torch.Size([1, n])
    alpha = Mil_Attention(L_dim=128, output_dim=1, kernel_regularizer=l2(weight_decay), name='alpha', use_gated=args.useGated)(fc1)
    x_mul = multiply([alpha, fc1]) # KxL torch.Size([1, 512])矩阵a和b矩阵相乘，比如a的维度是(1, 2)，b的维度是(2, 3)，返回的就是(1, 3)的矩阵

    out = Last_Sigmoid(output_dim=1, name='FC1_sigmoid')(x_mul) # 1x1 torch.Size([1, 1])输出预测概率，0-1之间
    #
    model = Model(inputs=[data_input], outputs=[out])

    print(model.summary())

    if useMulGpu == True:
        parallel_model = multi_gpu_model(model, gpus=2)
        parallel_model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss='categorical_crossentropy', metrics=[bag_accuracy])
    else:
        model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss='categorical_crossentropy', metrics=[bag_accuracy])
        parallel_model = model

    return parallel_model



