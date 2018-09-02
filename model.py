from keras import layers
from keras import models
from sliding_window_layer import SlidingWindowLayer
from keras.layers.wrappers import TimeDistributed
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras import callbacks
from keras.optimizers import SGD
from hdf5 import HDF5DatasetGenerator
from callback import *
from config import *


def ctc_lambda_func(args):
    # https://www.tensorflow.org/api_docs/python/tf/keras/backend/ctc_batch_cost
    y_true, y_pred, input_length, label_length = args
    return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)


# def sliding_window_layer(inputs, window_width=32, slide_stride=4):
#     # inputs: batches*32*280*1
#     for b in range(inputs.shape[0]):
#         batch_input = inputs[b, :, :, :].reshape((1, inputs.shape[1], inputs.shape[2], inputs.shape[3]))
#         for w in range(0, batch_input.shape[2] - window_width, slide_stride):
#             if w == 0:
#                 output_batch = batch_input[:, :, w:(w + 1) * window_width, :]
#             else:
#                 output_batch = np.concatenate((output_batch, batch_input[:, :, w:w + window_width, :]), axis=0)
#
#         if b == 0:
#             output = output_batch
#         else:
#             output = np.concatenate((output, output_batch), axis=0)
#     return output
def sliding_window_layer(inputs, window_width=32, slide_stride=4):
    # inputs: (batch_size,32,280,1)
    batch_size, height, width = inputs.shape[:3]
    num_steps = (width - window_width) // slide_stride
    windows = []
    for step_idx in range(num_steps):
        start = slide_stride * step_idx
        end = start + window_width
        window = inputs[:, :, start:end, :]
        # window = K.expand_dims(window, axis=1)
        windows.append(window)
    output = K.stack(windows, axis=1)
    # get_shape() 返回的是 TensorShape 对象
    # swl_output_shape = output.get_shape().as_list()
    return output


def swl_lambda_func(input):
    swl_output = tf.py_func(sliding_window_layer, [input], tf.float32)
    return swl_output


def kale(input_shape=(32, None, 1), num_classes=5991, max_string_len=10):
    input = layers.Input(shape=input_shape)
    # swl_output = tf.py_func(sliding_window_layer, [input], tf.float32)
    swl_output = layers.Lambda(swl_lambda_func)(input)
    xx = layers.Reshape((-1, 32, 32, 1))(swl_output)
    print(xx.get_shape())
    print(xx.shape)
    xx._keras_shape = tuple(xx.get_shape().as_list())
    print(K.int_shape(xx))
    # swl_output = SlidingWindowLayer()(input)
    x = TimeDistributed(layers.Conv2D(50, (3, 3), padding='same', activation='relu', name='conv1'))(xx)
    x = TimeDistributed(layers.BatchNormalization(axis=-1))(x)
    x = TimeDistributed(layers.Conv2D(100, (3, 3), padding='same', activation='relu', name='conv2'))(x)
    x = TimeDistributed(layers.Dropout(0.1))(x)
    x = TimeDistributed(layers.Conv2D(100, (3, 3), padding='same', activation='relu', name='conv3'))(x)
    x = TimeDistributed(layers.Dropout(0.1))(x)
    x = TimeDistributed(layers.BatchNormalization(axis=-1))(x)
    x = TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(layers.Conv2D(150, (3, 3), padding='same', activation='relu', name='conv4'))(x)
    x = TimeDistributed(layers.BatchNormalization(axis=-1))(x)
    x = TimeDistributed(layers.Conv2D(200, (3, 3), padding='same', activation='relu', name='conv5'))(x)
    x = TimeDistributed(layers.Dropout(0.2))(x)
    x = TimeDistributed(layers.Conv2D(200, (3, 3), padding='same', activation='relu', name='conv6'))(x)
    x = TimeDistributed(layers.Dropout(0.2))(x)
    x = TimeDistributed(layers.BatchNormalization(axis=-1))(x)
    x = TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(layers.Conv2D(250, (3, 3), padding='same', activation='relu', name='conv7'))(x)
    x = TimeDistributed(layers.BatchNormalization(axis=-1))(x)
    x = TimeDistributed(layers.Conv2D(300, (3, 3), padding='same', activation='relu', name='conv8'))(x)
    x = TimeDistributed(layers.Dropout(0.3))(x)
    x = TimeDistributed(layers.Conv2D(300, (3, 3), padding='same', activation='relu', name='conv9'))(x)
    x = TimeDistributed(layers.Dropout(0.3))(x)
    x = TimeDistributed(layers.BatchNormalization(axis=-1))(x)
    x = TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(layers.Conv2D(350, (3, 3), padding='same', activation='relu', name='conv10'))(x)
    x = TimeDistributed(layers.BatchNormalization(axis=-1))(x)
    x = TimeDistributed(layers.Conv2D(400, (3, 3), padding='same', activation='relu', name='conv11'))(x)
    x = TimeDistributed(layers.Dropout(0.4))(x)
    x = TimeDistributed(layers.Conv2D(400, (3, 3), padding='same', activation='relu', name='conv12'))(x)
    x = TimeDistributed(layers.Dropout(0.4))(x)
    x = TimeDistributed(layers.BatchNormalization(axis=-1))(x)
    x = TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(layers.Flatten())(x)
    x = TimeDistributed(layers.Dense(900, activation='relu'))(x)
    x = TimeDistributed(layers.Dropout(0.5))(x)
    x = TimeDistributed(layers.Dense(200, activation='relu'))(x)
    classifier_output = TimeDistributed(layers.Dense(num_classes, activation='softmax'))(x)
    # classifier = models.Model(inputs=input, outputs=classifier_output)
    # classifier.summary()
    label = layers.Input(name='label', shape=[max_string_len], dtype='int64')
    # 序列的长度,此模型中为
    seq_length = layers.Input(name='seq_length', shape=[1], dtype='int64')
    label_length = layers.Input(name='label_length', shape=[1], dtype='int64')
    ctc_output = layers.Lambda(ctc_lambda_func, output_shape=(1,),
                               name='ctc')([label, classifier_output, seq_length, label_length])
    model = models.Model(inputs=[input, label, seq_length, label_length], outputs=[ctc_output])
    model.summary()
    return model


model = kale()
# clipnorm seems to speeds up convergence
sgd = SGD(lr=0.1, momentum=0.9, nesterov=True, clipnorm=5)
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
# model = models.load_model('models/synthetic_model_0829_3000000_1.3227_1.0404.hdf5',
#                   custom_objects={'<lambda>': lambda y_true, y_pred: y_pred})
gen = HDF5DatasetGenerator(TRAIN_DB_PATH, batch_size=100).generator
val_gen = HDF5DatasetGenerator(VALIDATION_DB_PATH, batch_size=100).generator

# callbacks
training_monitor = TrainingMonitor(figure_path='synthetic_0829_3000000.jpg', json_path='synthetic_0829_3000000.json',
                                   start_at=5)
accuracy_evaluator = AccuracyEvaluator(TEST_DB_PATH, batch_size=100)
learning_rate_updator = LearningRateUpdator(init_lr=0.1)
callbacks = [
    # Interrupts training when improvement stops
    callbacks.EarlyStopping(
        # Monitors the model’s validation accuracy
        monitor='val_loss',
        # Interrupts training when accuracy has stopped
        # improving for more than one epoch (that is, two epochs)
        patience=10,
    ),
    # Saves the current weights after every epoch
    callbacks.ModelCheckpoint(
        # Path to the destination model file
        filepath='models/synthetic_model_0829_3000000.hdf5',
        # These two arguments mean you won’t overwrite the
        # model file unless val_loss has improved, which allows
        # you to keep the best model seen during training.
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    training_monitor,
    # accuracy_evaluator
    learning_rate_updator
]
model.fit_generator(gen(), steps_per_epoch=3000000 // 100,
                    callbacks=callbacks,
                    epochs=100,
                    validation_data=val_gen(),
                    validation_steps=279600 // 100)
