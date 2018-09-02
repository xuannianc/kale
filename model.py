from keras import layers
from keras import models
from sliding_window_layer import SlidingWindowLayer
from keras.layers.wrappers import TimeDistributed
import keras.backend as K


def ctc_lambda_func(args):
    # https://www.tensorflow.org/api_docs/python/tf/keras/backend/ctc_batch_cost
    y_true, y_pred, input_length, label_length = args
    return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)


def kale(input_shape=(32, 280, 1), num_classes=5991, max_string_len=10):
    input = layers.Input(shape=input_shape)
    swl_output = SlidingWindowLayer()(input)
    x = TimeDistributed(layers.Conv2D(50, (3, 3), padding='same', activation='relu', name='conv1'))(swl_output)
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


k = kale()
