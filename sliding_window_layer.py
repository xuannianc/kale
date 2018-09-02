from keras import backend as K
from keras.engine.topology import Layer
import numpy as np


class SlidingWindowLayer(Layer):
    def __init__(self, window_width=32, slide_stride=4, **kwargs):
        """
        :param window_width: 滑动窗口高度一定为 32, 宽度默认为 32
        :param slide_stride: 滑动的步长, 根据步长计算窗口的个数
        :param kwargs:
        """
        self.slide_stride = slide_stride
        self.window_width = window_width
        # 因为 output_shape 这个属性应该已经被使用, 所以这里用 swl_output_shape 来代替
        self.swl_output_shape = None
        super(SlidingWindowLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SlidingWindowLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        batch_size, height, width = x.shape[:3]
        num_steps = (width - self.window_width) // self.slide_stride
        windows = []
        for step_idx in range(num_steps):
            start = self.slide_stride * step_idx
            end = start + self.window_width
            window = x[:, :, start:end, :]
            # window = K.expand_dims(window, axis=1)
            windows.append(window)
        output = K.stack(windows, axis=1)
        # get_shape() 返回的是 TensorShape 对象
        self.swl_output_shape = output.get_shape().as_list()
        return output

    def compute_output_shape(self, input_shape):
        # 必须返回 tuple 而不能是 TensorShape
        return tuple(self.swl_output_shape)
