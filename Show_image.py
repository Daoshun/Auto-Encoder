import tensorflow
import keras
import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Model  # 泛型模型
from keras.layers import Dense, Input
from keras.layers.core import Dropout
from keras.layers.noise import GaussianNoise
import matplotlib.pyplot as plt

# X shape (60,000 28x28), y shape (10,000, )
(x_train, _), (x_test, y_test) = mnist.load_data()

# 数据预处理
x_train = x_train.astype('float32') / 255. - 0.5  # minmax_normalized
x_test = x_test.astype('float32') / 255. - 0.5  # minmax_normalized
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))
# x_train = np.random.normal(x_train)
# x_test = np.random.normal(x_test)  # 增加噪声
print(x_train.shape)
print(x_test.shape)
print(x_train[0])
print(x_test[0])