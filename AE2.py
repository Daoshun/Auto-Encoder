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
x_train = np.random.normal(x_train)
#x_test = np.random.normal(x_test)  # 增加噪声
print(x_train.shape)
print(x_test.shape)

# 压缩特征维度至2维
encoding_dim = 2

# this is our input placeholder
input_img = Input(shape=(784,))

# 编码层
# encoded = GaussianNoise(0.3)(input_img)
# encoded = Dense(128,activation='relu')(encoded)
encoded = Dense(1000, activation='relu')(input_img)
encoded = Dense(500, activation='relu')(encoded)
encoded = Dense(250, activation='relu')(encoded)
encoded = Dense(30, activation='relu')(encoded)
encoder_output = Dense(encoding_dim)(encoded)

# 解码层
decoded = Dense(30, activation='relu')(encoder_output)
encoded = Dense(250, activation='relu')(encoded)
decoded = Dense(500, activation='relu')(decoded)
decoded = Dense(1000, activation='relu')(decoded)
decoded = Dense(784, activation='tanh')(decoded)

# 构建自编码模型
autoencoder = Model(inputs=input_img, outputs=decoded)
#Model.add(Dropout(0.5))#dropout 降低train的正确率，但是会提高test的正确率

# 构建编码模型
encoder = Model(inputs=input_img, outputs=encoder_output)
#encoder = Model.add(Dropout(0.5))#dropout 降低train的正确率，但是会提高test的正确率

# compile autoencoder
autoencoder.compile(optimizer='adam', loss='mse')

# training
autoencoder.fit(x_train, x_train, epochs=20, batch_size=256, shuffle=True)

# result = Model.evaluate(x_train, y_train)
# print('\nTrain Acc', result[1])
# result = Model.evaluate(x_test, y_test)
# print('\nTest Acc', result[1])

# plotting
encoded_imgs = encoder.predict(x_test)
plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test, s=3)
plt.colorbar()
plt.show()