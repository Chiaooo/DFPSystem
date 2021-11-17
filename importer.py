import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2.cv2 as cv2
import os
from PIL import Image as im
import tensorflow as tf
from sklearn.model_selection import train_test_split, GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout


# 设定tensorflow环境变量，指定使用GPU-0
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 1
sess = tf.compat.v1.Session(config=config)


def load_images_from_folder(folder, eyes=0):
    count = 0
    error_count = 0
    images = []
    for filename in os.listdir(folder):
        try:
            img = cv2.imread(os.path.join(folder, filename))
            img = cv2.resize(img, (80, 80))
            # 0：睁眼 1：闭眼
            images.append([img, eyes])
        except:
            error_count += 1
            print('导入异常数量：' + str(error_count))
            continue
        count += 1
        if count % 500 == 0:
            print('成功导入图片数量：' + str(count))
    return images


# 获取数据并加载函数（睁眼数据）
folder = "data/train/new_open_eyes"
open_eye = load_images_from_folder(folder, 0)

# 获取数据并加载函数（闭眼数据）
folder = "data/train/new_closed_eyes"
close_eye = load_images_from_folder(folder, 1)

# SUM
eye = close_eye + open_eye
data = im.fromarray(open_eye[0][0])

# X为图像 y为标签
X = []
y = []
for features, label in eye:
    X.append(features)
    y.append(label)
X = np.array(X).reshape(-1, 80, 80, 3)
y = np.array(y)
X = X / 255.0

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=24)
X_train = X_train.reshape(X_train.shape[0], 80, 80, 3)
X_test = X_test.reshape(X_test.shape[0], 80, 80, 3)

initial_GS = pd.read_csv('data/gridsearches/initial_gridsearch_results.csv').drop(columns='Unnamed: 0')
initial_GS.sort_values('rank_test_score').head()


def model_func_deep(dense_neurons_1=128, dense_neurons_2=128, dense_neurons_3=128, layout='2*3x3', filters=32,
                    dropout=None,
                    pooling=None):
    # 实例化模型
    model = Sequential()

    # 5x5与2*3x3布局
    if layout == '5x5':
        model.add(Conv2D(
            filters=filters,
            kernel_size=(5, 5),
            activation='relu',
            input_shape=(80, 80, 3)))
    if layout == '3x3' or layout == '2*3x3':
        model.add(Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            activation='relu',
            input_shape=(80, 80, 3)))

    # 卷积层池化

    if pooling != None:
        model.add(MaxPooling2D(pool_size=(2, 2)))

    # 第二卷积层

    if layout == '2*3x3':
        model.add(Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            activation='relu'))

        # 第二卷积层池化

    if pooling != None and layout == '2*3x3':
        model.add(MaxPooling2D(pool_size=(2, 2)))

    # 展平图像阵列，进入密集层

    model.add(Flatten())

    # 增加密集层和dropout,防止过拟合

    model.add(Dense(dense_neurons_1, activation='relu'))

    if dropout != None:
        model.add(Dropout(dropout))

    model.add(Dense(dense_neurons_2, activation='relu'))

    if dropout != None:
        model.add(Dropout(dropout))

    model.add(Dense(dense_neurons_3, activation='relu'))

    if dropout != None:
        model.add(Dropout(dropout))

        # 输出层，sigmoid用来处理二进制分类

    model.add(Dense(1, activation='sigmoid'))

    # 损失：二进制交叉熵，指标：PR

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=[tf.keras.metrics.AUC(curve='PR')])

    return model


early_stop = tf.keras.callbacks.EarlyStopping(min_delta=.05, patience=5)

params_deep = {
    "layout": ['2*3x3'],
    "dense_neurons_1": [128, 256],
    "dense_neurons_2": [512, 1028],
    "dense_neurons_3": [1028],
    "filters": [32],
    "dropout": [0.3],
    "pooling": [1],
    "epochs": [32]
}
nn_deep = KerasClassifier(build_fn=model_func_deep, batch_size=500, verbose=1)

gs_deep = GridSearchCV(estimator=nn_deep, param_grid=params_deep, cv=3, scoring='average_precision')

gs_deep.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=[early_stop])

# 存储网格搜索结果
df = pd.DataFrame(gs_deep.cv_results_)
df.to_csv('data/result/results.csv')
