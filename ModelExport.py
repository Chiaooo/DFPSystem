import numpy as np
import tensorflow as tf
import cv2.cv2 as cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout


# 设定tensorflow环境变量，指定使用GPU-0
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


def load_images_from_folder(folder, eyes=0):
    count = 0
    error_count = 0
    images = []
    for filename in os.listdir(folder):
        try:
            img = cv2.imread(os.path.join(folder, filename))
            img = cv2.resize(img, (80, 80))
            images.append([img, eyes])
        except:
            error_count += 1
            print('导入异常数量：' + str(error_count))
            continue
        count += 1
        if count % 500 == 0:
            print('成功导入图片数量：' + str(count))

    return images


folder = "data/train/new_open_eyes"
open_eye = load_images_from_folder(folder, 0)
folder = "data/train/new_closed_eyes"
close_eye = load_images_from_folder(folder, 1)
eye = close_eye + open_eye

X = []
y = []

for features, label in eye:
    X.append(features)
    y.append(label)
X = np.array(X).reshape(-1, 80, 80, 3)
y = np.array(y)
X = X / 255.0
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=24)
X_train = X_train.reshape(X_train.shape[0], 80, 80, 3)
X_test = X_test.reshape(X_test.shape[0], 80, 80, 3)

model = Sequential()
# 第一道2D层
model.add(Conv2D(
    filters=32,  # 过滤器数量
    kernel_size=(3, 3),  # 过滤器尺寸
    activation='relu',  # 激活函数
    input_shape=(80, 80, 3)  # 裁剪图片
))

# 池化
model.add(MaxPooling2D(pool_size=(2, 2)))  # 池化的维度

# 第二道2D层
model.add(Conv2D(
    filters=32,
    kernel_size=(3, 3),
    activation='relu'
))

# 第三道2D层
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

# 用256个节点加入第一个密集层
model.add(Dense(256, activation='relu'))
# 防止过拟合
model.add(Dropout(0.4))

# 52->第二个密集层
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))

# 1024->第三个密集层
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=[tf.keras.metrics.AUC(curve='PR')])

model.fit(X_train, y_train, batch_size=400, epochs=32)
score = model.evaluate(X_test, y_test, verbose=1)

labels = model.metrics_names
preds = model.predict(X_test)
confusion_matrix = tf.math.confusion_matrix(labels=y_test, predictions=preds.round()).numpy()

# 分类指标
tn, fp, fn, tp = confusion_matrix.ravel()

# 特效度
specificty = tn / (tn + fp)
print(f'specificity: {specificty}')

# 灵敏度
sensitivity = tp / (tp + fn)
print(f'sensitivity / recall: {sensitivity}')

# 正确率
accuracy = (tn + tp) / (tn + fp + fn + tp)
print(f'accuracy: {accuracy}')

# 精度
precision = tp / (tp + fp)
print(f'precision: {precision}')

# 应多重循环确定最优指标
model.save('output/best_model_5.h5')
