'''
  Tensorflow 2.1 + Python3.6環境にて動作確認
'''
import os
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage import util
from sklearn.model_selection import train_test_split
from PIL import Image

# --- Parameter Setting ---
# Train with Extended ImageDataGenerator
USE_EXT_GENERATOR = True

# 最大エポック数
MAX_EPOCH = 100

# 打ち切り判断数
ES_PATIENCE = 10
# SaveBestOnly
SAVE_BEST_ONLY = True

# バッチサイズ
BATCH_SIZE = 16

# 入力データサイズ
IMAGE_W = 32
IMAGE_H = 32

# 分類カテゴリ数
N_CATEGORIES = 10

# 初期学習率
LEARN_RATE = 0.001

# チェックポイント保存フォルダ
CP_DIR = 'checkpoint'
DEBUG_DIR = 'debug'
os.makedirs(CP_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

# --- Model Setting ---

## CNN定義 簡単な分類ならこれぐらいで可能
def Conv_L4(input_shape):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape, kernel_initializer='random_uniform', activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='random_uniform', activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='random_uniform', activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2, 2)))
    model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='random_uniform', activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2, 2)))
    return model

'''
拡張ImageDataGenerator
'''
class ExtImageDataGenerator(ImageDataGenerator):
    def __init__(self,
                 noise_pb=0.25,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        # noise parameter
        assert noise_pb >= 0.0
        self.noise_pb = noise_pb

    # My Augumentation
    def add_noise(self, rgb):
        return util.random_noise(rgb, var=0.001)

    # Override flow()
    def flow(self, *args, **kwargs):
        batches = super().flow(*args, **kwargs)

        loop_cont = 0
        while True:
            loop_cont += 1
            batch_x, batch_y = next(batches)
            # My Augumentation
            for i,img in enumerate(batch_x):
                if np.random.random() < self.noise_pb:
                    batch_x[i] = self.add_noise(img)

            if loop_cont <= 4: #膨大になるので最初の何バッチかだけ出力
                for i in range(batch_x.shape[0]):
                    img_to_save = Image.fromarray(np.uint8(batch_x[i]*255.0))
                    img_to_save.save(os.path.join(DEBUG_DIR,str(loop_cont)+'_'+str(i)+'.png'))

            yield (batch_x, batch_y)

    # Override flow_from_directory()
    def flow_from_directory(self, *args, **kwargs):
        batches = super().flow_from_directory(*args, **kwargs)

        while True:
            batch_x, batch_y = next(batches)
            # My Augumentation
            for i,img in enumerate(batch_x):
                if np.random.random() < self.noise_pb:
                    batch_x[i] = self.add_noise(img)

            yield (batch_x, batch_y)


# Input layer 作成
input_tensor = Input(shape=(IMAGE_H, IMAGE_W, 3))

# Base Model 作成
base_model = Conv_L4((IMAGE_H, IMAGE_W, 3))

# Output layer 作成
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(N_CATEGORIES, activation='softmax')(x)

# 全体Model作成
model = Model(inputs=base_model.input, outputs=predictions)

# Optimizer 選択
model.compile(optimizer=SGD(lr=LEARN_RATE, momentum=0.8), loss='categorical_crossentropy',metrics=['accuracy'])

# Summary表示
model.summary()


# --- Train Setting ---

# 学習セットのロード(CIFAR-10)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# testは最後の評価に使うので、validationは別途trainから分割して使う
(x_train, x_valid, y_train, y_valid) = train_test_split(x_train, y_train, test_size=0.3)

# 正規化
x_train = x_train.astype('float32') / 255.0
x_valid = x_valid.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
# One-Hot Vector化
y_train = keras.utils.to_categorical(y_train, N_CATEGORIES)
y_valid = keras.utils.to_categorical(y_valid, N_CATEGORIES)
y_test = keras.utils.to_categorical(y_test, N_CATEGORIES)

# Callback選択
cb_funcs = []

# Checkpoint作成設定
check_point = ModelCheckpoint(filepath = os.path.join(CP_DIR, 'epoch{epoch:03d}-{val_loss:.5f}.hdf5'), monitor='val_loss', verbose=1, save_best_only=SAVE_BEST_ONLY, mode='auto')
cb_funcs.append(check_point)

# Early-stopping Callback設定
if ES_PATIENCE >= 0:
    early_stopping = EarlyStopping(patience=ES_PATIENCE, verbose=1)
    cb_funcs.append(early_stopping)

# Generator定義
## Train
if USE_EXT_GENERATOR:
    train_datagen = ExtImageDataGenerator(horizontal_flip = True, channel_shift_range = 0.3, noise_pb=0.5)
else :
    train_datagen = ImageDataGenerator()
train_generator = train_datagen.flow(
   x_train, y_train,
   batch_size=BATCH_SIZE,
   shuffle=True,
   seed = 12345
)
## Validation
if USE_EXT_GENERATOR:
    valid_datagen = ExtImageDataGenerator()
else:
    valid_datagen = ImageDataGenerator()
valid_generator = valid_datagen.flow(
   x_valid, y_valid,
   batch_size=BATCH_SIZE,
   shuffle=True,
   seed = 12345
)



# モデル訓練実行
history = model.fit_generator(
    train_generator,
    steps_per_epoch=x_train.shape[0]//BATCH_SIZE,
    epochs=MAX_EPOCH,
    verbose=1,
    validation_data=valid_generator,
    validation_steps=x_test.shape[0]//BATCH_SIZE,
    callbacks=cb_funcs
)


# 評価
ev = model.evaluate(x_test,y_test)
print("loss:{}".format(ev[0]))
print("acc :{}".format(ev[1]))
