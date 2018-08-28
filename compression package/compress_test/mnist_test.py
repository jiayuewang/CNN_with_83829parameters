import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from nnCompression.IterativeCompresser import IterativeCompresser

batch_size = 100
num_classes = 10
epochs = 1

img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['acc'])

compresser = IterativeCompresser(model=model, metric_name='acc', metric_tolerance=0.99, sparsity={'conv2d_1': 0.94, 'conv2d_2': 0.97, 'dense_1': 0.98, 'dense_2': 0.82},
                 input_filepath='weights.h5',
                 output_filepath='out.h5',
                 extra_sparsity=0.05,
                 decrease_rate=0.01)

compresser.compress(
                 x=x_train,
                 y=y_train,
                 batch_size=batch_size,
                 epochs=1,
                 verbose=1,
                 validation_data=(x_test, y_test),
                 shuffle=True,
                 class_weight=None,
                 sample_weight=None,
                 steps_per_epoch=None,
                 validation_steps=None)
print(compresser.sparsity)