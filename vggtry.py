import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import initializers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.optimizers import SGD
from keras import regularizers
import sys

#f = open("test.out", 'w')
#sys.stdout = f

# dimensions of our images.
img_width, img_height = 224, 224

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = 'data/Train'
validation_data_dir = 'data/Validation'
nb_train_samples = 1600
nb_validation_samples = 400
epochs = 10
batch_size = 10
category = 2
i=0
j=0


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)
    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    print("In Function")
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    print (bottleneck_features_train)
    np.save(open('bottleneck_features_train.npy', 'wb'),
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(open('bottleneck_features_validation.npy', 'wb'),
            bottleneck_features_validation)

def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy','rb'))
    #print (train_data)
    #train_labels = np.array([0] * int((float(nb_train_samples) / 3))+[1] * int((float(nb_train_samples) / 3)) + [2] * int((float(nb_train_samples) / 3)))
    train_labels = np.array([0] * 800 + [1] * 800)
    print (int(float(nb_train_samples) / category))
    validation_data = np.load(open('bottleneck_features_validation.npy','rb'))
    print (int(float(nb_validation_samples) / category))
    validation_labels = np.array([0] * 200 + [1] * 200)
    print (train_data.shape[1:])
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(512, activation='relu',kernel_initializer=initializers.glorot_uniform(seed = None)))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu',kernel_initializer=initializers.glorot_uniform(seed = None),kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5 ))
    model.add(Dense(1, activation='sigmoid'))

    sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    hist = model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    print(hist.history)
    model.save_weights(top_model_weights_path)


#save_bottlebeck_features()
train_top_model()
#f.close()