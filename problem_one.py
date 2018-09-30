import os
import glob
import socket
import matplotlib.pyplot as plt

from keras import layers
from keras import models
from keras import optimizers

from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator


host = socket.gethostname()
if host == "cibl-thinkpad" or "R2Q5":
    base_dir = os.path.dirname(os.path.realpath(__file__))
    try:
        import multiprocessing
        num_cores = multiprocessing.cpu_count() 
    except (ImportError, NotImplementedError):
        num_cores = 2
        pass
else:
  print ('Unknown host!')
  sys.exit()


# define hyperparameters and data paths
EPOCHS = 200
TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 16
IMG_WIDTH, IMG_HEIGHT, num_channels = 28, 28, 3
plot_switch = True

train_data_dir = os.path.join(base_dir, 'Images/Cells/Train')
test_data_dir = os.path.join(base_dir, 'Images/Cells/Test')


def define_model(input_shape):
    """ Define the convolutional neural network.

    Args:
    input_shape = (image_width, image_height, #channels) = shape of the images being fed to the input layer.
    """
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))
    return model


def plot_training(history_name):
    """ Plot the traing and test accuracies and loss during training the Neural Net.

    Args:
    history_name = The history object returned by the function "fit_generator".
    """
    acc = history_name.history['acc']
    test_acc = history_name.history['val_acc']
    loss = history_name.history['loss']
    test_loss = history_name.history['val_loss']

    epochs = range(1,len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, test_acc, 'r', label='Validation accuracy')
    plt.title('Training and test accuracy')
    plt.legend()
    plt.axhline(y=0.9, color='black', linestyle='--')
    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, test_loss, 'r', label='Test loss')
    plt.title('Training and test loss')
    plt.legend()
    plt.show()


def train():
    """ Define data streams and train the Neural Net.
    
    """
    datagen = ImageDataGenerator(
    	rescale=1. / 255,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.25,
        fill_mode='nearest')

    train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=TRAIN_BATCH_SIZE,
        class_mode='categorical',
        shuffle=True)

    test_generator = datagen.flow_from_directory(
        test_data_dir,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=TEST_BATCH_SIZE,
        class_mode='categorical',
        shuffle=False)

    num_classes = len(train_generator.class_indices)
    num_train_samples = len(train_generator.filenames)
    train_labels = to_categorical(train_generator.classes, num_classes=num_classes)
    num_test_samples = len(test_generator.filenames)
    test_labels = to_categorical(test_generator.classes, num_classes=num_classes)

    # compile
    model = define_model((IMG_WIDTH, IMG_HEIGHT, num_channels))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit_generator(
    	train_generator,
    	steps_per_epoch= num_train_samples/TRAIN_BATCH_SIZE,
    	epochs=EPOCHS,
    	validation_data=test_generator,
    	validation_steps=num_test_samples/TEST_BATCH_SIZE)

    model.save('four_class_cell_classification.h5')

    if plot_switch:
        plot_training(history)


if __name__ == '__main__':
    train()
    