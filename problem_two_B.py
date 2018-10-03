import os
import glob
import socket
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Model
from keras.utils import plot_model
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
plot_switch = False
ALPHA = 2.0
 
mnist_train_data_dir = os.path.join(base_dir, 'Images/MNIST/Train')
mnist_test_data_dir = os.path.join(base_dir, 'Images/MNIST/Test')
cifar_train_data_dir = os.path.join(base_dir, 'Images/CIFAR10/Train')
cifar_test_data_dir = os.path.join(base_dir, 'Images/CIFAR10/Test')


def define_model(input_shape, num_classes, NN_name):
    """ Define the convolutional neural network.

    Args:
    input_shape = (image_width, image_height, #channels) = shape of the images being fed to the input layer.
    """
    img_input = Input(shape=input_shape, name='{}_img_input'.format(NN_name))
    x = Conv2D(32, (3, 3), activation='relu', name='{}_conv1'.format(NN_name))(img_input)
    x = MaxPooling2D((2, 2), name='{}_pool1'.format(NN_name))(x)
    x = Conv2D(64, (3, 3), activation='relu', name='{}_conv2'.format(NN_name))(x)
    x = MaxPooling2D((2, 2), name='{}_pool2'.format(NN_name))(x)
    x = Conv2D(32, (3, 3), activation='relu', name='{}_conv3'.format(NN_name))(x)
    x = Flatten(name='{}_flatten'.format(NN_name))(x)
    x = Dense(64, activation='relu', name='{}_fc1'.format(NN_name))(x)
    out = Dense(num_classes, activation='softmax', name='{}_prediction'.format(NN_name))(x)
    model = Model(inputs=img_input, outputs=out, name="model_{}".format(NN_name))
    return model

def data_generators(train_data_dir, test_data_dir):
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
    return train_generator, test_generator


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
    mnist_train_generator, mnist_test_generator = data_generators(
        mnist_train_data_dir,
        mnist_test_data_dir)
    cifar_train_generator, cifar_test_generator = data_generators(
        cifar_train_data_dir,
        cifar_test_data_dir)

    mnist_num_classes = len(mnist_train_generator.class_indices)
    mnist_num_train_samples = len(mnist_train_generator.filenames)
    mnist_num_test_samples = len(mnist_test_generator.filenames)
    mnist_train_labels = to_categorical(
        mnist_train_generator.classes,
        num_classes=mnist_num_classes)
    mnist_test_labels = to_categorical(
        mnist_test_generator.classes,
        num_classes=mnist_num_classes)
    
    cifar_num_classes = len(cifar_train_generator.class_indices)
    cifar_num_train_samples = len(cifar_train_generator.filenames)
    cifar_num_test_samples = len(cifar_test_generator.filenames)
    cifar_train_labels = to_categorical(
        cifar_train_generator.classes,
        num_classes=cifar_num_classes)
    cifar_test_labels = to_categorical(
        cifar_test_generator.classes,
        num_classes=cifar_num_classes)

    # compile
    input_shape = (IMG_WIDTH, IMG_HEIGHT, num_channels)
    shared = Conv2D(32, (3, 3), activation='relu', name='shared_conv')

    img_input_mnist = Input(shape=input_shape, name='mnist_img_input')
    x_mnist = shared(img_input_mnist)
    x_mnist = MaxPooling2D((2, 2), name='mnist_pool1')(x_mnist)
    x_mnist = Conv2D(64, (3, 3), activation='relu', name='mnist_conv2')(x_mnist)
    x_mnist = MaxPooling2D((2, 2), name='mnist_pool2')(x_mnist)
    x_mnist = Conv2D(32, (3, 3), activation='relu', name='mnist_conv3')(x_mnist)
    x_mnist = Flatten(name='mnist_flatten')(x_mnist)
    x_mnist = Dense(64, activation='relu', name='mnist_fc1')(x_mnist)
    out_mnist = Dense(mnist_num_classes, activation='softmax', name='mnist_prediction')(x_mnist)
    model_mnist = Model(inputs=img_input_mnist, outputs=out_mnist, name="model_mnist")
    
    img_input_cifar = Input(shape=input_shape, name='cifar_img_input')
    x_cifar = shared(img_input_cifar)
    x_cifar = MaxPooling2D((2, 2), name='cifar_pool1')(x_cifar)
    x_cifar = Conv2D(64, (3, 3), activation='relu', name='cifar_conv2')(x_cifar)
    x_cifar = MaxPooling2D((2, 2), name='cifar_pool2')(x_cifar)
    x_cifar = Conv2D(32, (3, 3), activation='relu', name='cifar_conv3')(x_cifar)
    x_cifar = Flatten(name='cifar_flatten')(x_cifar)
    x_cifar = Dense(64, activation='relu', name='cifar_fc1')(x_cifar)
    out_cifar = Dense(cifar_num_classes, activation='softmax', name='cifar_prediction')(x_cifar)
    model_cifar = Model(inputs=img_input_cifar, outputs=out_cifar, name="model_cifar")
    
    # model_mnist = define_model(
    #     (IMG_WIDTH, IMG_HEIGHT, num_channels),
    #     mnist_num_classes,
    #     "mnist")
    # print("[DEBUG] model_mnist.img_input: ", model_mnist(img_input))
    # model_cifar = define_model(
    #     (IMG_WIDTH, IMG_HEIGHT, num_channels),
    #     cifar_num_classes,
    #     "cifar")
    
    merged = Model(inputs=[img_input_mnist, img_input_cifar], outputs=[out_mnist, out_cifar])
    # plot_model(merged, to_file='demo.png', show_shape=True)
    merged.compile(
        optimizer='rmsprop',
        # loss=['categorical_crossentropy', 'categorical_crossentropy'],
        # loss_weights=[1., ALPHA],
        loss={'mnist_prediction': 'categorical_crossentropy', 'cifar_prediction': 'categorical_crossentropy'},
        loss_weights={'mnist_prediction': 1., 'cifar_prediction': ALPHA},
        metrics=['accuracy'])
    history = merged.fit_generator(
        [mnist_train_generator, cifar_train_generator],
        steps_per_epoch= mnist_num_train_samples/TRAIN_BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=[mnist_test_generator, cifar_test_generator],
        validation_steps=mnist_num_test_samples/TEST_BATCH_SIZE)


    if plot_switch:
        plot_training(history)


if __name__ == '__main__':
    train()
    