import matplotlib.pyplot as plt
import numpy as np
import PIL as image_lib
import tensorflow as tflow
import keras
import cv2
from keras.layers import Dense, Activation, Dropout, Reshape, Permute
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix
import pathlib
import os

# Import dataset
data_set = "dataset"
directory = data_set
data_dir = pathlib.Path(directory)


def main():
    # create the training subset
    img_height, img_width = 180, 180
    batch_size = 32
    train_ds = tflow.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        label_mode='categorical',
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # create the validation subset
    validation_ds = tflow.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        label_mode='categorical',
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # import the pre-trained learning model
    resnet_model = Sequential()
    pretrained_model = tflow.keras.applications.ResNet50(
        include_top=False,
        input_shape=(180, 180, 3),
        pooling='avg',
        classes=3,
        weights='imagenet'
    )

    for each_layer in pretrained_model.layers:
        each_layer.trainable = False
    resnet_model.add(pretrained_model)
    resnet_model.add(Flatten())
    resnet_model.add(Dense(512, activation='relu'))
    resnet_model.add(Dense(3, activation='softmax'))

    # Train and the Model
    resnet_model.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = resnet_model.fit(
        train_ds, validation_data=validation_ds, epochs=10)

    # Evaluate the Model and Visualize the Results
    plt.figure(figsize=(8, 8))
    epochs_range = range(10)
    plt.plot(epochs_range,
             history.history['accuracy'], label='Training Accuracy')
    plt.plot(epochs_range,
             history.history['val_accuracy'], label='Validation Accuracy')
    plt.axis(ymin=0.4, ymax=1)
    plt.title('Training and Validation Accuracy')
    plt.grid()
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'])
    plt.show()

    # Confusion Matrix Calculation
    y_true = []
    y_pred = []
    class_names = ["LTE", "NR", "NR_LTE"]
    for images, labels in validation_ds:
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        predictions = resnet_model.predict(images)
        y_pred.extend(np.argmax(predictions, axis=1))

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Calculate confusion matrix percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Plot confusion matrix with percentage values
    plt.figure(figsize=(8, 8))
    plt.imshow(cm_percent, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Add percentage values in each cell grid
    thresh = cm_percent.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f'{cm[i, j]} ({cm_percent[i, j]:.1f}%)',
                     horizontalalignment="center",
                     color="white" if cm_percent[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    # Predict a class for the image
    sample_image = cv2.imread(str('test.png'))
    sample_image_resized = cv2.resize(sample_image, (180, 180))
    sample_image = np.expand_dims(sample_image_resized, axis=0)
    # Predict a class for the image
    image_pred = resnet_model.predict(sample_image)

    # Convert into human readable output label
    image_output_class = class_names[np.argmax(image_pred)]
    print("The predicted class is: ", image_output_class)

    # Plot the model architecture and save it as an image
    plot_model(resnet_model, to_file='resnet50_model.png', show_shapes=True)


if __name__ == '__main__':
    main()
