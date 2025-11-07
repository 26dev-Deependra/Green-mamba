import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array




# Define dataset directory and classes
dataset_dir = "dataset"
classes = ["LTE", "LTE_NR", "NR"]  # Folder names representing the classes
class_mapping = {cls: idx for idx, cls in enumerate(classes)}

# Parameters
image_size = (180, 180)  # Image size for resizing
batch_size = 1
num_classes = len(classes)

# Load images and labels
def load_data(dataset_dir, classes, image_size):
    images = []
    labels = []
    for cls in classes:
        folder_path = os.path.join(dataset_dir, cls)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = load_img(img_path, target_size=image_size)
            img_array = img_to_array(img) / 255.0  # Normalize pixel values
            images.append(img_array)
            
            # Create a label mask with the same size as the image
            label_mask = np.zeros((image_size[0], image_size[1], num_classes))
            label_mask[:, :, class_mapping[cls]] = 1  # One-hot encode the class
            labels.append(label_mask)
    return np.array(images), np.array(labels)

# Load the data
images, labels = load_data(dataset_dir, classes, image_size)

# Split data into train, validation, and test sets
train_images, temp_images, train_labels, temp_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42
)
val_images, test_images, val_labels, test_labels = train_test_split(
    temp_images, temp_labels, test_size=0.5, random_state=42
)

# Convert to TensorFlow datasets
def to_tf_dataset(images, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(buffer_size=len(images)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Prepare datasets
train_dataset = to_tf_dataset(train_images, train_labels, batch_size)
val_dataset = to_tf_dataset(val_images, val_labels, batch_size)
test_dataset = to_tf_dataset(test_images, test_labels, batch_size)

# Define U-Net model
def unet_model(input_shape, num_classes):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Encoder
    conv1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    # Bottleneck
    bottleneck = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)

    # Decoder
    up1 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(bottleneck)
    concat1 = tf.keras.layers.concatenate([conv1, up1], axis=-1)
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(concat1)
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)

    # Output layer
    outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='softmax')(conv2)

    return tf.keras.models.Model(inputs, outputs)

# Initialize U-Net
input_shape = (image_size[0], image_size[1], 3)  # Input shape for the model
model = unet_model(input_shape, num_classes)

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # For multi-class segmentation
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,
    batch_size=batch_size
)

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")