import tensorflow as tf
from tensorflow.keras.layers import Dense, RandomFlip, RandomRotation, RandomZoom, RandomHeight, RandomWidth, Rescaling
from tensorflow.keras import Sequential
from tensorflow.keras.activations import softmax

import cv2
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


RANDOM_SEED = 69
BATCH_SIZE  = 32
EPOCHS      = 30
IMAGE_SIZE  = (224, 224)
LABEL_MODE  = "categorical"
AUGMENTATION_FACTOR = 0.2
TRAIN_DIR   = "dataset_splitted/train"
TEST_DIR    = "dataset_splitted/val"
TAGS = [
    "animal", "car", "cloud", "dance", "fire", "flower", "food", "sunset", "water"
]


class VityaModel:
    def __init__(self) -> None:
        augmentaion_layer = Sequential([
            RandomFlip("horizontal", seed=RANDOM_SEED),
            RandomRotation(AUGMENTATION_FACTOR, seed=RANDOM_SEED),
            RandomZoom(AUGMENTATION_FACTOR, seed=RANDOM_SEED),
            RandomHeight(AUGMENTATION_FACTOR, seed=RANDOM_SEED),
            RandomWidth(AUGMENTATION_FACTOR, seed=RANDOM_SEED),
            Rescaling(1 / 255.)
        ])

        augmentaion_layer
        base_model = tf.keras.applications.ResNet50V2(include_top=False)
        base_model.trainable = False

        input_layer = tf.keras.layers.Input(shape=(224, 224, 3), name="input_layer")
        x = augmentaion_layer(input_layer)
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
        output_layer = Dense(len(TAGS), activation=softmax, name="output_layer")(x)

        model = tf.keras.Model(input_layer, output_layer)
        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["accuracy"]
        )
        model_checkpoint_path = "./checkpoints/vitya_weights"
        model.load_weights(model_checkpoint_path)
        print("Compiled VityaModel")

        self.model = model


Vitya = VityaModel()



def images_from_video(filepath: str) -> np.array:
    images = []

    vidcap = cv2.VideoCapture(filepath)
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    delta = length // 10
    current_frame, count = 1, 1
    for i in range(length):
        success, image = vidcap.read(current_frame)
        if success and i == current_frame:
            image = cv2.resize(image, (224, 224), interpolation = cv2.INTER_AREA)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = tf.expand_dims(np.asarray(image), axis=0)
            images.append(image)
            current_frame += delta
            count += 1

    vidcap.release()

    return images


def classify_image(image: tf.image) -> str:
    prediction = Vitya.model.predict(image)
    tag = TAGS[np.argmax(prediction)]
    return tag


def classify_video(filepath: str) -> list[str]:
    images = images_from_video(filepath)
    tags = np.array([])
    for image in images:
        tag = classify_image(image)
        print(tag)
        tags = np.append(tags, tag)
    print(tags)
    tags = Counter(tags)
    return tags.most_common()[0][0]
    


print(classify_video("train_video\_Argentina a cloud after sunset over Fitz Roy_preview.mp4"))
