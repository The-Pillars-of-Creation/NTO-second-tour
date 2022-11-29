import tensorflow as tf
from tensorflow.keras.layers import Dense, RandomFlip, RandomRotation, RandomZoom, RandomHeight, RandomWidth, Rescaling
from tensorflow.keras import Sequential
from tensorflow.keras.activations import softmax

import cv2
import numpy as np
import pandas as pd
from alive_progress import alive_bar
import os

from collections import Counter


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


RANDOM_SEED = 69
BATCH_SIZE = 32
EPOCHS = 30
IMAGE_SIZE = (224, 224)
LABEL_MODE = "categorical"
AUGMENTATION_FACTOR = 0.2
TRAIN_DIR = "dataset_splitted/train"
TEST_DIR = "dataset_splitted/val"
LABELS = {
    'water': 0, 
    'car': 1, 
    'cloud': 2, 
    'food': 3, 
    'flower': 4, 
    'dance': 5, 
    'animal': 6, 
    'sunset': 7,
    'fire': 8
}
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


Vitya = VityaModel().model



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
    prediction = Vitya.predict(image)
    tag = TAGS[np.argmax(prediction)]
    return tag


def classify_video(filepath: str) -> list[str]:
    images = images_from_video(filepath)
    tags = np.array([])
    for image in images:
        tag = classify_image(image)
        tags = np.append(tags, tag)
    tags = Counter(tags)
    return LABELS.get(tags.most_common()[0][0], 0)


def main():
    test_data = pd.read_csv("input/test.csv")
    predictions = []
    with alive_bar(len(test_data.path)) as bar:
        for path in test_data.path:
            prediction = classify_video(f"video/{path}")
            predictions.append(prediction)
            bar()
    
    out_data = pd.DataFrame({"path": test_data.path, "labels": predictions})
    os.makedirs("output", exist_ok=True)
    out_data.to_csv("output/predictions.csv", index=False)


if __name__ == "__main__":
    main()
