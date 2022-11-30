# Vitya model

Наша нейросеть Витя умеет различать видео и классифицировать их на 9 категорий

| label | id |
| --- | --- |
| water | 0 |
| car | 1 |
| cloud | 2 |
| food | 3 |
| flower | 4 |
| dance | 5 |
| animal | 6 |
| sunset | 7 |
| fire | 8 |

Мы использовали ResNet50V2 для покадравой классификации видео и давали тэг по наиболее частому тэгу кадров.


```python
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
        base_model = tf.keras.applications.ResNet50V2(include_top=False, weights="./checkpoints/vitya_weights")
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
        
        print("Compiled VityaModel")

        self.model = model
```

## Обучение модели [train.ipynb](train.ipynb)

```python
history_1 = model_1.fit(
    train_data,
    epochs=EPOCHS,
    steps_per_epoch=len(train_data),
    validation_data=test_data,
    validation_steps=int(0.15 * len(test_data)),
    callbacks=[
        tf.keras.callbacks.CSVLogger("history/history.csv"),
        checkpoint_callback,
        create_tensorboard_callback(dir_name="tensorboard", experiment_name="vitya")
    ]
)
```

| Файл | Назначение |
| --- | --- |
| [solution.py](solution.py) | итоговое решение для отправки в систему НТО |
| [solution.ipynb](solution.ipynb) | тестировка модели на тестовых фотографиях |
| [train.ipynb](train.ipynb) | обучение модели |
| [first_try.ipynb](first_try.ipynb) | первая попытка обучения модели |
| [dataload.py](dataload.py) | первоначальная загрузка данных для pytorch, не используется |