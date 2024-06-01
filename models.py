import tensorflow as tf
from tensorflow.keras import layers, models

def lenet4_model(input_shape, num_classes):
    model = models.Sequential()

    model.add(layers.Conv2D(6, kernel_size = (5, 5), strides = (1, 1), activation = "tanh", input_shape = input_shape))
    model.add(layers.AveragePooling2D(pool_size = (2, 2), strides = (2, 2)))

    model.add(layers.Conv2D(16, kernel_size = (5, 5), strides = (1, 1), activation = "tanh"))
    model.add(layers.AveragePooling2D(pool_size = (2, 2), strides = (2, 2)))

    model.add(layers.Conv2D(120, kernel_size = (5, 5), strides = (1, 1), activation = "tanh"))

    model.add(layers.Flatten())

    model.add(layers.Dense(84, activation = "tanh"))
    model.add(layers.Dense(num_classes, activation = "softmax"))

    model.compile(loss = "sparse_categorical_crossentropy", optimizer = "SGD", metrics = ["accuracy"])

    return model


def lenet5_model(input_shape, num_classes):
    model = models.Sequential()
    
    model.add(layers.Conv2D(6, kernel_size = (5, 5), activation = "tanh", input_shape = input_shape))
    model.add(layers.AveragePooling2D(pool_size = (2, 2)))

    model.add(layers.Conv2D(16, kernel_size = (5, 5), activation = "tanh"))
    model.add(layers.AveragePooling2D(pool_size = (2, 2)))

    model.add(layers.Flatten())
    
    model.add(layers.Dense(120, activation = "tanh"))
    model.add(layers.Dense(84, activation = "tanh"))
    model.add(layers.Dense(num_classes, activation = "softmax"))

    model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
    
    return model