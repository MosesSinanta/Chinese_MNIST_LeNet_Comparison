import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from preprocessing import preprocess_dataset
from models import lenet4_model, lenet5_model

def format_float(value):
    return '{:.3f}'.format(value)

csv_file = "chinese_mnist.csv"
dataset = "dataset"

X_train, X_test, y_train, y_test = preprocess_dataset(csv_file, dataset)

X_train = np.expand_dims(X_train, axis = -1)
X_test = np.expand_dims(X_test, axis = -1)

num_classes = len(np.unique(y_train))

#model = lenet4_model(input_shape = (32, 32, 1), num_classes = num_classes)
model = lenet5_model(input_shape = (32, 32, 1), num_classes = num_classes)

history = model.fit(X_train, y_train, epochs = 100, batch_size = 32, validation_split = 0.2)

y_pred = model.predict(X_test).argmax(axis = 1)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose = 2)

conf_matrix = confusion_matrix(y_test, y_pred)

train_loss = history.history["loss"]
val_loss = history.history["val_loss"]
train_acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

train_loss = [format_float(loss) for loss in history.history["loss"]]
val_loss = [format_float(loss) for loss in history.history["val_loss"]]
train_acc = [format_float(acc) for acc in history.history["accuracy"]]
val_acc = [format_float(acc) for acc in history.history["val_accuracy"]]
test_loss = format_float(test_loss)
test_acc = format_float(test_acc)

results = {
    "train_loss": train_loss,
    "val_loss": val_loss,
    "train_accuracy": train_acc,
    "val_accuracy": val_acc,
    "test_accuracy": test_acc,
    "test_loss": test_loss,
    "confusion_matrix": conf_matrix.tolist()
}

with open("LeNet-5 results.json", "w") as file:
    json.dump(results, file)

print("Results saved to JSON")