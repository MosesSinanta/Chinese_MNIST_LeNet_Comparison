import cv2
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

def preprocess_dataset(csv_file, dataset_path):
    data = pd.read_csv(csv_file)
    
    images = []
    labels = []

    for index, row in data.iterrows():
        suite_id = row["suite_id"]
        sample_id = row["sample_id"]
        code = row["code"]
        
        file_name = f"input_{suite_id}_{sample_id}_{code}.jpg"
        file_path = os.path.join(dataset_path, file_name)
        
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            image = cv2.resize(image, (32, 32))
            images.append(image)
            
            label = int(code) - 1
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)
    
    images = images.astype("float32") / 255.0

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size = 0.2, random_state = 42)

    return X_train, X_test, y_train, y_test
