import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

dataset_path = r"C:\Users\MTK\OneDrive\Desktop\datasets"  # PLS WRITE YOUR DATASET PATH :) OWNER : MEHMET TALHA KUMCU

img_paths = []
labels = []

for image_name in os.listdir(dataset_path):
    if image_name.startswith("example") and image_name.endswith(".jpg"):
        image_path = os.path.join(dataset_path, image_name)
        img_paths.append(image_path)
        class_name = image_name.split(".")[0]
        labels.append(class_name)


train_paths, test_paths, train_labels, test_labels = train_test_split(
    img_paths, labels, test_size=0.2, random_state=42
)
train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_paths, train_labels, test_size=0.25, random_state=42
)


def extract_features(img_paths):
    features = []

    for image_path in img_paths:
        image = cv2.imread(image_path)
        hog = cv2.HOGDescriptor()
        hog_features = hog.compute(image)
        features.append(hog_features.flatten())

    return features


train_features = extract_features(train_paths)
val_features = extract_features(val_paths)
test_features = extract_features(test_paths)


feature_names = [f"feature_{i}" for i in range(len(train_features[0]))]
train_df = pd.DataFrame(train_features, columns=feature_names)
val_df = pd.DataFrame(val_features, columns=feature_names)
test_df = pd.DataFrame(test_features, columns=feature_names)


train_stats = train_df.describe()


if len(train_features[0]) > 10:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    val_scaled = scaler.transform(val_features)

    pca = PCA(n_components=10)
    train_features = pca.fit_transform(train_scaled)
    val_features = pca.transform(val_scaled)


svm_model = SVC(kernel="linear")
svm_model.fit(train_features, train_labels)


val_predictions = svm_model.predict(val_features)
val_accuracy = accuracy_score(val_labels, val_predictions)


num_classes = len(set(labels))
image_shape = (32, 32, 3)

train_labels_categorical = to_categorical(train_labels, num_classes=num_classes)
val_labels_categorical = to_categorical(val_labels, num_classes=num_classes)

train_features_cnn = np.array(train_features).reshape(-1, *image_shape)
val_features_cnn = np.array(val_features).reshape(-1, *image_shape)


num_classes = len(set(labels))
image_shape = (32, 32, 3)

train_labels_categorical = to_categorical(train_labels, num_classes=num_classes)
val_labels_categorical = to_categorical(val_labels, num_classes=num_classes)

train_features_cnn = np.array(train_features).reshape(-1, *image_shape)
val_features_cnn = np.array(val_features).reshape(-1, *image_shape)

cnn_model = Sequential()
cnn_model.add(Conv2D(32, (3, 3), activation="relu", input_shape=image_shape))
cnn_model.add(MaxPooling2D((2, 2)))
cnn_model.add(Conv2D(64, (3, 3), activation="relu"))
cnn_model.add(MaxPooling2D((2, 2)))
cnn_model.add(Conv2D(128, (3, 3), activation="relu"))
cnn_model.add(MaxPooling2D((2, 2)))
cnn_model.add(Flatten())
cnn_model.add(Dense(128, activation="relu"))
cnn_model.add(Dense(num_classes, activation="softmax"))

cnn_model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)

cnn_model.fit(
    train_features_cnn,
    train_labels_categorical,
    epochs=10,
    batch_size=32,
    validation_data=(val_features_cnn, val_labels_categorical),
)


test_features_cnn = np.array(test_features).reshape(-1, *image_shape)
test_labels_categorical = to_categorical(test_labels, num_classes=num_classes)
test_loss, test_accuracy = cnn_model.evaluate(
    test_features_cnn, test_labels_categorical
)

print("Test Accuracy:", test_accuracy)


svm_model.save("svm_model.h5")
cnn_model.save("cnn_model.h5")


class ImageClassifier:
    def __init__(self, svm_model_path, cnn_model_path):
        self.svm_model = load_model(svm_model_path)
        self.cnn_model = load_model(cnn_model_path)

    def predict_svm(self, features):
        return self.svm_model.predict(features)

    def predict_cnn(self, features):
        features_cnn = np.array(features).reshape(-1, *image_shape)
        predictions = self.cnn_model.predict(features_cnn)
        return np.argmax(predictions, axis=1)


if val_accuracy >= 0.85:
    print("The verification accuracy meets the requirement.")


img_classifier = ImageClassifier("svm_model.h5", "cnn_model.h5")

svm_predictions = img_classifier.predict_svm(test_features)
cnn_predictions = img_classifier.predict_cnn(test_features)

svm_accuracy = accuracy_score(test_labels, svm_predictions)
cnn_accuracy = accuracy_score(test_labels, cnn_predictions)

print("SVM Model Accuracy:", svm_accuracy)
print("CNN Model Accuracy:", cnn_accuracy)
