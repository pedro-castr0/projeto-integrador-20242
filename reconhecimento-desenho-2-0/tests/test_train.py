import tkinter
from tkinter import *
from tkinter import messagebox

import customtkinter

from PIL import Image, ImageDraw, ImageTk
import cv2 as cv
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pickle
import os.path

# === CONFIGURAÇÕES ===
img_size = 784  # 28x28

# === LEITURA DO ARQUIVO BINÁRIO ===

cat_data = np.fromfile("image_base/bin/cat.bin", dtype=np.uint8).reshape((2000, img_size))
cat_labels = np.zeros(cat_data.shape[0], dtype=np.uint8)

dog_data = np.fromfile("image_base/bin/dog.bin", dtype=np.uint8).reshape((2000, img_size))
dog_labels = np.ones(dog_data.shape[0], dtype=np.uint8)

cat_data = np.fromfile("image_base/bin/cat.bin", dtype=np.uint8).reshape((2000, img_size))
cat_labels = np.zeros(cat_data.shape[0], dtype=np.uint8)



# === JUNÇÃO E PRÉ-PROCESSAMENTO ===
img_list = np.vstack((dog_data, cat_data))
class_list = np.concatenate((dog_labels, cat_labels))

img_list = img_list / 255.0  # Normalização

# === DIVISÃO E TREINAMENTO ===
X_train, X_test, y_train, y_test = train_test_split(img_list, class_list, test_size=0.2, random_state=42)

clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# === AVALIAÇÃO ===
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Acurácia: {acc * 100:.2f}%")