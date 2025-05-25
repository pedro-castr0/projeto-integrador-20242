import tkinter
from tkinter import *
from tkinter import messagebox

import customtkinter

import PIL
from PIL import Image, ImageDraw, ImageTk
import cv2 as cv
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pickle
import os.path

class DrawPredict:
    def __init__(self):
        self.init_train_GUI()

    def init_train_GUI(self):
        customtkinter.set_appearance_mode("dark")
        customtkinter.set_default_color_theme("blue")

        self.root = customtkinter.CTk()
        self.root.title("Treinamento | Adivinhação de Objetos")

        book_image = ImageTk.PhotoImage(Image.open("images/Group 10.png"))
        lamp_image = ImageTk.PhotoImage(Image.open("images/Group 12.png"))
        trash_image = ImageTk.PhotoImage(Image.open("images/Group 13.png"))
        play_image = ImageTk.PhotoImage(Image.open("images/Rectangle 15.png"))

        bt_top = customtkinter.CTkFrame(self.root)
        bt_top.pack(side="top", fill="x", padx=2, pady=2)

        bt_top.columnconfigure(0, weight=1)
        bt_top.columnconfigure(1, weight=1)
        bt_top.columnconfigure(2, weight=1)
        bt_top.columnconfigure(3, weight=40)
        bt_top.columnconfigure(4, weight=1)

        bt_train = customtkinter.CTkButton(bt_top, 
                                           command=self.train, 
                                           corner_radius=4, 
                                           width=40, 
                                           height=40, 
                                           image=book_image, 
                                           text="", 
                                           border_width=1, 
                                           border_color="white", 
                                           fg_color="#1EA1F3")
        bt_train.grid(row=0, column=0, padx=2, pady=2)

        bt_predict = customtkinter.CTkButton(bt_top, 
                                             command=self.predict, 
                                             corner_radius=4, 
                                             width=40, 
                                             height=40, 
                                             image=lamp_image, 
                                             text="", 
                                             border_width=1, 
                                             border_color="white", 
                                             fg_color="#FFD54F")
        bt_predict.grid(row=0, column=2, padx=2, pady=2)

        bt_start = customtkinter.CTkButton(bt_top, 
                                           command=self.init_game_GUI, 
                                           corner_radius=4, 
                                           width=40, 
                                           height=40, 
                                           image=play_image, 
                                           text="", 
                                           border_width=1, 
                                           border_color="white", 
                                           fg_color="#01A437")
        bt_start.grid(row=0, column=4, padx=2, pady=2)

        bt_clear = customtkinter.CTkButton(bt_top, 
                                           command=self.clear, 
                                           corner_radius=4, 
                                           width=40, 
                                           height=40, 
                                           image=trash_image, 
                                           text="", 
                                           border_width=1, 
                                           border_color="white", 
                                           fg_color="#FD3B3B")
        bt_clear.grid(row=0, column=1, padx=2, pady=2)

        self.screen = Canvas(self.root, width=600, height=600, bg='white')
        self.screen.pack(expand=True, fill="both")
        self.screen.pack(padx=10, pady=10)
        self.screen.bind("<B1-Motion>", self.paint)

        self.image = Image.new("RGB", (500, 500), "white")
        self.draw = ImageDraw.Draw(self.image)


    def train(self):
        # === CONFIGURAÇÕES ===
        img_size = 784  # 28x28
        base_path = "image_base/bin/"
        classes = ['airplane', 'bee', 'bicycle', 'cat', 'dog', 'flower', 'house', 'smiley face', 'snail', 'tree']

        # === LEITURA DOS DADOS ===
        all_imgs = []
        all_labels = []

        for i, class_name in enumerate(classes):
            file_path = os.path.join(base_path, f"{class_name}.bin")
            
            # Verificar tamanho para deduzir número de imagens
            num_imgs = os.path.getsize(file_path) // img_size
            
            print(f"Lendo {num_imgs} imagens de '{class_name}'")
            
            data = np.fromfile(file_path, dtype=np.uint8).reshape((num_imgs, img_size))
            labels = np.full(num_imgs, i, dtype=np.uint8)  # rótulo i para essa classe
            
            all_imgs.append(data)
            all_labels.append(labels)

        # === UNIR TUDO ===
        X = np.vstack(all_imgs)
        y = np.concatenate(all_labels)

        # === NORMALIZAR ===
        X = X / 255.0

        # === DIVISÃO E TREINAMENTO ===
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = SVC(kernel='linear')
        clf.fit(X_train, y_train)

        # === AVALIAÇÃO ===
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"Acurácia: {acc * 100:.2f}%")

    def predict(self):
        # Salva o desenho feito pelo usuário
        self.image.save("temp.png")

        # Abre e redimensiona a imagem para 28x28
        img = PIL.Image.open("temp.png").convert("L")  # Converte para escala de cinza
        img = img.resize((28, 28), PIL.Image.LANCZOS)
        img_array = np.array(img).reshape(1, 784) / 255.0  # Normaliza como no treinamento

        # Faz a predição com o classificador treinado
        prediction = self.classifier.predict(img_array)

        # Mapeia o número da classe para o nome correspondente
        class_names = [
            "avião",        # 0 - airplane
            "abelha",       # 1 - bee
            "bicicleta",    # 2 - bicycle
            "gato",         # 3 - cat
            "cachorro",     # 4 - dog
            "flor",         # 5 - flower
            "casa",         # 6 - house
            "carinha feliz",# 7 - smiley face
            "caracol",      # 8 - snail
            "árvore"        # 9 - tree
        ]

        nome_classe = class_names[prediction[0]]

        tkinter.messagebox.showinfo(
            "Minigame | Adivinhação de Objetos",
            f"Esse desenho provavelmente é um(a) {nome_classe}.",
            parent=self.root
        )

    def paint(self, event):
        x1 = (event.x - 1)
        y1 = (event.y - 1)
        x2 = (event.x + 1)
        y2 = (event.y + 1)

        self.screen.create_rectangle(x1, y1, x2, y2, fill="black", width=self.brush)
        self.draw.rectangle([x1, y2, x2 + self.brush, y2 + self.brush], fill="black", width=self.brush)
    
    def clear(self):
        self.screen.delete("all")
        self.draw.rectangle([0, 0, 1000, 1000], fill="white")

    def on_closing(self):
        if tkinter.messagebox.askyesno("Sair?", "Deseja sair do programa?", parent=self.root):
            self.root.destroy()
