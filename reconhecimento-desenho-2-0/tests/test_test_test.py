import tkinter
from tkinter import *
from tkinter import messagebox

import customtkinter

from PIL import Image, ImageDraw, ImageTk, ImageOps
import cv2 as cv
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pickle
import os.path

class DrawPredict:
    def __init__(self):
        self.classifier = None
        self.load_model()
        self.init_train_GUI()

    def load_model(self):
        if os.path.exists("modelo.pkl"):
            with open("modelo.pkl", "rb") as f:
                self.classifier = pickle.load(f)

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

        for i in range(5):
            bt_top.columnconfigure(i, weight=1)

        customtkinter.CTkButton(bt_top, command=self.train, width=40, height=40, image=book_image,
                                text="", border_width=1, border_color="white", fg_color="#1EA1F3").grid(row=0, column=0, padx=2, pady=2)

        customtkinter.CTkButton(bt_top, command=self.clear, width=40, height=40, image=trash_image,
                                text="", border_width=1, border_color="white", fg_color="#FD3B3B").grid(row=0, column=1, padx=2, pady=2)

        customtkinter.CTkButton(bt_top, command=self.predict, width=40, height=40, image=lamp_image,
                                text="", border_width=1, border_color="white", fg_color="#FFD54F").grid(row=0, column=2, padx=2, pady=2)

        customtkinter.CTkButton(bt_top, command=self.init_game_GUI, width=40, height=40, image=play_image,
                                text="", border_width=1, border_color="white", fg_color="#01A437").grid(row=0, column=4, padx=2, pady=2)

        self.screen = Canvas(self.root, width=600, height=600, bg='white')
        self.screen.pack(expand=True, fill="both", padx=10, pady=10)
        self.screen.bind("<B1-Motion>", self.paint)

        self.image = Image.new("RGB", (600, 600), "white")
        self.draw = ImageDraw.Draw(self.image)

        self.brush = 22

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def train(self):
        img_size = 784
        base_path = "image_base/bin/"
        classes = ['airplane', 'bee', 'bicycle', 'cat', 'dog', 'flower', 'house', 'smiley face', 'snail', 'tree']

        all_imgs, all_labels = [], []

        for i, class_name in enumerate(classes):
            file_path = os.path.join(base_path, f"{class_name}.bin")
            num_imgs = os.path.getsize(file_path) // img_size

            print(f"Lendo {num_imgs} imagens de '{class_name}'")

            data = np.fromfile(file_path, dtype=np.uint8).reshape((num_imgs, img_size))
            labels = np.full(num_imgs, i, dtype=np.uint8)

            all_imgs.append(data)
            all_labels.append(labels)

        X = np.vstack(all_imgs) / 255.0
        y = np.concatenate(all_labels)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = SVC(kernel='linear')
        clf.fit(X_train, y_train)

        acc = accuracy_score(y_test, clf.predict(X_test))
        print(f"Acurácia: {acc * 100:.2f}%")

        self.classifier = clf
        with open("modelo.pkl", "wb") as f:
            pickle.dump(clf, f)

        messagebox.showinfo("Treinamento", "Modelo treinado e salvo com sucesso!", parent=self.root)

    def predict(self):
        if not self.classifier:
            messagebox.showerror("Erro", "Você precisa treinar ou carregar um modelo!", parent=self.root)
            return

        self.image.save("temp.png")

        img = Image.open("temp.png").convert("L")
        img = ImageOps.invert(img)
        img = img.resize((28, 28), Image.LANCZOS)

        img_array = np.array(img).reshape(1, 784) / 255.0

        prediction = self.classifier.predict(img_array)

        class_names = [
            "avião", "abelha", "bicicleta", "gato", "cachorro",
            "flor", "casa", "carinha feliz", "caracol", "árvore"
        ]

        nome_classe = class_names[prediction[0]]

        messagebox.showinfo("Minigame | Adivinhação de Objetos", f"Esse desenho provavelmente é um(a) {nome_classe}.", parent=self.root)

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)

        self.screen.create_rectangle(x1, y1, x2, y2, fill="black", width=self.brush)
        self.draw.rectangle([x1, y1, x2 + self.brush, y2 + self.brush], fill="black")

    def clear(self):
        self.screen.delete("all")
        self.draw.rectangle([0, 0, 600, 600], fill="white")

    def on_closing(self):
        if messagebox.askyesno("Sair?", "Deseja sair do programa?", parent=self.root):
            self.root.destroy()

    def init_game_GUI(self):
        # Coloque aqui o que você quiser iniciar para o jogo em si
        messagebox.showinfo("Minigame", "A lógica do jogo ainda não foi implementada.", parent=self.root)

DrawPredict()