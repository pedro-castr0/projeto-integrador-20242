import tkinter
from tkinter import *
from tkinter import messagebox
import customtkinter
from PIL import Image, ImageDraw, ImageTk
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class DrawPredict:
    def __init__(self):
        self.classifier = None
        self.class_names = ['airplane', 'dog','flower', 'house', 'snail', 'tree']
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

        # Imagens para botões
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

        self.prob_label = customtkinter.CTkLabel(self.root, text="Probabilidades aparecerão aqui", anchor="w", justify="left")
        self.prob_label.pack(pady=(0, 10))
        self.update_prediction_loop()

        self.image = Image.new("RGB", (600, 600), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.brush = 28

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def train(self, classes_para_usar=None, max_imgs_por_classe=20000):
        img_size = 784  # 28x28
        base_path = "image_base/bin/"

        if classes_para_usar is None:
            classes_para_usar = self.class_names

        all_imgs, all_labels = [], []

        for class_name in classes_para_usar:
            file_path = os.path.join(base_path, f"{class_name}.bin")
            if not os.path.exists(file_path):
                print(f"Aviso: '{class_name}' não encontrado em {file_path}")
                continue

            num_imgs = os.path.getsize(file_path) // img_size
            num_imgs = min(num_imgs, max_imgs_por_classe)

            print(f"Lendo {num_imgs} imagens de '{class_name}'")

            data = np.fromfile(file_path, dtype=np.uint8)[:num_imgs * img_size].reshape((num_imgs, img_size))
            labels = np.full(num_imgs, self.class_names.index(class_name), dtype=np.uint8)

            all_imgs.append(data)
            all_labels.append(labels)

        # Normaliza e junta os dados
        X = np.vstack(all_imgs) / 255.0
        y = np.concatenate(all_labels)

        # Divide em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Cria e treina o modelo Random Forest
        clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
        clf.fit(X_train, y_train)

        # Avaliação
        acc = accuracy_score(y_test, clf.predict(X_test))
        print(f"Acurácia: {acc * 100:.2f}%")

        # Salva o modelo
        self.classifier = clf
        with open("modelo.pkl", "wb") as f:
            pickle.dump(clf, f)

        # Mensagem visual
        messagebox.showinfo("Treinamento", "Modelo treinado e salvo com sucesso!", parent=self.root)

    def predict_proba(self):
        self.image.save("temp.png")
        img = Image.open("temp.png").convert("L").resize((28, 28), Image.LANCZOS)
        img_array = np.array(img).reshape(1, -1) / 255.0

        return self.classifier.predict_proba(img_array)[0]

    def predict(self):
        try:
            probs = self.predict_proba()
            index = np.argmax(probs)
            messagebox.showinfo("Previsão", f"Desenho reconhecido como: {self.class_names[index]}", parent=self.root)
        except Exception as e:
            messagebox.showerror("Erro na Previsão", str(e), parent=self.root)

    def update_prediction_loop(self):
        try:
            if self.classifier is not None:
                probs = self.predict_proba()
                texto = "\n".join([f"{name}: {probs[i]*100:.2f}%" for i, name in enumerate(self.class_names)])
                self.prob_label.configure(text=texto)
            else:
                self.prob_label.configure(text="Probabilidades: (modelo não carregado)")
        except Exception as e:
            self.prob_label.configure(text="Erro ao prever: " + str(e))

        self.root.after(1000, self.update_prediction_loop)

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.screen.create_oval(x1, y1, x2 + self.brush, y2 + self.brush, fill="black", outline="black")
        self.draw.ellipse([x1, y1, x2 + self.brush, y2 + self.brush], fill="black")

    def clear(self):
        self.screen.delete("all")
        self.draw.rectangle([0, 0, 600, 600], fill="white")

    def on_closing(self):
        if messagebox.askyesno("Sair?", "Deseja sair do programa?", parent=self.root):
            self.root.destroy()

    def init_game_GUI(self):
        messagebox.showinfo("Minigame", "A lógica do jogo ainda não foi implementada.", parent=self.root)

DrawPredict()