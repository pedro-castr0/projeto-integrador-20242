import os
import numpy as np
from PIL import Image, ImageDraw, ImageTk

import tkinter
from tkinter import Canvas, messagebox

import customtkinter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class DrawPredict:
    def __init__(self):
        self.class_names = ['Avião', 'Formiga', 'Maçã', 'Machado', 'Bola de basquete', 'Abelha', 'Bicicleta', 'Pássaro', 'Gato', 'Círculo', 'Relógio', 'Cachorro', 'Rosquinha', 'Torre Eiffel', 'Flor', 'Casa', 'Rosto sorridente', 'Caracol', 'Quadrado', 'Árvore']
        self.classifier = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
        self.init_train_GUI()

    def load_model(self):
        if os.path.exists("modelo_pytorch.pt"):
            model = Net(num_classes=len(self.class_names)).to(self.device)
            model.load_state_dict(torch.load("modelo_pytorch.pt", map_location=self.device))
            model.eval()
            self.classifier = model

    def init_train_GUI(self):
        customtkinter.set_appearance_mode("dark")
        customtkinter.set_default_color_theme("blue")

        self.root = customtkinter.CTk()
        self.root.title("Treinamento | Adivinhação de Objetos")

        bt_top = customtkinter.CTkFrame(self.root)
        bt_top.pack(side="top", fill="x", padx=2, pady=2)

        for i in range(4):
            bt_top.columnconfigure(i, weight=1)

        def img(path): return ImageTk.PhotoImage(Image.open(path))

        customtkinter.CTkButton(bt_top, command=self.train, width=40, height=40, image=img("images/Group 10.png"),
                                text="", border_width=1, border_color="white", fg_color="#1EA1F3").grid(row=0, column=0, padx=2, pady=2)
        
        customtkinter.CTkButton(bt_top, command=self.clear, width=40, height=40, image=img("images/Group 13.png"),
                                text="", border_width=1, border_color="white", fg_color="#FD3B3B").grid(row=0, column=1, padx=2, pady=2)
        
        customtkinter.CTkButton(bt_top, command=self.predict, width=40, height=40, image=img("images/Group 12.png"),
                                text="", border_width=1, border_color="white", fg_color="#FFD54F").grid(row=0, column=2, padx=2, pady=2)
        
        customtkinter.CTkButton(bt_top, command=self.init_game_GUI, width=40, height=40, image=img("images/Rectangle 15.png"),
                                text="", border_width=1, border_color="white", fg_color="#01A437").grid(row=0, column=4, padx=2, pady=2)

        self.screen = Canvas(self.root, width=600, height=600, bg='black')
        self.screen.pack(expand=True, fill="both", padx=10, pady=10)
        self.screen.bind("<B1-Motion>", self.paint)

        self.prob_frame = customtkinter.CTkFrame(self.root)
        self.prob_frame.pack(pady=(0, 10), padx=10, fill="both")

        self.prob_labels = []

        cols = 4
        self.columns_data = [[] for _ in range(cols)]
        for i, name in enumerate(self.class_names):
            self.columns_data[i % cols].append(name)

        for col in range(cols):
            col_frame = customtkinter.CTkFrame(self.prob_frame)
            col_frame.grid(row=0, column=col, sticky="nw")
            labels = []
            for name in self.columns_data[col]:
                lbl = customtkinter.CTkLabel(col_frame, text=f"{name}: 0.00%", anchor="w", justify="left")
                lbl.pack(anchor="w")
                labels.append(lbl)
            self.prob_labels.append(labels)
        self.update_prediction_loop()

        self.image = Image.new("RGB", (600, 600), "black")
        self.draw = ImageDraw.Draw(self.image)
        self.brush = 28

        preditc_bottom = customtkinter.CTkFrame(self.root)
        preditc_bottom.pack(side="top", fill="x", padx=2, pady=2)

        for i in range(4):
            bt_top.columnconfigure(i, weight=1)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def train(self, classes_para_usar=None, max_imgs_por_classe=100000):
        img_size = 784
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

            if num_imgs == 0:
                print(f"Nenhuma imagem válida para '{class_name}'")
                continue

            data = np.fromfile(file_path, dtype=np.uint8)[:num_imgs * img_size].reshape((num_imgs, img_size))
            labels = np.full(num_imgs, self.class_names.index(class_name), dtype=np.uint8)

            all_imgs.append(data)
            all_labels.append(labels)

        if not all_imgs:
            messagebox.showerror("Erro", "Nenhuma imagem foi carregada para treinamento.", parent=self.root)
            return

        X = np.vstack(all_imgs).astype(np.float32) / 255.0
        y = np.concatenate(all_labels).astype(np.int64)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=64, shuffle=True)
        test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=64)

        model = Net(num_classes=len(self.class_names)).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(20):
            model.train()
            total_loss = 0

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        model.eval()
        correct = total = 0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)

        acc = (correct / total) * 100
        print(f"Acurácia: {acc:.2f}%")

        torch.save(model.cpu().state_dict(), "modelo_pytorch.pt")
        self.classifier = model.cpu()

        messagebox.showinfo("Treinamento", "Modelo treinado e salvo com sucesso!", parent=self.root)

        model.eval()
        correct = total = 0

        with torch.no_grad():

            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)

        print(f"Acurácia: {(correct / total) * 100:.2f}%")

        torch.save(model.cpu().state_dict(), "modelo_pytorch.pt")

        self.classifier = model.cpu()

        messagebox.showinfo("Treinamento", "Modelo treinado e salvo com sucesso!", parent=self.root)

    def predict_proba(self):
        self.image.save("temp.png")

        img = Image.open("temp.png").convert("L").resize((28, 28), Image.LANCZOS)
        img_array = np.array(img).reshape(1, 784).astype(np.float32) / 255.0

        if np.all(img_array == 0):
            return np.zeros(len(self.class_names))
        
        device = next(self.classifier.parameters()).device
        input_tensor = torch.tensor(img_array).to(device)

        with torch.no_grad():
            self.classifier.eval()
            logits = self.classifier(input_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()

        return probs

    def predict(self):
        try:
            probs = self.predict_proba()
            
            if np.all(probs == 0):
                messagebox.showinfo("Previsão", "Nada foi desenhado na imagem.", parent=self.root)
                return

            index = np.argmax(probs)
            messagebox.showinfo("Previsão", f"Desenho reconhecido como: {self.class_names[index]}", parent=self.root)

        except Exception as e:
            messagebox.showerror("Erro na Previsão", str(e), parent=self.root)

    def update_prediction_loop(self):
        try:
            if self.classifier:
                probs = self.predict_proba()

                for col in range(4):
                    for i, name in enumerate(self.columns_data[col]):
                        class_index = self.class_names.index(name)
                        prob = probs[class_index] * 100
                        self.prob_labels[col][i].configure(text=f"{name}: {prob:.2f}%")
        except Exception as e:
            for col_labels in self.prob_labels:
                for label in col_labels:
                    label.configure(text=f"Erro na previsão: {str(e)}")

        self.root.after(1000, self.update_prediction_loop)

    def paint(self, event):
        x1, y1 = event.x - 1, event.y - 1
        x2, y2 = event.x + 1 + self.brush, event.y + 1 + self.brush
        self.screen.create_oval(x1, y1, x2, y2, fill="white", outline="white")
        self.draw.ellipse([x1, y1, x2, y2], fill="white")

    def clear(self):
        self.screen.delete("all")
        self.draw.rectangle([0, 0, 600, 600], fill="black")

    def on_closing(self):
        if messagebox.askyesno("Sair?", "Deseja sair do programa?", parent=self.root):
            self.root.destroy()

    def init_game_GUI(self):
        messagebox.showinfo("Minigame", "A lógica do jogo ainda não foi implementada.", parent=self.root)

DrawPredict()