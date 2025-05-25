import numpy as np
import cv2  # ou use PIL
import matplotlib.pyplot as plt

# === CONFIGURAÇÕES ===
img_index = 9120  # por exemplo, queremos a 100ª imagem
img_size = 28
img_area = img_size * img_size

with open("image_base/bin/bicycle.bin", "rb") as f:
    f.seek(img_index * img_area)  # pula direto para a posição desejada
    img_bytes = f.read(img_area)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8).reshape((img_size, img_size))

# === EXIBIR (opcional) ===
plt.imshow(img_array, cmap='gray')
plt.title(f"Imagem {img_index}")
plt.show()

# === SALVAR COMO PNG (opcional) ===
cv2.imwrite(f"dog_{img_index}.png", img_array)