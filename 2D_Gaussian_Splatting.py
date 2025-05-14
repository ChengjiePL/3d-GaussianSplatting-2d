import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR
import torch.nn as nn
import torch
import gc
import os
import imageio
import yaml
from torch.optim import Adam
from datetime import datetime
from PIL import Image
import requests

# ===========================
# Crear carpeta de guardado
# ===========================
now = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
output_folder = f"output_epochs_{now}"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print(f"Las imágenes de cada epoch se guardarán en: {output_folder}")

# ===========================
# Función para generar Gaussian Splats 2D
# ===========================
def generate_2D_gaussian_splatting(kernel_size, scale, rotation, coords, colours, image_size=(256, 256, 3), device="cpu"):
    batch_size = colours.shape[0]

    # Configuración de rotación y escala
    scale = scale.view(batch_size, 2)
    rotation = rotation.view(batch_size)

    # Matrices de transformación
    cos_rot = torch.cos(rotation)
    sin_rot = torch.sin(rotation)

    R = torch.stack([
        torch.stack([cos_rot, -sin_rot], dim=-1),
        torch.stack([sin_rot, cos_rot], dim=-1)
    ], dim=-2)

    S = torch.diag_embed(scale)
    covariance = R @ S @ S @ R.transpose(-1, -2)
    inv_covariance = torch.inverse(covariance)

    # Crear el kernel
    x = torch.linspace(-5, 5, kernel_size, device=device)
    y = torch.linspace(-5, 5, kernel_size, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    xy = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(batch_size, -1, -1, -1)

    z = torch.einsum('bxyi,bij,bxyj->bxy', xy, -0.5 * inv_covariance, xy)
    kernel = torch.exp(z) / (2 * torch.tensor(np.pi, device=device) * torch.sqrt(torch.det(covariance))).view(batch_size, 1, 1)

    # Normalización y expansión para RGB
    kernel_max = kernel.amax(dim=(-2, -1), keepdim=True)
    kernel_normalized = kernel / kernel_max
    kernel_rgb = kernel_normalized.unsqueeze(1).expand(-1, 3, -1, -1)

    # Padding y transformación afín
    pad_h = image_size[0] - kernel_size
    pad_w = image_size[1] - kernel_size
    padding = (pad_w // 2, pad_w // 2 + pad_w % 2, pad_h // 2, pad_h // 2 + pad_h % 2)
    kernel_rgb_padded = F.pad(kernel_rgb, padding, "constant", 0)

    # Aplicación de la transformación afín
    b, c, h, w = kernel_rgb_padded.shape
    theta = torch.zeros(b, 2, 3, dtype=torch.float32, device=device)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, :, 2] = coords

    grid = F.affine_grid(theta, size=(b, c, h, w), align_corners=True)
    kernel_rgb_padded_translated = F.grid_sample(kernel_rgb_padded, grid, align_corners=True)

    # Aplicación de colores y sumatoria final
    rgb_values_reshaped = colours.unsqueeze(-1).unsqueeze(-1)
    final_image_layers = rgb_values_reshaped * kernel_rgb_padded_translated
    final_image = final_image_layers.sum(dim=0)
    final_image = torch.clamp(final_image, 0, 1)
    final_image = final_image.permute(1, 2, 0)

    return final_image

# ===========================
# Cargar imagen objetivo
# ===========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
original_image = Image.open("a.jpeg")
image_size = (256, 256, 3)
original_image = original_image.resize((image_size[0], image_size[1]))
original_image = original_image.convert('RGB')
original_array = np.array(original_image) / 255.0
target_tensor = torch.tensor(original_array, dtype=torch.float32, device=device)

# ===========================
# Parámetros del entrenamiento
# ===========================
num_epochs = 500
KERNEL_SIZE = 101
learning_rate = 0.01
primary_samples = 1000
backup_samples = 1000
num_samples = primary_samples + backup_samples
W_values = torch.randn(num_samples, 9, device=device)  # Valores iniciales
W = nn.Parameter(W_values)

# Optimización
optimizer = Adam([W], lr=learning_rate)
scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=1e-8, total_iters=num_epochs)
loss_history = []

# ===========================
# Bucle de entrenamiento
# ===========================
for epoch in range(num_epochs):
    gc.collect()
    torch.cuda.empty_cache()

    # Extraer parámetros
    output = W
    batch_size = output.shape[0]
    scale = torch.sigmoid(output[:, 0:2])
    rotation = np.pi / 2 * torch.tanh(output[:, 2])
    alpha = torch.sigmoid(output[:, 3])
    colours = torch.sigmoid(output[:, 4:7])
    pixel_coords = torch.tanh(output[:, 7:9])

    # Generar imagen
    colours_with_alpha = colours * alpha.view(batch_size, 1)
    g_tensor_batch = generate_2D_gaussian_splatting(KERNEL_SIZE, scale, rotation, pixel_coords, colours_with_alpha, image_size, device=device)
    loss = nn.functional.mse_loss(g_tensor_batch, target_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    loss_history.append(loss.item())

    # ===========================
    # Guardado de cada época
    # ===========================
    generated_array = g_tensor_batch.cpu().detach().numpy()
    img = Image.fromarray((generated_array * 255).astype(np.uint8))
    filename = f"epoch_{epoch:04d}.png"
    file_path = os.path.join(output_folder, filename)
    img.save(file_path)
    print(f"[Epoch {epoch+1}/{num_epochs}] Imagen guardada en: {file_path}")

print(f"\nTodas las imágenes se guardaron en: {output_folder}")
