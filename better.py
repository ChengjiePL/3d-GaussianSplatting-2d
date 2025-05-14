import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import os

# ===========================
# Configuración inicial
# ===========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# Imagen objetivo
target_image = Image.open("a.jpeg").convert("RGB")
target_image = torchvision.transforms.Resize((256, 256))(target_image)
target_image = torchvision.transforms.ToTensor()(target_image).to(device)

# Parámetros
img_size = (256, 256)
num_gaussians = 2000
sigma_min, sigma_max = 2.0, 8.0  # Tamaño mínimo y máximo de las gaussianas
output_folder = "output_iterations"

# ===========================
# Crear carpeta de guardado
# ===========================
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ===========================
# Inicialización aleatoria mejorada
# ===========================
positions = torch.rand(num_gaussians, 3, device=device) * 2 - 1
positions[:, 2] += 2.0  
colors = torch.rand(num_gaussians, 3, device=device)
sizes = torch.rand(num_gaussians, device=device) * 0.1 + 0.05
positions.requires_grad = True
colors.requires_grad = True
sizes.requires_grad = True

# ===========================
# Crear coordenadas de la imagen
# ===========================
x_range = torch.linspace(-1, 1, img_size[0], device=device)
y_range = torch.linspace(-1, 1, img_size[1], device=device)
x_grid, y_grid = torch.meshgrid(x_range, y_range, indexing='ij')

# ===========================
# Proyección en 2D (Renderizado Gaussiano)
# ===========================
def project_gaussians(positions, colors, sizes):
    """
    Renderiza las gaussianas en un plano 2D
    """
    img = torch.zeros(3, img_size[0], img_size[1], device=device)

    for i in range(num_gaussians):
        x, y, z = positions[i]
        
        if z > 0:  # Solo renderizamos si está frente a la cámara
            screen_x = (x / z)
            screen_y = (y / z)
            
            # Calcula el tamaño proyectado
            sigma = torch.clamp(sizes[i] / z, sigma_min / 100, sigma_max / 100)

            # Generar el mapa gaussiano
            gaussian = torch.exp(-((x_grid - screen_x)**2 + (y_grid - screen_y)**2) / (2 * sigma**2))
            
            # Mezclar el color y añadir al buffer de imagen
            img += colors[i].view(3, 1, 1) * gaussian

    return torch.clamp(img, 0, 1)

# ===========================
# Optimización (ajuste de posición, color y tamaño)
# ===========================
optimizer = torch.optim.Adam([positions, colors, sizes], lr=0.005)
num_epochs = 500
gradient_clip_value = 1.0  # Valor para evitar saltos grandes

for epoch in tqdm(range(num_epochs)):
    optimizer.zero_grad()
    img = project_gaussians(positions, colors, sizes)
    loss = nn.functional.mse_loss(img, target_image)
    loss.backward()
    
    # Gradient Clipping
    torch.nn.utils.clip_grad_norm_([positions, colors, sizes], gradient_clip_value)
    optimizer.step()
    
    # Guardar cada iteración
    save_path = f"{output_folder}/iter_{epoch:04d}.png"
    torchvision.utils.save_image(img, save_path)

    if epoch % 50 == 0 or epoch == num_epochs - 1:
        print(f"Epoch {epoch} - Pérdida: {loss.item()}")
        plt.imshow(img.permute(1, 2, 0).cpu().detach().numpy())
        plt.title(f"Iteración {epoch}")
        plt.show()

# Guardar resultado final
torchvision.utils.save_image(img, "resultado_final.png")
print(f"Imágenes guardadas en la carpeta '{output_folder}'")
