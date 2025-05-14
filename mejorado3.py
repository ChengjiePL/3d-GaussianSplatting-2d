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
sigma_min, sigma_max = 2.0, 8.0
output_folder = "output_iterations_optimized"
decay_factor = 0.98
min_sigma = 0.01
num_epochs = 500  # Número de iteraciones reducido

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
# Proyección en 2D (Optimizado con Broadcasting)
# ===========================
def project_gaussians(positions, colors, sizes, epoch):
    """
    Renderiza todas las gaussianas en paralelo en un solo paso.
    """
    img = torch.zeros(3, img_size[0], img_size[1], device=device)

    # ===========================
    # Broadcasting para proyectar todas las gaussianas en un solo paso
    # ===========================
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    valid = z > 0
    
    if valid.sum() == 0:
        return img

    # Solo las válidas (las que están al frente de la cámara)
    x, y, z = x[valid], y[valid], z[valid]
    screen_x = (x / z)
    screen_y = (y / z)

    # ===========================
    # Decaimiento progresivo del tamaño
    # ===========================
    current_decay = decay_factor ** epoch
    sigma = torch.clamp(sizes[valid] / z * current_decay, min_sigma, sigma_max / 100)

    # ===========================
    # Broadcasting para generar todas las gaussianas en un solo paso
    # ===========================
    # Expando dimensiones para permitir operaciones matriciales
    screen_x = screen_x.view(-1, 1, 1)
    screen_y = screen_y.view(-1, 1, 1)
    sigma = sigma.view(-1, 1, 1)
    
    # ===========================
    # Generar el mapa gaussiano
    # ===========================
    gaussians = torch.exp(-((x_grid - screen_x)**2 + (y_grid - screen_y)**2) / (2 * sigma**2))
    colored_gaussians = gaussians.unsqueeze(1) * colors[valid].view(-1, 3, 1, 1)

    # ===========================
    # Sumatoria en paralelo
    # ===========================
    img = colored_gaussians.sum(dim=0)
    
    return torch.clamp(img, 0, 1)

# ===========================
# Optimización
# ===========================
optimizer = torch.optim.Adam([positions, colors, sizes], lr=0.005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.8)

gradient_clip_value = 1.0

plt.ion()
fig, ax = plt.subplots()

for epoch in tqdm(range(num_epochs)):
    optimizer.zero_grad()
    img = project_gaussians(positions, colors, sizes, epoch)
    loss = nn.functional.mse_loss(img, target_image)
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_([positions, colors, sizes], gradient_clip_value)
    optimizer.step()
    scheduler.step()
    
    # Guardar cada iteración
    save_path = f"{output_folder}/iter_{epoch:04d}.png"
    torchvision.utils.save_image(img, save_path)

    # Visualización en vivo sin bloquear
    if epoch % 50 == 0 or epoch == num_epochs - 1:
        ax.clear()
        ax.imshow(img.permute(1, 2, 0).cpu().detach().numpy())
        ax.set_title(f"Iteración {epoch} - Pérdida: {loss.item():.6f}")
        plt.pause(0.001)

torchvision.utils.save_image(img, "resultado_final.png")
print(f"Imágenes guardadas en la carpeta '{output_folder}'")
