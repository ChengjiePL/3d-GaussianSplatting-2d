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
num_gaussians = 3000
sigma_thre = 10
grad_thre = 0.06
output_folder = "output_iterations_gs2d"
num_epochs = 500  
num_iter_per_epoch = 100

# ===========================
# Crear carpeta de guardado
# ===========================
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ===========================
# Inicialización aleatoria mejorada (GS2D Style)
# ===========================
def random_init_param():
    sigma = torch.rand(size=(num_gaussians, 2)).to(device) - 3  # Tamaños
    rho = torch.rand(size=(num_gaussians, 1)).to(device) * 2    # Rotación
    mean = torch.atanh(torch.rand(size=(num_gaussians, 2)).to(device) * 2 - 1)  # Posición
    color = torch.atanh(torch.rand(size=(num_gaussians, 3)).to(device))         # Color
    alpha = torch.zeros(size=(num_gaussians, 1)).to(device) - 0.01              # Transparencia
    w = torch.cat([sigma, rho, mean, color, alpha], dim=1)
    return nn.Parameter(w)

# ===========================
# Meshgrid para la imagen
# ===========================
x_range = torch.linspace(0, img_size[0] - 1, img_size[0], device=device)
y_range = torch.linspace(0, img_size[1] - 1, img_size[1], device=device)
x_grid, y_grid = torch.meshgrid(x_range, y_range, indexing='xy')

# ===========================
# Función para renderizar las gaussianas con rotación
# ===========================
def draw_gaussian(sigma, rho, mean, color, alpha):
    """
    Renderiza las gaussianas con elipticidad y rotación
    """
    r = rho.view(-1, 1, 1)
    sx = sigma[:, :1, None]
    sy = sigma[:, 1:, None]
    dx = x_grid.unsqueeze(0) - mean[:, 0].view(-1, 1, 1)
    dy = y_grid.unsqueeze(0) - mean[:, 1].view(-1, 1, 1)

    # Fórmula extendida con rotación
    v = -0.5 * (((sy * dx) ** 2 + (sx * dy) ** 2) - 2 * dx * dy * r * sx * sy) / (sx**2 * sy**2 * (1 - r**2) + 1e-8)
    v = torch.exp(v)
    v = v * alpha.view(-1, 1, 1)
    img = torch.sum(v.unsqueeze(1) * color.view(-1, 3, 1, 1), dim=0)
    return torch.clamp(img, 0, 1)

# ===========================
# Función para extraer parámetros del tensor
# ===========================
def parse_param(w):
    sigma = (torch.sigmoid(w[:, 0:2])) * torch.tensor(img_size[::-1], device=device) * 0.25
    rho = torch.tanh(w[:, 2:3])
    mean = (0.5 * torch.tanh(w[:, 3:5]) + 0.5) * torch.tensor(img_size[::-1], device=device)
    color = 0.5 * torch.tanh(w[:, 5:8]) + 0.5
    alpha = 0.5 * torch.tanh(w[:, 8:9]) + 0.5
    return sigma, rho, mean, color, alpha

# ===========================
# Optimización
# ===========================
w = random_init_param()
optimizer = torch.optim.AdamW([w], lr=0.005)

plt.ion()
fig, ax = plt.subplots()

for epoch in tqdm(range(num_epochs)):
    optimizer.zero_grad()
    sigma, rho, mean, color, alpha = parse_param(w)
    img = draw_gaussian(sigma, rho, mean, color, alpha)
    
    loss = nn.functional.l1_loss(img, target_image)
    loss.backward()
    optimizer.step()
    
    # Guardado de la imagen
    save_path = f"{output_folder}/iter_{epoch:04d}.png"
    torchvision.utils.save_image(img, save_path)

    if epoch % 50 == 0 or epoch == num_epochs - 1:
        ax.clear()
        ax.imshow(img.permute(1, 2, 0).cpu().detach().numpy())
        ax.set_title(f"Iteración {epoch} - Pérdida: {loss.item():.6f}")
        plt.pause(0.001)

torchvision.utils.save_image(img, "resultado_final.png")
print(f"Imágenes guardadas en la carpeta '{output_folder}'")
