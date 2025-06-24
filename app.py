import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import torch
from torchvision.transforms import ToPILImage

# Reuse these values from your training script
latent_dim = 100
num_classes = 10
embedding_dim = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
to_pil = ToPILImage()

# Generator definition (same as training)
class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = torch.nn.Embedding(num_classes, embedding_dim)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + embedding_dim, 256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(256, 512),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(512, 784),
            torch.nn.Tanh()
        )

    def forward(self, z, labels):
        label_input = self.label_emb(labels)
        x = torch.cat([z, label_input], dim=1)
        out = self.model(x)
        return out.view(-1, 1, 28, 28)

# Load generator once
G = Generator().to(device)
G.load_state_dict(torch.load("generator.pth", map_location=device))
G.eval()

def generate_number_picture(number):
    """
    Generate 5 images of the specified digit (0-9) using the trained Generator.
    Returns: list of 28x28 numpy arrays
    """
    number = int(number)
    z = torch.randn(5, latent_dim).to(device)
    labels = torch.full((5,), number, dtype=torch.long).to(device)

    with torch.no_grad():
        generated_images = G(z, labels).cpu().numpy()

    # Normalize from [-1, 1] to [0, 1]
    generated_images = (generated_images + 1) / 2.0

    return [img.squeeze() for img in generated_images]

# Streamlit UI
st.title("Digit Generator (0-9)")

number = st.number_input("Enter a digit (0-9)", min_value=0, max_value=9, step=1)
generate = st.button("Generate Image")

images = None
if generate:
    st.subheader(f"Generated images for '{number}'")
    cols = st.columns(5)
    images = generate_number_picture(number)

if images:
    # Convert numpy arrays to PIL images for display
    pil_imgs = [ to_pil(img) for img in images ]

    st.image(
            pil_imgs,
            caption=[str(number)] * 5,
            width=56
        )