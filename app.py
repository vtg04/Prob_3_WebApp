
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

latent_size = 100
num_classes = 10
image_size = 28 * 28

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_size + num_classes, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, image_size),
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        return self.model(x)

G = Generator()
G.load_state_dict(torch.load("generator.pth", map_location=torch.device('cpu')))
G.eval()

st.title("🧠 Handwritten Digit Generator (0–9)")
digit = st.selectbox("Select a digit to generate:", list(range(10)))

if st.button("Generate"):
    st.write(f"Generating 5 samples of digit `{digit}`...")
    fig, axs = plt.subplots(1, 5, figsize=(15, 3))
    z = torch.randn(5, latent_size)
    labels = torch.tensor([digit] * 5)
    fake_images = G(z, labels).view(-1, 28, 28).detach().numpy()
    for i in range(5):
        axs[i].imshow((fake_images[i] + 1) / 2, cmap='gray')
        axs[i].axis('off')
    st.pyplot(fig)
