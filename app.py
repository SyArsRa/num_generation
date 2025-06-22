import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Dummy generator function (replace with your real one)
def generate_number_picture(number):
    """
    Generate a single MNIST-like digit image.
    Replace this with your GAN generator output.
    """
    img = np.random.rand(28, 28) * (number / 9)  # Just a dummy image
    return img

# Streamlit UI
st.title("Digit Generator (0-9)")

number = st.number_input("Enter a digit (0-9)", min_value=0, max_value=9, step=1)
generate = st.button("Generate")

if generate:
    st.subheader(f"Generated images for '{number}'")
    cols = st.columns(5)

    for i in range(5):
        img = generate_number_picture(number)

        # Plot image to buffer
        buf = BytesIO()
        plt.imsave(buf, img, cmap='gray', format='png')
        cols[i].image(buf.getvalue(), width=100, caption=f"Sample {i+1}")
