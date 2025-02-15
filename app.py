import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import re

# Import custom layers
from custom_layers import TransformerEncoderLayer, TransformerDecoderLayer, Embeddings, CNN_Encoder

# Load the trained model
caption_model = tf.keras.models.load_model('D:/Projects/CaptionFlow(Kaggle)/model.h5', custom_objects={
    'TransformerEncoderLayer': TransformerEncoderLayer,
    'TransformerDecoderLayer': TransformerDecoderLayer,
    'Embeddings': Embeddings,
    'CNN_Encoder': CNN_Encoder
})

def preprocess_image(image):
    image = image.resize((224, 224))  # Adjust size as needed
    image = np.array(image) / 255.0  # Normalize image
    return np.expand_dims(image, axis=0)

def generate_caption(image):
    preprocessed_image = preprocess_image(image)
    caption = caption_model.predict(preprocessed_image)
    return caption

# Streamlit app layout
st.title('Image Caption Generator')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Generate caption
    caption = generate_caption(image)
    st.write(f"Caption: {caption}")

