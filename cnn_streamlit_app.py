import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('models/cnn_model_with_dropout.h5')

st.title('MNIST Digit Classifier')

st.write("Upload an image of a handwritten digit (28x28 pixels)")

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    # Convert the file to an image
    image = Image.open(uploaded_file).convert('L')
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Preprocess the image
    image = image.resize((28, 28))
    image = img_to_array(image)
    image = image.reshape(1, 28, 28, 1)
    image = image / 255.0
    
    # Make a prediction
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    st.write(f'Predicted Class: {predicted_class}')