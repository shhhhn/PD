import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Streamlit page configuration
st.set_page_config(page_title="Scalpel Classification System", layout="wide")

st.write(
    "<div style='text-align: center; font-size: 50px;'>Welcome to the Scalpel Classification System</div>",
    unsafe_allow_html=True,
)

# Function to preprocess the uploaded image
def load_image(image_file):
    """Preprocess the uploaded image to make it compatible with the model."""
    img = Image.open(image_file)
    img = img.resize((255, 255))  # Resize to fit 100x100 dimensions (adjust as needed)
    img = np.array(img)
    if img.shape[-1] == 4:  # Handle transparency
        img = img[..., :3]
    img = img.flatten()  # Flatten into a 1D array
    img = img[:10000]  # Ensure the array matches 10000 elements
    img = img.astype('float32')
    img /= 255.0  # Normalize pixel values
    img = img.reshape(1, -1)  # Match model input shape (1, 10000)
    return img

# Function to predict the class of the uploaded image
def predict(image, model, labels):
    """Predict the class of the uploaded image."""
    img = load_image(image)
    try:
        result = model.predict(img)
        predicted_class = np.argmax(result, axis=1)
        return labels[predicted_class[0]]
    except Exception as e:
        raise ValueError(f"Error during prediction: {e}\nInput shape: {img.shape}")

# Load the trained model
try:
    model = load_model('1mParamsModel (1).h5')  # Update with your model's filename
except Exception as e:
    st.error(f"Error loading the model: {e}")

# Load class labels
def load_labels(filename):
    try:
        with open(filename, 'r') as file:
            labels = file.readlines()
        labels = [label.strip() for label in labels]
        return labels
    except FileNotFoundError:
        st.error(f"Labels file '{filename}' not found.")
        return []

# Page title and file upload functionality
st.title("Scalpel Object Classification")

test_image = st.file_uploader("Upload an Image of the Object:", type=["jpg", "jpeg", "png"])
if test_image is not None:
    try:
        st.image(test_image, width=300, caption="Uploaded Image")
        labels = load_labels("labels.txt")  # Update with your labels filename

        if st.button("Classify"):
            st.write("Classifying the object...")
            if labels:
                predicted_category = predict(test_image, model, labels)
                st.success(f"Predicted Category: {predicted_category}")
            else:
                st.error("Labels not found. Please upload the label file.")
    except Exception as e:
        st.error(f"Error processing the uploaded image: {e}")
