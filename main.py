import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from tensorflow.keras.models import load_model

# Streamlit page configuration
st.set_page_config(page_title="Scalpel Object Detection System", layout="wide")

st.write(
    "<div style='text-align: center; font-size: 50px;'>Welcome to the Scalpel Object Detection System</div>",
    unsafe_allow_html=True,
)

# Function to preprocess the uploaded image
def load_image(image_file):
    """Preprocess the uploaded image for the model."""
    img = Image.open(image_file)
    img = img.resize((300, 300))  # Resize to 300x300 (adjust based on model input)
    img_array = np.array(img)
    if img_array.shape[-1] == 4:  # Handle transparency
        img_array = img_array[..., :3]
    img_array = img_array.astype('float32')
    img_array /= 255.0  # Normalize pixel values
    img_array = img_array.reshape(1, 300, 300, 3)  # Match model input shape
    return img, img_array

# Function to predict objects in the uploaded image
def detect_objects(image, model, labels):
    """Detect objects in the uploaded image."""
    original_img, processed_img = load_image(image)
    try:
        predictions = model.predict(processed_img)[0]  # Assume model outputs bounding boxes and class probabilities
        boxes = predictions[:, :4]  # First four values: bounding box (x_min, y_min, x_max, y_max)
        scores = predictions[:, 4]  # Fifth value: confidence score
        classes = predictions[:, 5].astype(int)  # Remaining values: class indices

        # Filter results based on confidence threshold
        confidence_threshold = 0.5
        detected_objects = [(box, labels[class_idx], score) for box, class_idx, score in zip(boxes, classes, scores) if score > confidence_threshold]

        # Draw bounding boxes on the original image
        draw = ImageDraw.Draw(original_img)
        for box, label, score in detected_objects:
            x_min, y_min, x_max, y_max = box
            draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red", width=2)
            draw.text((x_min, y_min - 10), f"{label}: {score:.2f}", fill="red")

        return original_img, detected_objects
    except Exception as e:
        raise ValueError(f"Error during detection: {e}")

# Load the trained object detection model
try:
    model = load_model('object_detection_model.h5')  # Update with your model's filename
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
st.title("Scalpel Object Detection")

test_image = st.file_uploader("Upload an Image of the Object:", type=["jpg", "jpeg", "png"])
if test_image is not None:
    try:
        st.image(test_image, width=300, caption="Uploaded Image")
        labels = load_labels("labels.txt")  # Update with your labels filename

        if st.button("Detect Objects"):
            st.write("Detecting objects...")
            if labels:
                detected_image, detected_objects = detect_objects(test_image, model, labels)
                st.image(detected_image, caption="Detected Objects", use_column_width=True)

                st.write("### Detected Objects")
                for box, label, score in detected_objects:
                    st.write(f"- **{label}** with confidence {score:.2f}")
            else:
                st.error("Labels not found. Please upload the label file.")
    except Exception as e:
        st.error(f"Error processing the uploaded image: {e}")
