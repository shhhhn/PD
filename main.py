import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from io import BytesIO

# Streamlit page configuration
st.set_page_config(page_title="Scalpel Object Detection System", layout="wide")

st.write(
    "<div style='text-align: center; font-size: 50px;'>Welcome to the Scalpel Object Detection System</div>",
    unsafe_allow_html=True,
)

# Load a pre-trained object detection model
@st.cache_resource
def load_model():
    # Load your model from the path (replace with your model's path)
    model = tf.saved_model.load("ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model")  # Replace with your model path
    return model

# Function to preprocess the uploaded image
def load_image(image_file):
    """Preprocess the uploaded image to make it compatible with the object detection model."""
    img = Image.open(image_file)
    img = img.convert('L')  # Convert image to grayscale (if required by your model)
    img = img.resize((100, 100))  # Resize to 100x100 as expected by your model
    img = np.array(img)
    img = img.astype('float32') / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=-1)  # Add the channel dimension (to match (100, 100, 1))
    img = np.expand_dims(img, axis=0)  # Add batch dimension (to match (1, 100, 100, 1))
    return img

# Run object detection on the uploaded image
def detect_objects(model, image_np):
    """Detect objects in the image and draw bounding boxes around them."""
    # Convert the image to a tensor
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis,...]  # Add batch dimension

    # Run the detection
    model_fn = model.signatures['serving_default']
    detections = model_fn(input_tensor)

    # Get detected boxes, class labels, and confidence scores
    boxes = detections['detection_boxes'][0].numpy()
    class_ids = detections['detection_classes'][0].numpy().astype(int)
    scores = detections['detection_scores'][0].numpy()

    return boxes, class_ids, scores

# Function to draw bounding boxes on the image
def draw_boxes(image, boxes, class_ids, scores, threshold=0.5):
    """Draw bounding boxes on the image."""
    h, w, _ = image.shape

    for i in range(len(boxes)):
        if scores[i] > threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            xmin, ymin, xmax, ymax = int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)

            # Draw bounding box
            image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Add label
            label = f"Class {class_ids[i]}: {scores[i]:.2f}"
            image = cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return image

# Page title and file upload functionality
st.title("Scalpel Object Detection")

test_image = st.file_uploader("Upload an Image of the Object:", type=["jpg", "jpeg", "png"])

if test_image is not None:
    try:
        # Load and preprocess the image
        image_np = load_image(test_image)
        st.image(image_np[0], width=300, caption="Uploaded Image")  # Show the original uploaded image

        # Load the object detection model
        model = load_model()

        # Detect objects
        boxes, class_ids, scores = detect_objects(model, image_np)

        # Draw bounding boxes on the image
        image_with_boxes = draw_boxes(image_np[0], boxes, class_ids, scores)

        # Display the result with bounding boxes
        st.image(image_with_boxes, width=500, caption="Detected Objects")
    except Exception as e:
        st.error(f"Error processing the uploaded image: {e}")
