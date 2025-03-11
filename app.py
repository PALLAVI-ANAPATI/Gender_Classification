import tensorflow as tf
import numpy as np
from mtcnn import MTCNN
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

# Declaring constants
size = 249
target_size = (size, size)

# Load the model
model = tf.keras.models.load_model("xception_v2_02_0.938.h5")

def main():
    st.title("🔍 Gender Classification App")
    st.sidebar.title("⚙ Select Mode")
    
    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        ["🖼️ Predict gender from image", "📸 Predict gender from camera"]
    )

    if app_mode == "🖼️ Predict gender from image":
        uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            process_and_predict(uploaded_file)

    elif app_mode == "📸 Predict gender from camera":
        picture = st.camera_input("Take a picture")
        if picture is not None:
            process_and_predict(picture)

def process_and_predict(image_file):
    # Replacing "Processed Image" with new text
    st.markdown("<h3 style='text-align: left; font-size: 20px;'>Here is the picture you've uploaded:</h3>", unsafe_allow_html=True)

    # Convert image to correct format
    image = Image.open(image_file)
    image_array = np.array(image)

    # Create face detector
    detector = MTCNN()
    faces = detector.detect_faces(image_array)

    # Get faces and coordinates
    cropped_imgs, coords = get_face_coords(image_array, faces)

    if not cropped_imgs:
        st.error("⚠ No faces detected. Try another image.")
        return

    # Make predictions
    result, predictions = predict_img(cropped_imgs)

    # Draw rectangles and display results
    draw_rect(image_array, coords, result)

    # Count the number of males and females
    male_count = result.count("Male")
    female_count = result.count("Female")
    total_faces = len(result)

    # Adjust font size based on image size
    summary_font_size = "26px"  # Increased font size
    count_font_size = "22px"

    # Move summary to the *left side*
    st.markdown(f"<h2 style='text-align: left; font-size: {summary_font_size};'>📊 Gender Classification Summary:</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"<h4 style='text-align: center; font-size: {count_font_size};'>👨 Male Count: {male_count}</h4>", unsafe_allow_html=True)

    with col2:
        st.markdown(f"<h4 style='text-align: center; font-size: {count_font_size};'>👩 Female Count: {female_count}</h4>", unsafe_allow_html=True)

    with col3:
        st.markdown(f"<h4 style='text-align: center; font-size: {count_font_size};'>👥 Total Count: {total_faces}</h4>", unsafe_allow_html=True)
# Function to extract faces from the image
def get_face_coords(image, result_list):
    coords = []
    cropped_imgs = []

    for result in result_list:
        if result['confidence'] > 0.96:  # Only consider high-confidence faces
            x1, y1, width, height = result['box']
            x2, y2 = x1 + width, y1 + height
            coords.append([x1, y1, width, height])

            cropped_face = image[y1:y2, x1:x2]
            cropped_imgs.append(cropped_face)
    
    return cropped_imgs, coords

# Function to predict gender
def predict_img(cropped_imgs):
    results = []
    predictions = []

    for crop in cropped_imgs:
        img = Image.fromarray(crop, 'RGB')
        img = img.resize(target_size)
        img = np.array(img) / 255.0
        img = img.reshape(1, size, size, 3)

        # Prediction
        pred = model.predict(img)
        pred = pred[0][0]
        predictions.append(pred)

        if pred >= 0.5:
            results.append("Male")
        else:
            results.append("Female")

    return results, predictions

# Function to draw rectangles and labels (with smaller text and no box around text)
def draw_rect(image, coords, results):
    fig, ax = plt.subplots(figsize=(3, 3))  # Decrease figure size
    ax.imshow(image)

    for (x1, y1, width, height), label in zip(coords, results):
        color = "blue" if label == "Male" else "red"
        rect = Rectangle((x1, y1), width, height, fill=False, color=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, label, color=color, fontsize=6, fontweight='bold')  # Adjusted font size

    ax.axis("off")
    st.pyplot(fig)

if __name__ == '__main__':
    main()

