#  Face-Detection & Gender-Classification
![imgalt](https://github.com/PALLAVI-ANAPATI/Gender_Classification/blob/main/Dataset_examples/01.jpg)
# 📌Description
This project utilizes transfer learning with the Xception model to achieve 94% accuracy in gender classification. It employs MTCNN for face detection, data augmentation to enhance generalization, and matplotlib for visual representation (blue for males, red for females). The model, deployed on Streamlit Cloud, allows users to predict gender from live camera input or uploaded images, making it a robust and efficient solution. 🚀

---
## 📑 Table of Contents
- [🚀 Key Features & Methodology](#-key-features--methodology)
- [📂 Dataset](#-dataset)
- [🛠️ Dependencies](#%EF%B8%8F-dependencies)
- [🔄 Data Preprocessing](#-data-preprocessing)
- [🏗️ Model Architectures Used](#%EF%B8%8F-model-architectures-used)
- [📌 Conclusion](#-conclusion)
- [🎬 Model Demo](#-model-demo)

---

## 🚀 Key Features & Methodology
✅ **Transfer Learning with Xception:** Frozen CNN layers, retrained dense layers for gender classification.

✅ **Data Augmentation:** Applied transformations to prevent overfitting and improve model generalization.

✅ **Optimized Training with Callbacks:** Saved the best model for peak performance.

✅ **High Accuracy:** Achieved **94% accuracy** in classification.

✅ **Face Detection with MTCNN:** Detected faces before classification.

✅ **Visual Representation with Matplotlib:**
   - 🔵 **Male Faces**: Marked with a blue rectangle.
   - 🔴 **Female Faces**: Marked with a red rectangle.
   - 
✅ **Cropped Face-Based Prediction:** Each detected face is resized before feeding it to the model.

✅ **Gender Count Feature:** Counts the number of male and female faces in an image.

✅ **Deployment with Streamlit:** Test the model easily via Streamlit Cloud.

✅ **Multiple Input Options:**
   - 📷 **Live Camera Feed** (Phone/Laptop)
   - 🖼️ **Uploaded Images**

---

## 📂 Dataset
1️⃣ **47K+ Cropped Faces** Dataset - [🔗 Kaggle Source](https://www.kaggle.com/cashutosh/gender-classification-dataset)  
2️⃣ **200K+ Uncropped Faces** Dataset - [🔗 Kaggle Source](https://www.kaggle.com/ashishjangra27/gender-recognition-200k-images-celeba)

---

## 🛠️ Dependencies
Before running the project, ensure the following Python libraries are installed:

- `tensorflow` - Deep learning model development
- `keras` - High-level API for deep learning models
- `matplotlib` - Data visualization
- `numpy` - Numerical data handling
- `pandas` - Data manipulation and analysis
- `mtcnn` - Face detection
- `pillow` - Image processing
- `streamlit` - Web-based application
- `opencv-python` - Image and video processing

### 📥 Install Dependencies:
```sh
pip install tensorflow keras matplotlib numpy pandas mtcnn pillow streamlit opencv-python
```

---

## 🔄 Data Preprocessing
To improve model robustness and accuracy, we applied the following preprocessing techniques:

### **📝 Training Data Preprocessing:**
✅ **Horizontal Flip** - Random flipping for better orientation recognition.

✅ **Width & Height Shift (40%)** - Handles position variations.

✅ **Zoom Augmentation (30%)** - Recognizes faces at different distances.

✅ **Rotation Augmentation (20°)** - Manages head tilts.

✅ **Rescaling (1/255)** - Normalizes pixel values for stable training.

### **📝 Test Data Preprocessing:**
✅ **Rescaling Only** - No additional transformations, ensuring fair evaluation.

These steps enhance the model’s ability to generalize across diverse conditions while maintaining accuracy. 🚀

---

## 🏗️ Model Architectures Used
We used **Xception** due to its high accuracy and efficiency:

- **Xception**: An advanced **CNN** that utilizes **depthwise separable convolutions**, reducing computational cost while maintaining high performance. It achieved **94% accuracy** in our experiments.

---

## 🔚 Conclusion
This project successfully implements a **deep learning-based gender classification system** using the **Xception model**. By leveraging **MTCNN** for face detection and **Streamlit** for a user-friendly interface, the application can:

✅ **Detect faces** in an uploaded image or live camera feed.  
✅ **Classify gender** with high accuracy (**94%**).  
✅ **Provide a simple and intuitive user experience.**  

### 📌 Future Improvements:
🚀 **Expand the dataset** with more diverse regional images.  
📈 **Fine-tune the model** with additional training techniques.  
🖥️ **Optimize for real-time performance** on edge devices.  

This project demonstrates the power of **deep learning in computer vision** and can be extended for various classification tasks. 🚀

---

## 🎬 Model Demo
![imgalt](https://github.com/PALLAVI-ANAPATI/Gender_Classification/blob/main/Dataset_examples/02.png)




