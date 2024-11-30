# Brain Tumor Classification

## Introduction
This project is a machine learning-based application designed to detect and classify brain tumors from medical images. It utilizes a trained neural network to analyze images and predict the presence and type of tumor. The application provides a user-friendly API for image uploads or URL-based predictions, with results displayed dynamically.

---

## Model Details
The application uses a deep learning model built with TensorFlow/Keras.

- **Model File**: `keras_Model.h5`
- **Label File**: `labels.txt`
- **Input Requirements**: RGB images resized to 224x224.
- **Preprocessing**: Images are normalized to a range of -1 to 1.
- **Output**: Predictions with a confidence score.

---

## App Architecture
The application is developed using **FastAPI**, a modern web framework for building APIs.

### Features:
- **Image Upload**: Users can upload images for tumor classification.
- **URL Input**: Users can provide an image URL for predictions.
- **Dynamic Web Pages**: Results are displayed using templated HTML pages.
- **User Data Management**: Tracks user details for future expansion (e.g., database integration).

### Endpoints:
- **`GET /`**: Home page with an upload form.
- **`POST /predict`**: Predicts tumor type from an uploaded image.
- **`POST /predict-url`**: Predicts tumor type from an image URL.

---

## Installation Instructions

### Prerequisites:
- Python 3.8 or higher.
- A virtual environment (recommended for dependency isolation).

### Steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/chandan043/Brain-Tumor-Classification.git
    cd Brain-Tumor-Classification
    ```

2. **Set Up a Virtual Environment**:
    - macOS/Linux:
      ```bash
      python3 -m venv env
      source env/bin/activate
      ```
    - Windows:
      ```bash
      python -m venv env
      .\env\Scripts\activate
      ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Ensure Model and Labels Are Available**:
    Place the following files in the project directory:
    - `keras_Model.h5` (Trained model file).
    - `labels.txt` (Classification labels).

5. **Run the Application**:
    ```bash
    uvicorn main:app --reload
    ```

6. **Access the Application**:
    Open your browser and navigate to:
    ```
    http://127.0.0.1:8000
    ```

---

## Usage Instructions

### How to Use:
1. **Upload an Image**:
    - Visit the home page (`http://127.0.0.1:8000`).
    - Upload an image for prediction.
    - Receive a classification result with a confidence score.

2. **Use the URL Input**:
    - Submit a URL to the `/predict-url` endpoint.
    - The API fetches and processes the image, returning predictions.

### Example Response:
![pic1](https://github.com/user-attachments/assets/41d15ff8-419b-4a0e-8699-7e0d0457065f)
![pic2](https://github.com/user-attachments/assets/539fe093-3553-4ea9-b248-380e73723f50)
![pic3](https://github.com/user-attachments/assets/85a5d8ac-c97c-4d10-8c81-e7a5a2130b1b)
![pic4](https://github.com/user-attachments/assets/4dab04c8-1a1e-487c-9ba6-4a14b0605155)
![pic5](https://github.com/user-attachments/assets/9fa6c964-85b2-4eeb-b73b-c8d14d7047df)

