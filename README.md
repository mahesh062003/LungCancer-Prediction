# Medical Image Classification Web Application

This web application is designed to classify medical images, specifically CT scans and histopathology images, into different categories using deep learning models. The application is built using Flask, a lightweight Python web framework.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Routes](#routes)
- [Models](#models)
- [Image Preprocessing](#image-preprocessing)
- [Prediction Functions](#prediction-functions)
- [Contributing](#contributing)
- [License](#license)

## Overview

The application allows users to upload medical images, which are then classified using pre-trained deep learning models. The classifications include types of lung cancer and benign tissues. The results are returned to the user as JSON responses.

### Supported Models:
- **CT Scan Model:** Classifies CT scan images into `Adenocarcinoma`, `Benign`, or `Squamous Cell Carcinoma`.
- **Histopathology Model:** Classifies histopathology images into `Lung adenocarcinoma`, `Lung benign tissue`, `Lung squamous cell carcinoma`, or `None`.

## Installation

To set up this application locally, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Place your trained models in the appropriate directories:**
    - `ct_scan_model.h5` in the root directory.
    - `histopathology_model.h5` in the `models/` directory.

5. **Run the application:**
    ```bash
    python app.py
    ```

6. **Access the application:**
   - Open your web browser and go to `http://127.0.0.1:5000/`.

## Usage

### Home Page
The home page provides links to two main functionalities:
- CT Scan Image Classification
- Histopathology Image Classification

### Routes

- **`/` (Home Page):** The landing page with navigation to CT scan and histopathology image classification.

- **`/ct_scan` (CT Scan Classification):** 
  - `GET` request: Renders the CT scan upload form.
  - `POST` request: Accepts an uploaded CT scan image, preprocesses it, and returns the predicted class as JSON.

- **`/histo_image` (Histopathology Image Classification):**
  - `GET` request: Renders the histopathology image upload form.
  - `POST` request: Accepts an uploaded histopathology image, preprocesses it, and returns the predicted class as JSON.

## Models

The application loads two pre-trained deep learning models:

1. **CT Scan Model:** Located at `ct_scan_model.h5`. It classifies images into three categories:
    - Adenocarcinoma
    - Benign
    - Squamous Cell Carcinoma

2. **Histopathology Model:** Located at `models/histopathology_model.h5`. It classifies images into four categories:
    - Lung adenocarcinoma
    - Lung benign tissue
    - Lung squamous cell carcinoma
    - None

## Image Preprocessing

- **CT Scan Images:**
  - Images are resized to `150x150` pixels.
  - Pixel values are normalized (divided by 255.0).
  - Images are reshaped to fit the model's expected input format.

- **Histopathology Images:**
  - Images are resized to `150x150` pixels.
  - Pixel values are normalized.
  - The image is expanded along the batch dimension before being passed to the model.

## Prediction Functions

- **CT Scan Prediction:** 
  - The image is preprocessed using the `preprocess_ct_image` function.
  - The model predicts the class, which is then mapped to a human-readable label.

- **Histopathology Prediction:**
  - The image is preprocessed using the `preprocess_histo_image` function.
  - The model predicts the class, and the result is returned as a label from the `histo_class_labels` dictionary.

## Contributing

If you'd like to contribute to this project, please fork the repository and submit a pull request. Issues and feature requests are welcome.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
