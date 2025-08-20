# Facial Emotion Recognition for Mental Health Diagnostics

This project explores the use of **deep learning-based Facial Emotion
Recognition (FER)** to support **mental health diagnostics and
interventions**. It leverages **Convolutional Neural Networks (CNNs)**
trained on established datasets (e.g., FER2013, JAFFE) to classify human
emotions from facial expressions and demonstrate potential applications
in digital health.

------------------------------------------------------------------------

##  Project Overview

-   Mental health disorders affect over **970 million people
    worldwide**. Diagnosis often relies on **subjective interviews** and
    **self-reports**, which are prone to bias and stigma.
-   This project applies **AI and Computer Vision** to detect emotions
    (Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral) as
    supplementary diagnostic tools.
-   It includes:
    -   CNN model training & evaluation
    -   Visualization of performance (accuracy, confusion matrix)
    -   Real-time emotion recognition demo with webcam
    -   Research-backed documentation on ethical, clinical, and
        technical aspects

------------------------------------------------------------------------

## 📂 Repository Structure

    facial-emotion-recognition-mental-health/
    │── fer_model.py          # Model training and evaluation
    │── real_time_demo.py     # Real-time detection with webcam
    │── requirements.txt      # Dependencies
    │── models/               # Pretrained models
    │── datasets/             # Instructions for FER2013/JAFFE dataset
    │── reports/Final_Report.pdf
    │── results/              # Outputs: graphs, confusion matrix
    │── README.md

------------------------------------------------------------------------

## ⚙️ Installation

1.  Clone the repository:

    ``` bash
    git clone https://github.com/<your-username>/facial-emotion-recognition-mental-health.git
    cd facial-emotion-recognition-mental-health
    ```

2.  Create a virtual environment and install dependencies:

    ``` bash
    pip install -r requirements.txt
    ```

### Dependencies

-   Python 3.8+
-   TensorFlow / Keras
-   NumPy, Pandas, Matplotlib, Seaborn
-   OpenCV
-   scikit-learn

------------------------------------------------------------------------

## 📊 Training & Evaluation

1.  Download and extract the dataset (FER2013 or JAFFE).

2.  Update the dataset path inside `fer_model.py`.

3.  Run training:

    ``` bash
    python fer_model.py
    ```

4.  Results (accuracy/loss plots, confusion matrix, classification
    report) will be saved in the `results/` folder.

------------------------------------------------------------------------

## 🎥 Real-Time Emotion Detection

Run:

``` bash
python real_time_demo.py
```

A webcam window will open, detecting faces and displaying classified
emotions in real-time.

------------------------------------------------------------------------

## 📖 Report

A detailed report of this project, including literature review,
methodology, ethical implications, and results, is available in: -
[`reports/Final_Report.pdf`](reports/Final_Report.pdf)

------------------------------------------------------------------------

## 🔮 Future Work

-   Improve dataset diversity for fairness and inclusivity
-   Add temporal modeling (RNNs, Transformers) for emotion tracking over
    time
-   Integrate multimodal data (voice, text, heart rate)
-   Enhance explainability of CNN predictions (Grad-CAM, SHAP, LIME)

------------------------------------------------------------------------

## 📜 License

This project is licensed under the MIT License.

------------------------------------------------------------------------

## 👥 Contributors

-   **Hillary Onyango** -- Project Implementation & Report
-   Supervisors / Acknowledgements (if any)

------------------------------------------------------------------------
