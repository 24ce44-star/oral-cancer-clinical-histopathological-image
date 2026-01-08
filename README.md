# Oral-cancer-clinical+histopathological-image

Oral Cancer Detection using ResNet-50
Overview

This project focuses on oral cancer detection using deep learning techniques applied to two types of medical images:

Clinical oral images
![image alt](https://github.com/24ce44-star/oral-cancer-clinical-histopathological-image/blob/1276094c60d2b9f126ea4ffa99c7f105cd52f39f/Screenshot%202026-01-08%20145911.png)

Histopathological images
![image alt](https://github.com/24ce44-star/oral-cancer-clinical-histopathological-image/blob/1276094c60d2b9f126ea4ffa99c7f105cd52f39f/Screenshot%202026-01-08%20152926.png)

A ResNet-50 architecture is used as the backbone model to classify images into cancerous and non-cancerous categories.

Due to file size limitations, trained model weights are not included in this repository. Instructions to download the model are provided below.

# app.py overview
![image alt()



Model Architecture

Base Model: ResNet-50

Framework: PyTorch

Task: Binary classification (Cancer vs Non-Cancer)

Dataset Description
1. Clinical Images

Clinical images consist of oral cavity photographs captured under real-world conditions.

Label Encoding (Important):

0 → Cancer

1 → Non-Cancer

Data Split:

Training: 80%

Validation: 20%

(Note: Exact image counts vary, but the split ratio is consistent.)

2. Histopathological Images

Histopathological images are microscopic tissue samples used for medical diagnosis.

Classes:

Oral Normal

Oral Squamous Cell Carcinoma (Oral SCC)

Label Encoding:

1 → Cancer (Oral SCC)

0 → Non-Cancer (Oral Normal)

Data Split:

Training set:

4000 images (Oral Normal)

4000 images (Oral SCC)

Validation set:

1001 images (Oral Normal)

1001 images (Oral SCC)

This results in a balanced dataset for both training and validation.

Data Preprocessing

Image resizing and normalization

Standard augmentation techniques (used during training)

Separate preprocessing pipelines for clinical and histopathological images as required

Model Weights

Trained model weights are not stored in this repository due to size constraints.

📦 Download the trained model from the link below:
👉 (Add Google Drive / HuggingFace link here)

After downloading, place the model file in the following directory:

models/best.pt
Project Structure
oral-cancer-clinical-histopathological-image/
│
├── app.py
├── train.py
├── inference.py
├── requirements.txt
├── README.md
├── .gitignore
└── models/        # Model files are downloaded separately
How to Run

Clone the repository

Install dependencies:

pip install -r requirements.txt

Download the trained model and place it in models/

Run inference or application script:

python app.py
Notes

Label encoding differs between clinical and histopathological datasets and should be handled carefully during training and inference.

This project is intended for research and educational purposes.

Acknowledgements

This project was developed as part of an academic / hackathon-oriented deep learning initiative focusing on medical image analysis.
