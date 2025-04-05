# LSTM Autoencoder for Time Series Anomaly Detection

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/rajatsingh0702/LSTMAE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1h62dcS5nWos4wczenkG8iDKTJiHRZIqk)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <!-- Choose your license -->

This repository contains the implementation of an LSTM (Long Short-Term Memory) Autoencoder for detecting anomalies in time series data. You can either use a pre-trained model provided here or train a new model on your own CSV dataset.

An interactive demo is available on Hugging Face Spaces, and a Google Colab notebook is provided for experimentation.

## Key Features

*   **LSTM Autoencoder:** Built using PyTorch.
*   **Two Modes:**
    1.  **Use Pre-trained Model:** Quickly analyze time series data using the included model.
    2.  **Train on Custom Data:** Upload your own CSV file to train a new LSTM Autoencoder tailored to your specific data.
*   **Comprehensive Output:** Generates insightful plots and artifacts:
    *   Andrews Curves Plot
    *   Training Loss Curve
    *   Anomaly Score Distribution
    *   Evaluation Curve (e.g., ROC Curve, Precision-Recall Curve, or your custom "ANDRE" curve - *please clarify if "ANDRE" is a custom metric*)
*   **Downloadable Results:** Packages the trained model, data scalers, and all generated plots into a convenient ZIP file for download.
*   **Interactive Demo:** Hugging Face Space for easy interaction without local setup.
*   **Colab Notebook:** Experiment with the code, training, and evaluation in a Google Colab environment.

## How it Works

An LSTM Autoencoder is trained on 'normal' time series data.
1.  The **Encoder** (an LSTM network) learns to compress the input time series into a lower-dimensional latent representation.
2.  The **Decoder** (another LSTM network) learns to reconstruct the original time series from this latent representation.
3.  During inference, the model tries to reconstruct new, unseen time series sequences.
4.  If a sequence is similar to the normal data seen during training, the reconstruction error (the difference between the input and the reconstructed output) will be low.
5.  If a sequence contains anomalies (patterns not seen during training), the model struggles to reconstruct it accurately, resulting in a high reconstruction error.
6.  By setting a threshold on the reconstruction error, we can classify sequences as normal or anomalous.

## Installation (Local Setup)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Rajatsingh24/LSTM-based-Autoencoder.git
    cd LSTM-based-Autoencoder
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    *Make sure you have a `requirements.txt` file in your repository.*
    ```bash
    pip install -r requirements.txt
    ```
    
## Usage

You can interact with the model primarily through the Hugging Face Space or the Colab Notebook.