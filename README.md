# Eye Disease Classifier API

A machine learning API that classifies eye diseases from uploaded images using a pre-trained deep learning model.

## Overview

This project provides a REST API built with FastAPI that allows users to upload images of eyes and receive predictions about possible eye diseases. The application uses a computer vision model converted to ONNX format for efficient inference.

## Features

- Fast and efficient image classification using ONNX Runtime
- Simple REST API interface
- Support for common image formats
- Real-time predictions

## Tech Stack

- **FastAPI**: Modern, high-performance web framework for building APIs
- **ONNX Runtime**: Cross-platform inference engine for ONNX models
- **Pillow**: Python Imaging Library for image processing
- **TensorFlow**: Used for model development
- **Uvicorn**: ASGI server implementation

## Installation

### Prerequisites

- Python 3.12 or higher
- pip package manager

### Setup

1. Clone the repository
```bash
git clone https://github.com/yourusername/Eye-Disease-Classifier-API.git
cd Eye-Disease-Classifier-API
```

2. Create and activate a virtual environment
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Ensure the model file is in the correct location
```
model/eyes_diseases.onnx
```

## Usage

### Starting the API Server

Run the following command to start the API server:

```bash
python main.py
```

The API will be available at http://127.0.0.1:8000

### API Endpoints

1. **Home** - `GET /`
   - Returns a welcome message

2. **Predict** - `POST /predict`
   - Accepts an image file
   - Returns the predicted eye disease classification

## Model Information

The application uses a pre-trained deep learning model converted to ONNX format for efficient inference. The model can classify various eye diseases based on input images.

Supported classes are defined in the `utils.py` file's `class_names` list.

## Project Structure

```
Eye-Disease-Classifier-API/
├── main.py                # FastAPI application and endpoints
├── utils.py               # Utility functions for image processing
├── model/                 # Directory containing the ONNX model
│   └── eyes_diseases.onnx # Pre-trained eye disease classification model
├── requirements.txt       # Project dependencies
├── .venv/                 # Virtual environment (not tracked by git)
└── README.md              # Project documentation
```

## How It Works

1. The client sends an image to the API endpoint
2. The image is preprocessed using utilities in `utils.py`
3. The preprocessed image is fed into the ONNX model for inference
4. The model prediction is processed and returned as a JSON response

## Development

To extend or modify this project:

1. Update the model in the `model/` directory
2. Modify the preprocessing functions in `utils.py` if needed
3. Extend the API functionality in `main.py`

## Contributing

Contributions are welcome! Feel free to submit a Pull Request to enhance the project or fix any issues

## License

This project is free for anyone to use, modify, and deploy, especially for educational purposes. No specific license applies, ensuring maximum flexibility and accessibility for users.

## Contact

For inquiries or access to the trained model (as it's not uploaded due to size constraints), please connect with me on [LinkedIn](https://www.linkedin.com/in/nhassan96/).