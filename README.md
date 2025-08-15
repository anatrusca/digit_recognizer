# Handwritten Digit Recognizer

An interactive **Streamlit app** for recognizing handwritten digits (0–9) using a trained CNN model on the MNIST dataset.  
Draw a digit on the canvas, and the app will predict it with a confidence score.

## Getting Started

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run Streamlit
```bash
streamlit run app.py
```

## Usage

1. Draw a digit (0–9) in the white canvas.

2. Click "Predict".

3. View:

    3.1 Predicted digit

    3.2 Confidence score (%)

    3.3 Probability chart for all digits