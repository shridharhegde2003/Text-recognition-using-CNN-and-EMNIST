# Handwritten Character Recognition using CNN

This project is a web application that recognizes handwritten characters (A-Z, a-z, 0-9) using a Convolutional Neural Network (CNN). The model is trained on the EMNIST dataset and deployed with a Flask backend. The frontend provides two ways to input a character: by uploading an image or by drawing directly on a canvas.

### Key Features:
- **Robust CNN Model**: Trained on 62 classes with data augmentation for improved accuracy.
- **Dual Input**: Accepts both direct image uploads and drawings on an HTML5 canvas.
- **Advanced Image Preprocessing**: Uses OpenCV to process inputs for better real-world performance.
- **Interrupt-Safe Training**: The included training notebook (`EMNIST_Advanced_Training.ipynb`) saves the best model automatically.

---

## How to Run This Project

Follow these steps to get the application running on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/shridharhegde2003/Text-Recognition-using-CNN-and-EMNIST.git
cd Text-Recognition-using-CNN-and-EMNIST
```

### 2. Create a Python Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install all the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Run the Flask Application

Once the dependencies are installed, you can start the Flask server.

```bash
python app.py
```

### 5. Open in Browser

The application will be running locally. Open your web browser and navigate to:

[**http://127.0.0.1:5000**](http://127.0.0.1:5000)

You can now test the character recognition by uploading an image or drawing on the canvas.
