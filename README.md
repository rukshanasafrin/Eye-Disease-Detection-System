# 👁️ Eye Disease Detection

An AI-powered **Eye Disease Detection** web application that allows users to **upload eye images** and predicts whether an eye disease is present.

---

## 🚀 Features
- 📷 **Upload** an eye image for disease detection.
- 🔍 **AI-Powered Prediction** using a trained model.
- 🎨 **Smooth UI** with a **blurred background & animations**.
- 📱 **Responsive Design** (works on mobile & desktop).

---

## 🛠️ Technologies Used
- **Frontend**: HTML, CSS (with animations & responsive design)
- **Backend**: Flask (Python)
- **Machine Learning**: TensorFlow/Keras, Numpy, Grad-CAM for explainable AI
- **Deployment**: Flask server & GitHub (for hosting)

---

## 📥 Installation & Setup
Follow these steps to **run the project locally**:

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/rukshanasafrin/Eye-Disease-Detection-System.git
cd Eye-Disease-Detection-System
```

### 2️⃣ Install Dependencies
Make sure you have **Python 3.7+** installed. Then, install the required packages:
```sh
pip install -r requirements.txt
```

### 3️⃣ Run the Flask Server
```sh
python app.py
```
The app will start on **http://127.0.0.1:5000/**.

---

## 🖼️ Project Structure
```
Eye-Disease-Detection-System/
│── app.py                  # Flask application entry point
│── preprocess.py           # Image preprocessing logic
│── gradcam.py              # Grad-CAM visualization generation
│── requirements.txt        # Project dependencies
│── README.md               # Project documentation
│
├── static/
│   ├── styles.css          # Application styling
│   ├── images.jpeg         # UI / sample image asset
│   ├── preprocessed.jpg    # Example preprocessed image output
│   └── gradcam.jpg         # Example Grad-CAM output
│
└── templates/
    ├── index.html          # Upload page
    └── result.html         # Prediction result page
```

---

## 🎯 Future Improvements
🚀 **Upgrade AI Model** for more accuracy.  
📊 **Add Confidence Scores** in predictions.  
🌍 **Deploy to Cloud** for public access.  

---


## 🤝 Contributing
Pull requests are welcome! Feel free to fork the repository and make improvements.

---
