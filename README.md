# 👕 Fashion Recommendation & Virtual Try-On System

### A smart AI-powered fashion recommendation system that not only suggests similar outfits but also lets users virtually try them on using state-of-the-art diffusion models.
<p align="center">
  <img src="https://img.shields.io/badge/AI-Fashion-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Flask-Backend-black?style=for-the-badge" />
  <img src="https://img.shields.io/badge/MobileNetV2-CV-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/HuggingFace-Free-orange?style=for-the-badge" />
</p>
# 🚀 Overview

This project combines computer vision + deep learning + generative AI to create an interactive fashion experience:

🔍 Find visually similar clothes using feature extraction
👗 Try outfits virtually on a human model
⚡ Built with lightweight and free tools
# 🧠 How It Works
User uploads image
        ↓
MobileNetV2 extracts features
        ↓
System finds 5 similar clothes
        ↓
User clicks "👗 Try On"
        ↓
Flask sends:
    - Selected cloth image
    - base_model.jpg
        ↓
Request sent to IDM-VTON (HuggingFace)
        ↓
AI generates try-on result (~30 sec)
        ↓
Result shown in UI (modal)

# 🛠️ Tech Stack
🔹 Backend
Python
Flask
🔹 Machine Learning
MobileNetV2 (feature extraction)
NumPy (feature storage & similarity)
🔹 Virtual Try-On
IDM-VTON (HuggingFace Spaces)
🔹 Frontend
HTML / CSS / JavaScript


### 📂 Project Structure
Fashion Recommendation System/
│
├── app.py
├── build_index.py
├── features.npy
├── image_paths.npy
│
├── static/
│   ├── base_model.jpg
│   └── dataset_images/
│
└── templates/
    └── index.html

### 📸 Base Model Image Guidelines

For best try-on results, your base_model.jpg should follow these:

## ✅ Works Best
Front-facing pose
Standing straight (T-pose or relaxed arms)
Plain background (white / grey / beige)
Full body visible
Tight clothing (helps AI detect body shape)
## ❌ Avoid
Side angles or turned poses
Crossed arms
Busy or cluttered backgrounds
Loose or patterned clothing
Accessories like sunglasses, hats
🧪 Virtual Try-On API (Free Usage)
🤔 Is IDM-VTON Free Forever?

Yes — but with some practical limitations:

Factor	Reality
💸 Cost	Completely FREE
🔐 API Key	Not required
⏳ Uptime	Spaces sleep when idle (first request may take ~60s)
🧍 Queue	May wait if many users are using it
⚡ Rate Limit	No strict limit, but avoid spamming
⚠️ Shutdown Risk	If Space is removed, service goes down
🔄 Backup Try-On Models

If IDM-VTON is down, switch easily:

# Backup 1
tryon_client = Client("levihsu/OOTDiffusion")

# Backup 2
tryon_client = Client("Nymbo/Virtual-Try-On")
⚙️ Installation & Setup
1️⃣ Clone Repository
git clone https://github.com/your-username/fashion-recommendation-system.git
cd fashion-recommendation-system
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Build Feature Index
python build_index.py
4️⃣ Run Flask App
python app.py
5️⃣ Open in Browser
http://127.0.0.1:5000
🎯 Key Features
🔍 Image-based clothing recommendation
👗 Virtual try-on using AI
⚡ Fast similarity search
🌐 Web-based UI
💸 Completely free deployment (HuggingFace Spaces)
⚠️ Limitations
Try-on takes ~30 seconds
Depends on external HuggingFace Space availability
Quality depends heavily on base model image
Not real-time
💡 Future Improvements
Real human upload instead of fixed base model
Better cloth fitting using segmentation models
GPU acceleration for faster results
Multi-angle try-on
Mobile app integration
