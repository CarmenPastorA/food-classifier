# 🍽️ Food Image Classifier

This project uses Deep Learning with PyTorch to classify food images. It provides a FastAPI REST backend and a simple UI for testing.

## 🔧 Features

- PyTorch CNN baseline model
- FastAPI REST API
- Gradio-based UI for uploading images
- Ready-to-use structure for production deployment

## 📁 Folder Structure

```bash
food-classifier-api/
├── app/                         # Main app logic (API, models, services)
│   ├── api/                     # FastAPI routes
│   │     └── endpoints.py
│   ├── core/                    # Config and environment management
│   ├── services/                # Prediction logic
│   │   └── inference.py
│   ├── models/                  # PyTorch models and loader
│   │   ├── baseline_cnn.py
│   │   └── model_loader.py
│   ├── utils/                   # Image preprocessing and helpers
│   │   └── image_utils.py
│   └── main.py                  # FastAPI entrypoint
├── config/                      # Config files (YAML, JSON)
├── data/                        # Datasets
│   ├── raw/                     # Original data (not used directly)
│   └── processed/               # Ready-to-use data for training
│       └── train/
├── notebooks/                   # Jupyter notebooks for exploration
├── scripts/                     # Training and evaluation scripts
│   └── train.py
├── experiments/                 # Trained model checkpoints and logs
│   ├── checkpoints/
│   └── tensorboard_logs/
├── tests/                       # Unit tests
├── ui/                          # Gradio or Streamlit app
│   └── gradio_app.py
├── requirements.txt             # Python dependencies
├── .gitignore                   # Ignored files and folders
├── README.md                    # Project overview
└── setup.py                     # Optional for packaging
```

## 🚀 Installation

```bash
git clone https://github.com/your-username/food-classifier-api.git
cd food-classifier-api
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## 🧠 Train the model

```bash
python scripts/train.py
```

Place your images in `data/processed/train/` folder following the structure expected by ImageFolder.

## ▶️ Run the API

```bash
uvicorn app.main:app --reload
```

## 🖼️ Launch the UI

```bash
python ui/gradio_app.py
```

