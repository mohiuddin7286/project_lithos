# 🏔️ Project Lithos

> **AI-Based Landslide Detection & Early Warning System for North Eastern India**

Project Lithos is a deep learning computer vision framework and interactive web dashboard designed to detect landslide-susceptible zones and post-landslide scars from satellite imagery. Combining a PyTorch **U-Net** semantic segmentation model with environmental factor heuristics (rainfall, slope, soil, geology, vegetation loss, road exposure), Lithos provides a comprehensive early warning assessment tool for disaster monitoring and remote sensing.

---

## 🚀 Key Features

- **Multi-Format Remote Sensing Support**: Processes standard satellite image formats (`.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`) and multi-spectral HDF5 datasets (`.h5`, `.hdf5`) such as **Landslide4Sense**.
- **Deep Learning Segmentation**: Custom **U-Net** convolutional neural network with batch normalization and skip connections for pixel-level landslide boundary detection.
- **Multi-Factor Risk Assessment**: Fuses **Visual AI Detection (60%)** with **Environmental Heuristics (40%)** (24h rainfall, slope angle, soil/geological vulnerability, vegetation loss, road exposure) to produce a combined risk category (*Low*, *Medium*, *High*).
- **Interactive Streamlit Dashboard**:
  - **Live Overlay & Heatmaps**: Visualizes original satellite imagery alongside binary masks, transparent color overlays, and Jet colormap probability heatmaps.
  - **Analytics & Visualizations**: Interactive probability distribution histograms, area breakdown pie charts, risk scale gauges, and factor contribution bar charts.
  - **Export Capabilities**: One-click download buttons for generated overlay images, binary masks, and heatmap visualizations.
- **CLI Utilities**: Scripts for standalone model training, dataset validation, and CLI inference.

---

## 🛠️ Project Architecture

```
project_lithos/
├── app/
│   └── streamlit_app.py      # Streamlit web application interface
├── src/
│   ├── config.py             # Hyperparameters & path configurations
│   ├── dataset.py            # PyTorch Dataset loader (supports standard & HDF5 images/masks)
│   ├── model.py              # PyTorch U-Net neural network architecture
│   ├── train.py              # Training pipeline with Dice score evaluation & model saving
│   └── predict.py            # Command-line inference script
├── notebooks/
│   ├── project-lithos.ipynb
│   └── project_lithos_kaggle_landslide4sense.ipynb  # Training & exploration notebooks
├── data/                     # Image & mask datasets (gitignored / local storage)
│   ├── images/
│   └── masks/
├── outputs/
│   └── best_model.pt         # Saved PyTorch trained model weights
├── docs/                     # Project documentation & proposals
├── requirements.txt          # Python dependency specifications
└── README.md                 # Project documentation
```

---

## 📦 Installation & Setup

### 1. Prerequisites
- Python 3.9+
- CUDA-compatible GPU (optional, but recommended for faster training and inference)

### 2. Clone Repository & Environment Setup
```bash
git clone https://github.com/your-username/project_lithos.git
cd project_lithos

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💡 Usage Guide

### 🌐 Running the Streamlit Web Application

To launch the interactive dashboard:

```bash
streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501`.

1. **Upload Input**: Upload a satellite image patch (`.png`, `.jpg`, `.tif`) or a Landslide4Sense `.h5` file using the sidebar.
2. **Adjust Thresholds**: Modify the detection confidence threshold and overlay transparency sliders.
3. **Configure Environmental Indicators**: Input local 24-hour rainfall (mm), slope degree, soil & geology sensitivity, vegetation loss %, and road exposure.
4. **Inspect Results**: Explore predictions across 5 interactive tabs (**Prediction**, **Confidence Heatmap**, **Visualizations**, **Early Warning**, and **Details**).
5. **Download Outputs**: Download high-resolution overlay masks, binary prediction masks, and heatmaps directly.

---

### 🏋️ Model Training

To train the U-Net model on your custom dataset or Landslide4Sense data:

```bash
python src/train.py --images data/images --masks data/masks --epochs 5 --batch-size 4 --image-size 256
```

Key training features:
- Computes **BCEWithLogitsLoss** alongside pixel-wise **Dice Score**.
- Automatically saves the best performing checkpoint to `outputs/best_model.pt`.

---

### 🔮 Command-Line Inference

To run single-image inference via CLI:

```bash
python src/predict.py --image path/to/satellite_image.png --model outputs/best_model.pt --output prediction_mask.png --threshold 0.5
```

This outputs:
- Generated binary prediction mask (`prediction_mask.png`)
- Calculated landslide risk score (mean probability)
- Predicted landslide coverage area percentage (%)

---

## 📊 Model & Heuristics Overview

### U-Net Architecture
- **Input Channels**: 3 (RGB / normalized multi-spectral bands)
- **Encoder**: 4 downsampling blocks (DoubleConv with 3x3 Conv + BatchNorm + ReLU followed by 2x2 MaxPool)
- **Bridge**: 512-channel bottleneck representation
- **Decoder**: 4 upsampling blocks (2x2 ConvTranspose + skip connections concatenation + DoubleConv)
- **Output Layer**: 1-channel binary segmentation logits (processed via Sigmoid)

### Early Warning Risk Formula
$$\text{Combined Risk} = 0.60 \times \text{Visual AI Score} + 0.40 \times \text{Environment Score}$$

Where Environmental Score is derived from normalized parameters:
- **24-hour Rainfall**: 30% weight
- **Slope Angle**: 25% weight
- **Soil Sensitivity**: 15% weight
- **Geology Vulnerability**: 15% weight
- **Vegetation Loss**: 10% weight
- **Road / Infrastructure Exposure**: 5% weight

---

## 📑 License & Disclaimer

This project is developed as a prototype research tool. Risk categories and predictions should be validated with ground truth field observations and authoritative meteorological data before use in critical safety applications.
