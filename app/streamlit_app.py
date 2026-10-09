import sys
from pathlib import Path

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
MODEL_PATH = PROJECT_ROOT / "outputs" / "best_model.pt"

sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(SRC_DIR))

from src.config import IMAGE_SIZE, THRESHOLD
from src.model import UNet


def get_risk_category(score):
    if score >= 0.35:
        return "High Risk"
    if score >= 0.15:
        return "Medium Risk"
    return "Low Risk"


def get_risk_color(category):
    if category == "High Risk":
        return "#dc2626"
    if category == "Medium Risk":
        return "#d97706"
    return "#16a34a"


def normalize_factor(value, maximum):
    return min(max(value / maximum, 0.0), 1.0)


def calculate_environment_risk(rainfall_mm, slope_degree, soil_sensitivity, geology_sensitivity, vegetation_loss, road_exposure):
    rainfall_score = normalize_factor(rainfall_mm, 200.0)
    slope_score = normalize_factor(slope_degree, 60.0)
    soil_score = soil_sensitivity / 10.0
    geology_score = geology_sensitivity / 10.0
    vegetation_score = vegetation_loss / 100.0
    road_score = road_exposure / 10.0

    return (
        0.30 * rainfall_score
        + 0.25 * slope_score
        + 0.15 * soil_score
        + 0.15 * geology_score
        + 0.10 * vegetation_score
        + 0.05 * road_score
    )


def calculate_combined_warning_score(visual_score, environment_score):
    return 0.60 * visual_score + 0.40 * environment_score


@st.cache_resource
def load_trained_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


def first_h5_dataset(group):
    for value in group.values():
        if isinstance(value, h5py.Dataset):
            return value[()]
        if isinstance(value, h5py.Group):
            nested = first_h5_dataset(value)
            if nested is not None:
                return nested
    return None


def read_h5_image(uploaded_file):
    with h5py.File(uploaded_file, "r") as file:
        if "img" in file:
            image = file["img"][()]
        elif "image" in file:
            image = file["image"][()]
        elif "data" in file:
            image = file["data"][()]
        else:
            image = first_h5_dataset(file)

    if image is None:
        raise ValueError("No image dataset found inside the H5 file.")

    image = np.asarray(image)
    if image.ndim == 2:
        image = image[:, :, None]
    if image.shape[0] <= 20 and image.shape[-1] > 20:
        image = np.moveaxis(image, 0, -1)

    image = image[:, :, :3]
    if image.shape[-1] < 3:
        missing_channels = 3 - image.shape[-1]
        image = np.concatenate(
            [image, np.repeat(image[:, :, -1:], missing_channels, axis=-1)],
            axis=-1,
        )

    image = image.astype(np.float32)
    display_image = np.zeros_like(image, dtype=np.float32)
    for channel in range(3):
        values = image[:, :, channel]
        min_value = np.nanmin(values)
        max_value = np.nanmax(values)
        if max_value > min_value:
            display_image[:, :, channel] = (values - min_value) / (max_value - min_value)

    display_image = np.nan_to_num(display_image)
    return (display_image * 255).astype(np.uint8)


def prepare_uploaded_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        image = read_h5_image(uploaded_file)
    else:
        image = Image.open(uploaded_file).convert("RGB")
        image = np.array(image)

    resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    tensor = resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(tensor).permute(2, 0, 1).unsqueeze(0)
    return image, tensor


@torch.no_grad()
def predict_landslide(model, device, image_tensor, original_shape, threshold):
    logits = model(image_tensor.to(device))
    probability = torch.sigmoid(logits)[0, 0].cpu().numpy()
    probability = cv2.resize(probability, (original_shape[1], original_shape[0]))
    mask = (probability > threshold).astype(np.uint8) * 255
    return probability, mask


def create_overlay(image, mask, alpha):
    red_mask = np.zeros_like(image)
    red_mask[:, :, 0] = mask
    return cv2.addWeighted(image, 1 - alpha, red_mask, alpha, 0)


def create_heatmap(probability, image, alpha):
    heatmap = np.uint8(255 * probability)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)


def image_download_bytes(image):
    success, buffer = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not success:
        return None
    return buffer.tobytes()


def grayscale_download_bytes(image):
    success, buffer = cv2.imencode(".png", image)
    if not success:
        return None
    return buffer.tobytes()


def plot_probability_histogram(probability, threshold):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(probability.flatten(), bins=40, color="#2563eb", alpha=0.85)
    ax.axvline(threshold, color="#dc2626", linewidth=2, label=f"Threshold: {threshold:.2f}")
    ax.set_title("Prediction Probability Distribution")
    ax.set_xlabel("Landslide probability")
    ax.set_ylabel("Pixel count")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_area_breakdown(affected_percent):
    safe_percent = max(0.0, 100.0 - affected_percent)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie(
        [affected_percent, safe_percent],
        labels=["Predicted landslide", "Other area"],
        autopct="%1.1f%%",
        colors=["#dc2626", "#16a34a"],
        startangle=90,
    )
    ax.set_title("Predicted Area Breakdown")
    fig.tight_layout()
    return fig


def plot_risk_gauge(risk_score):
    fig, ax = plt.subplots(figsize=(7, 2.2))
    ax.barh(["Risk"], [0.15], color="#16a34a", height=0.5)
    ax.barh(["Risk"], [0.20], left=[0.15], color="#d97706", height=0.5)
    ax.barh(["Risk"], [0.65], left=[0.35], color="#dc2626", height=0.5)
    ax.scatter([risk_score], ["Risk"], color="#111827", s=120, zorder=3)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Risk score")
    ax.set_title("Risk Scale")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return fig


def plot_environment_factors(factors):
    labels = list(factors.keys())
    values = list(factors.values())
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=["#2563eb", "#7c3aed", "#d97706", "#dc2626", "#16a34a", "#0891b2"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Normalized contribution")
    ax.set_title("Early Warning Factor Scores")
    ax.tick_params(axis="x", rotation=25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    return fig


st.set_page_config(page_title="project_lithos", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
    }
    .status-card {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 16px;
        background: #ffffff;
    }
    .risk-pill {
        display: inline-block;
        color: white;
        padding: 8px 12px;
        border-radius: 999px;
        font-weight: 700;
    }
    .small-note {
        color: #6b7280;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("project_lithos")
st.caption("AI-based landslide detection prototype for North Eastern India")

if not MODEL_PATH.exists():
    st.error("Model file not found. Place best_model.pt inside project_lithos/outputs/")
    st.stop()

model, device = load_trained_model(str(MODEL_PATH))

with st.sidebar:
    st.header("Controls")
    uploaded_image = st.file_uploader(
        "Upload satellite image or Landslide4Sense H5",
        type=["png", "jpg", "jpeg", "tif", "tiff", "h5", "hdf5"],
    )
    threshold = st.slider("Detection threshold", 0.05, 0.95, float(THRESHOLD), 0.05)
    overlay_alpha = st.slider("Overlay strength", 0.10, 0.80, 0.35, 0.05)
    st.divider()
    st.header("Early Warning Factors")
    rainfall_mm = st.number_input("24-hour rainfall (mm)", min_value=0.0, max_value=500.0, value=80.0, step=5.0)
    slope_degree = st.number_input("Slope angle (degrees)", min_value=0.0, max_value=90.0, value=30.0, step=1.0)
    soil_sensitivity = st.slider("Soil sensitivity", 0, 10, 5)
    geology_sensitivity = st.slider("Geology sensitivity", 0, 10, 5)
    vegetation_loss = st.slider("Vegetation loss (%)", 0, 100, 30)
    road_exposure = st.slider("Road/drainage exposure", 0, 10, 4)
    st.divider()
    st.write("Model")
    st.code(str(MODEL_PATH), language="text")
    st.write("Input size")
    st.code(f"{IMAGE_SIZE} x {IMAGE_SIZE}", language="text")

if uploaded_image is None:
    st.info("Upload an image or `.h5` file from the sidebar to generate a landslide prediction.")
    st.stop()

image, image_tensor = prepare_uploaded_file(uploaded_image)
probability, mask = predict_landslide(model, device, image_tensor, image.shape[:2], threshold)
overlay = create_overlay(image, mask, overlay_alpha)
heatmap = create_heatmap(probability, image, overlay_alpha)

risk_score = float(probability.mean())
environment_score = calculate_environment_risk(
    rainfall_mm,
    slope_degree,
    soil_sensitivity,
    geology_sensitivity,
    vegetation_loss,
    road_exposure,
)
combined_warning_score = calculate_combined_warning_score(risk_score, environment_score)
risk_category = get_risk_category(combined_warning_score)
risk_color = get_risk_color(risk_category)
affected_pixels = int(np.count_nonzero(mask))
total_pixels = int(mask.size)
affected_percent = affected_pixels / total_pixels * 100
max_probability = float(probability.max())
mean_detected_probability = float(probability[mask > 0].mean()) if affected_pixels else 0.0

st.markdown(
    f"""
    <div class="status-card">
        <span class="risk-pill" style="background:{risk_color};">{risk_category}</span>
        <p class="small-note">
            Prediction generated from <b>{uploaded_image.name}</b>. This is a prototype result and should be verified with field or expert data.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
metric_col1.metric("Combined warning", f"{combined_warning_score:.3f}")
metric_col2.metric("Visual AI score", f"{risk_score:.3f}")
metric_col3.metric("Environment score", f"{environment_score:.3f}")
metric_col4.metric("Affected area", f"{affected_percent:.2f}%")

st.progress(min(combined_warning_score, 1.0), text=f"Combined early warning score: {combined_warning_score:.3f}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Prediction", "Confidence Heatmap", "Visualizations", "Early Warning", "Details"])

with tab1:
    col1, col2, col3 = st.columns(3)
    col1.image(image, caption="Input image", width="stretch")
    col2.image(mask, caption="Predicted landslide mask", width="stretch")
    col3.image(overlay, caption="Landslide overlay", width="stretch")

with tab2:
    col1, col2 = st.columns(2)
    col1.image(heatmap, caption="Probability heatmap overlay", width="stretch")
    col2.image((probability * 255).astype(np.uint8), caption="Raw probability map", width="stretch")

with tab3:
    chart_col1, chart_col2 = st.columns(2)
    chart_col1.pyplot(plot_probability_histogram(probability, threshold), width="stretch")
    chart_col2.pyplot(plot_area_breakdown(affected_percent), width="stretch")
    st.pyplot(plot_risk_gauge(combined_warning_score), width="stretch")

with tab4:
    factor_scores = {
        "Rainfall": normalize_factor(rainfall_mm, 200.0),
        "Slope": normalize_factor(slope_degree, 60.0),
        "Soil": soil_sensitivity / 10.0,
        "Geology": geology_sensitivity / 10.0,
        "Vegetation": vegetation_loss / 100.0,
        "Roads": road_exposure / 10.0,
    }
    st.subheader("Early Warning Assessment")
    ew_col1, ew_col2, ew_col3 = st.columns(3)
    ew_col1.metric("Visual AI score", f"{risk_score:.3f}")
    ew_col2.metric("Environment score", f"{environment_score:.3f}")
    ew_col3.metric("Combined warning", f"{combined_warning_score:.3f}")
    st.pyplot(plot_environment_factors(factor_scores), width="stretch")
    st.info(
        "The combined score uses 60% visual AI detection and 40% environmental indicators. "
        "Use these inputs as a project demonstration heuristic, not as an official warning system."
    )

with tab5:
    st.subheader("Result Summary")
    st.write(
        {
            "file_name": uploaded_image.name,
            "risk_category": risk_category,
            "combined_warning_score": round(combined_warning_score, 4),
            "visual_ai_score": round(risk_score, 4),
            "environment_score": round(environment_score, 4),
            "threshold": threshold,
            "rainfall_mm_24h": rainfall_mm,
            "slope_degree": slope_degree,
            "soil_sensitivity": soil_sensitivity,
            "geology_sensitivity": geology_sensitivity,
            "vegetation_loss_percent": vegetation_loss,
            "road_drainage_exposure": road_exposure,
            "affected_pixels": affected_pixels,
            "total_pixels": total_pixels,
            "affected_area_percent": round(affected_percent, 4),
            "max_probability": round(max_probability, 4),
            "mean_detected_probability": round(mean_detected_probability, 4),
        }
    )
    st.warning(
        "This app detects visual landslide-like regions in an uploaded patch. "
        "For real early warning, combine this with rainfall, slope, soil, geology, and local validation data."
    )

download_col1, download_col2, download_col3 = st.columns(3)
download_col1.download_button(
    "Download overlay",
    data=image_download_bytes(overlay),
    file_name="project_lithos_overlay.png",
    mime="image/png",
)
download_col2.download_button(
    "Download mask",
    data=grayscale_download_bytes(mask),
    file_name="project_lithos_mask.png",
    mime="image/png",
)
download_col3.download_button(
    "Download heatmap",
    data=image_download_bytes(heatmap),
    file_name="project_lithos_heatmap.png",
    mime="image/png",
)
