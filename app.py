"""Interactive Streamlit walkthrough for the student brain tumor U-Net."""
import io
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import streamlit as st
import torch

from src.brain_tumor_seg.model import UNet

CHANNELS = ("FLAIR", "T1w", "T1gd", "T2w")

st.set_page_config(page_title="NeuroMask Lab", page_icon="🧠", layout="wide")

@st.cache_resource(show_spinner="Loading the trained U-Net...")
def load_model(checkpoint_bytes: bytes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location=device, weights_only=True)
    model = UNet()
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.to(device).eval()
    return model, device

def load_nifti(uploaded_file):
    suffix = ".nii.gz" if uploaded_file.name.lower().endswith(".nii.gz") else ".nii"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name
        return nib.load(temp_path).get_fdata(dtype=np.float32)
    finally:
        if temp_path and Path(temp_path).exists():
            Path(temp_path).unlink()

def normalize_modalities(volume):
    """Match training preprocessing: per-channel brain-voxel z-score and clip."""
    volume = volume.copy()
    for channel in range(4):
        image = volume[..., channel]
        brain = image != 0
        if brain.any():
            values = image[brain]
            mean, std = float(values.mean()), float(values.std())
            volume[..., channel] = np.where(
                brain, np.clip((image - mean) / max(std, 1e-6), -5, 5), 0
            )
    return volume

def binary_scores(prediction, reference, eps=1e-7):
    intersection = int(np.logical_and(prediction, reference).sum())
    pred_sum, ref_sum = int(prediction.sum()), int(reference.sum())
    union = pred_sum + ref_sum - intersection
    if pred_sum + ref_sum == 0:
        return 1.0, 1.0
    dice = (2 * intersection + eps) / (pred_sum + ref_sum + eps)
    iou = (intersection + eps) / (union + eps)
    return dice, iou

def show_slice(volume, slice_index, display_channel, prediction, probabilities, reference=None):
    image = volume[:, :, slice_index, display_channel].T
    panels = [("MRI input", image, None, "gray")]
    if reference is not None:
        panels.append(("Reference mask", image, reference.T, "Greens"))
    panels.extend([
        ("U-Net mask", image, prediction.T, "Reds"),
        ("Model probability", probabilities.T, None, "magma"),
    ])
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, base, overlay, cmap) in zip(axes, panels):
        ax.imshow(base, cmap=cmap, origin="lower", vmin=0 if title == "Model probability" else None,
                  vmax=1 if title == "Model probability" else None)
        if overlay is not None:
            ax.imshow(overlay, cmap=cmap, alpha=0.48, origin="lower")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.title("🧠 NeuroMask Lab")
st.write("An interactive walkthrough of a student-built U-Net for brain MRI tumor segmentation.")
st.warning(
    "This educational demo is not a medical device and must not be used for diagnosis or treatment. "
    "Use only research or sample scans you are allowed to upload."
)

with st.sidebar:
    st.header("Load a model and scan")
    checkpoint_file = st.file_uploader("1. Trained checkpoint (.pt)", type=["pt"])
    mri_file = st.file_uploader("2. Four-channel MRI (.nii or .nii.gz)", type=["nii", "gz"])
    reference_file = st.file_uploader(
        "Optional: matching tumor label (.nii or .nii.gz)", type=["nii", "gz"]
    )

overview_tab, demo_tab, steps_tab = st.tabs(["Overview", "Try the model", "What is happening?"])

with overview_tab:
    st.subheader("What this project does")
    st.write(
        "The model receives four aligned MRI sequences for one patient and predicts a binary mask: "
        "tumor versus background. It processes one axial slice at a time. It does not diagnose a tumor "
        "or identify the tumor's sub-regions."
    )
    cards = st.columns(3)
    cards[0].metric("Input", "4 MRI channels")
    cards[1].metric("Model", "2D U-Net")
    cards[2].metric("Output", "Whole-tumor mask")
    st.markdown(
        "**Expected channel order:** FLAIR · T1w · T1gd · T2w.\n\n"
        "**Example data:** `data/Task01_BrainTumour/imagesTr/BRATS_004.nii.gz`; "
        "the matching reference label is in `labelsTr/`."
    )
    st.caption(
        "The first student run used 20 cases, 5 epochs, and CPU. Its test Dice was 0.7279 and IoU was 0.6571; "
        "these are preliminary slice-averaged scores from only 3 held-out patients."
    )

with demo_tab:
    st.subheader("Run one slice through the model")
    st.markdown(
        "Upload the checkpoint created by training and a registered, skull-stripped, four-channel BraTS-style "
        "NIfTI scan. The optional label lets you compare the prediction with ground truth for that patient."
    )
    if not checkpoint_file or not mri_file:
        st.info("Add both required files in the sidebar to begin the walkthrough.")
    else:
        try:
            model, device = load_model(checkpoint_file.getvalue())
            raw_volume = load_nifti(mri_file)
            if raw_volume.ndim != 4 or raw_volume.shape[-1] != 4:
                raise ValueError(f"Expected [X, Y, Z, 4] MRI data; received shape {raw_volume.shape}.")
            reference_volume = None
            if reference_file:
                reference_volume = load_nifti(reference_file)
                if reference_volume.shape != raw_volume.shape[:3]:
                    raise ValueError(
                        f"Reference shape {reference_volume.shape} does not match MRI spatial shape "
                        f"{raw_volume.shape[:3]}. Upload the matching label file."
                    )
            volume = normalize_modalities(raw_volume)
        except Exception as exc:
            st.error(f"Could not load the checkpoint or scan: {exc}")
        else:
            st.success(f"Loaded {mri_file.name}: {raw_volume.shape[0]} × {raw_volume.shape[1]} × "
                       f"{raw_volume.shape[2]} voxels, with 4 channels. Inference device: {device}.")
            left, right = st.columns(2)
            with left:
                display_channel = st.selectbox("MRI channel to display", CHANNELS)
            with right:
                threshold = st.slider("Mask threshold", 0.10, 0.90, 0.50, 0.05)
            channel_index = CHANNELS.index(display_channel)
            if reference_volume is not None and reference_volume.any():
                default_slice = int(np.argmax((reference_volume > 0).sum(axis=(0, 1))))
            else:
                default_slice = raw_volume.shape[2] // 2
            slice_index = st.slider(
                "Axial slice (move through the volume)", 0, raw_volume.shape[2] - 1, default_slice
            )
            with st.expander("Step 1 · Prepare the MRI", expanded=False):
                st.write(
                    "For each channel, the app finds nonzero brain voxels, subtracts their mean, divides by "
                    "their standard deviation, clips values to −5…5, and keeps zero-valued background at zero. "
                    "This matches the training preprocessing."
                )
            st.code(
                f"Raw volume: {raw_volume.shape}\n"
                f"Model input slice: (batch=1, channels=4, height={raw_volume.shape[0]}, width={raw_volume.shape[1]})",
                language="text",
            )
            if st.button("Run segmentation on this slice", type="primary"):
                x = torch.from_numpy(
                    np.ascontiguousarray(volume[:, :, slice_index, :].transpose(2, 0, 1)[None, ...])
                ).to(device)
                with st.spinner("The U-Net is predicting this slice..."):
                    with torch.inference_mode():
                        logits = model(x)
                        probabilities = torch.sigmoid(logits)[0, 0].cpu().numpy()
                prediction = probabilities >= threshold
                st.markdown(
                    "### Step 2 · U-Net prediction\n"
                    "The encoder extracts patterns at several spatial scales. The decoder upsamples those "
                    "features and combines them with matching encoder features to recover pixel-level detail."
                )
                st.markdown(
                    "### Step 3 · Probability becomes a mask\n"
                    "A sigmoid converts each output logit to a tumor probability from 0 to 1. Pixels at or above "
                    f"the selected threshold (**{threshold:.2f}**) are colored as tumor."
                )
                show_slice(
                    raw_volume, slice_index, channel_index, prediction, probabilities,
                    (reference_volume[:, :, slice_index] > 0) if reference_volume is not None else None,
                )
                st.metric("Predicted tumor pixels on this slice", f"{int(prediction.sum()):,}")
                if reference_volume is not None:
                    reference_mask = reference_volume[:, :, slice_index] > 0
                    dice, iou = binary_scores(prediction, reference_mask)
                    m1, m2 = st.columns(2)
                    m1.metric("Slice Dice", f"{dice:.3f}")
                    m2.metric("Slice IoU", f"{iou:.3f}")
                    st.caption("These are metrics for the selected slice only, not the full patient volume.")
                png = (prediction.T.astype(np.uint8) * 255)
                from PIL import Image
                image_buffer = io.BytesIO()
                Image.fromarray(png).save(image_buffer, format="PNG")
                st.download_button(
                    "Download this 2D mask as PNG", image_buffer.getvalue(),
                    file_name=f"{Path(mri_file.name).stem}_slice_{slice_index}_mask.png",
                    mime="image/png",
                )

with steps_tab:
    st.subheader("The pipeline, one step at a time")
    with st.expander("1. Four MRI sequences provide different views", expanded=True):
        st.write(
            "FLAIR, T1-weighted, contrast-enhanced T1 (T1gd), and T2-weighted images are aligned voxel by voxel. "
            "The model sees all four together; the display selector changes only the background image shown to you."
        )
    with st.expander("2. Preprocessing makes intensities comparable"):
        st.write(
            "The same per-channel nonzero-voxel z-score normalization used during training is applied before inference. "
            "The app expects already aligned BraTS-style data; it does not register, skull-strip, or resample scans."
        )
    with st.expander("3. U-Net maps four input channels to one tumor score map"):
        st.write(
            "The encoder downsamples and learns broad spatial context. The decoder upsamples and uses skip connections "
            "from the encoder to retain fine boundaries. This project uses a compact 2D U-Net, so it sees one axial "
            "slice at a time rather than the full 3D neighborhood."
        )
    with st.expander("4. Sigmoid and threshold create a binary segmentation"):
        st.write(
            "The model emits one logit per pixel. Sigmoid turns it into a probability, and the threshold slider "
            "controls how confident a pixel must be before it is included in the predicted tumor mask."
        )
    with st.expander("5. Optional ground truth enables a slice comparison"):
        st.write(
            "Upload the matching file from `labelsTr/` to show the reference mask and compute Dice and IoU for the "
            "currently selected slice. The labels are converted to a binary whole-tumor mask by treating every "
            "nonzero label as tumor."
        )