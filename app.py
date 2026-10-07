import streamlit as st
import subprocess
from pathlib import Path
import tempfile
import re
import sys

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Human Activity Recognition",
    page_icon="🧍",
    layout="centered"
)

# -----------------------------
# Project paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT = BASE_DIR / "checkpoints" / "best_model.pth"

CLASS_NAMES = [
    "falling",
    "jumping",
    "running",
    "sitting",
    "standing",
    "walking"
]

# -----------------------------
# Title
# -----------------------------
st.title("🧍 Human Activity Recognition")

st.write(
    "Upload a video and let the trained CNN3D model "
    "recognize the human activity."
)

st.divider()

# -----------------------------
# Upload video
# -----------------------------
uploaded_file = st.file_uploader(
    "🎥 Upload a video",
    type=["mp4", "avi", "mov", "mkv"]
)

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:

    st.video(uploaded_file)

    if st.button("🔍 Predict Activity"):

        st.info("Processing video...")

        # Save uploaded video temporarily
        suffix = Path(uploaded_file.name).suffix

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(uploaded_file.read())
            video_path = temp_file.name

        # Output video path
        output_video = str(
            Path(tempfile.gettempdir()) / "har_result.mp4"
        )

        # -----------------------------
        # Run inference.py
        # -----------------------------
        command = [
            sys.executable,
            str(BASE_DIR / "inference.py"),
            "--video",
            video_path,
            "--checkpoint",
            str(CHECKPOINT),
            "--class-names",
            ",".join(CLASS_NAMES),
            "--model-type",
            "cnn3d",
            "--num-frames",
            "16",
            "--frame-size",
            "224",
            "224",
            "--output-video",
            output_video
        ]

        try:

            result = subprocess.run(
                command,
                capture_output=True,
                text=True
            )

            # -----------------------------
            # Check for errors
            # -----------------------------
            if result.returncode != 0:

                st.error("❌ Prediction failed.")

                st.code(result.stderr)

            else:

                output = result.stdout

                # Show technical output
                with st.expander("View model output"):
                    st.code(output)

                # -----------------------------
                # Extract prediction
                # -----------------------------
                prediction_match = re.search(
                    r"Prediction:\s*(\w+)",
                    output
                )

                confidence_match = re.search(
                    r"Confidence:\s*([0-9.]+)",
                    output
                )

                if prediction_match:

                    prediction = prediction_match.group(1)

                    if confidence_match:
                        confidence = float(
                            confidence_match.group(1)
                        )
                    else:
                        confidence = 0.0

                    st.success("✅ Prediction completed!")

                    st.subheader("🎯 Result")

                    st.metric(
                        "Detected Activity",
                        prediction.upper()
                    )

                    st.metric(
                        "Confidence",
                        f"{confidence * 100:.2f}%"
                    )

                    # -----------------------------
                    # Probability display
                    # -----------------------------
                    st.subheader("📊 Activity Probabilities")

                    for class_name in CLASS_NAMES:

                        pattern = rf"{class_name}:\s*([0-9.]+)"

                        match = re.search(
                            pattern,
                            output
                        )

                        if match:

                            probability = float(
                                match.group(1)
                            )

                            st.write(
                                f"*{class_name.capitalize()}*"
                            )

                            st.progress(
                                min(probability, 1.0),
                                text=f"{probability * 100:.2f}%"
                            )

                    # -----------------------------
                    # Result video
                    # -----------------------------
                    if Path(output_video).exists():

                        st.subheader(
                            "🎬 Video with Prediction"
                        )

                        st.video(output_video)

                else:

                    st.warning(
                        "Prediction completed, "
                        "but the prediction text could not be detected."
                    )

        except Exception as e:

            st.error("❌ An error occurred.")

            st.exception(e)

        finally:

            # Remove temporary input video
            try:
                Path(video_path).unlink(
                    missing_ok=True
                )
            except Exception:
                pass       