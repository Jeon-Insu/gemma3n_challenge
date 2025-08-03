import uuid
from pathlib import Path
import av 
import streamlit as st
from aiortc.contrib.media import MediaRecorder
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import subprocess
import time

RECORD_DIR = Path("./recordings")
RECORD_DIR.mkdir(exist_ok=True)
IMAGE_DIR = RECORD_DIR / "images"
AUDIO_DIR = RECORD_DIR / "audio"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

def wait_for_file_complete(file_path, wait_seconds=5, check_interval=0.5):
    last_size = -1
    stable_count = 0
    required_stable_counts = int(wait_seconds / check_interval)
    while stable_count < required_stable_counts:
        if not file_path.exists():
            stable_count = 0
            last_size = -1
            time.sleep(check_interval)
            continue
        current_size = file_path.stat().st_size
        if current_size == last_size and current_size > 0:
            stable_count += 1
        else:
            stable_count = 0
            last_size = current_size
        time.sleep(check_interval)

def split_flv_to_audio_and_images(
        flv_path: str,
        audio_output_path : str,
        images_output_dir : Path,
        prefix: str,
        fps: int = 1,
        
):

    audio_cmd = [
        "ffmpeg", "-fflags", "+genpts", "-i", flv_path, "-vn", "-ar", "48000", "-ac", "2","-acodec", "pcm_s32le", audio_output_path, "-y"
    ]
    subprocess.run(audio_cmd, check=True)
    
    images_pattern = str(images_output_dir / f"{prefix}_frame_%04d.png")
    image_cmd = [
        "ffmpeg", "-i", flv_path, "-vf", f"fps={fps}", images_pattern, "-y",
    ]
    subprocess.run(image_cmd, check=True)

def main():

    st.title("WebRTC Video/Audio Recorder")

    if "prefix" not in st.session_state:
        st.session_state["prefix"] = str(uuid.uuid4())

    if "split_done" not in st.session_state:
        st.session_state["split_done"] = False

    prefix = st.session_state["prefix"]
    in_file = RECORD_DIR / f"{prefix}_input.flv"
    audio_output_path = AUDIO_DIR / f"{prefix}_audio.wav"
    

    def in_recorder_factory() -> MediaRecorder:
        return MediaRecorder(
            str(in_file), format="flv",
        ) 
    
    webrtc_streamer(
        key="record",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={
            "video" : True,
            "audio" : True,
        },
        in_recorder_factory=in_recorder_factory,
    )

    if in_file.exists() and not st.session_state["split_done"]:
        try:
            wait_for_file_complete(in_file)
            split_flv_to_audio_and_images(
                flv_path=str(in_file),
                audio_output_path=str(audio_output_path),
                images_output_dir=IMAGE_DIR,
                prefix=prefix
            )
            st.session_state["split_done"] = True
            st.success("🎉 Audio and image extraction completed automatically!")
        except subprocess.CalledProcessError as e:
            st.error("❌ ffmpeg error during splitting.")
            st.text(str(e))

    if in_file.exists():
        with in_file.open("rb") as f:
            st.download_button(
                "Download the recorded video", f, "video.flv"
            )

if __name__== "__main__":
    main()
