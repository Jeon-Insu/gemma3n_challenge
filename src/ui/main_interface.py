"""
Main Streamlit interface for the multimodal chat application - New Design
"""
import streamlit as st
from typing import Dict, Any, Optional, List
import io
import os
import uuid
import tempfile
import time
from pathlib import Path
from PIL import Image
from datetime import datetime
import json
import base64
import subprocess
from src.app_controller import get_app_controller
from src.services.llm_iteration_task import LLMIterationTask

# Video recording imports
try:
    import av
    from aiortc.contrib.media import MediaRecorder
    from streamlit_webrtc import WebRtcMode, webrtc_streamer
    VIDEO_RECORDER_AVAILABLE = True
except ImportError:
    VIDEO_RECORDER_AVAILABLE = False
    st.warning("Video recording not available. Please install streamlit-webrtc, aiortc, and av.")

# Audio recording will be implemented using HTML5 Web Audio API
AUDIO_RECORDER_AVAILABLE = True


class MainInterface:
    """Main interface class for the Streamlit application"""
    
    def __init__(self):
        self.app_controller = get_app_controller()
        self.screening_task = LLMIterationTask()
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        # User state with UUID (use existing video_prefix as user_id)
        if 'user_id' not in st.session_state:
            # Use existing video_prefix as user_id to maintain consistency
            if 'video_prefix' in st.session_state:
                st.session_state.user_id = st.session_state.video_prefix
            else:
                import uuid
                st.session_state.user_id = str(uuid.uuid4())
        
        # Model state
        if 'model_initialized' not in st.session_state:
            st.session_state.model_initialized = False
        
        # Task management
        if 'task_list' not in st.session_state:
            st.session_state.task_list = []
        
        if 'current_task_index' not in st.session_state:
            st.session_state.current_task_index = 0
        
        if 'task_completion_status' not in st.session_state:
            st.session_state.task_completion_status = {}
        
        if 'screening_started' not in st.session_state:
            st.session_state.screening_started = False
        
        # Media data storage
        if 'captured_camera_image' not in st.session_state:
            st.session_state.captured_camera_image = None
        
        if 'recorded_audio' not in st.session_state:
            st.session_state.recorded_audio = None
        
        if 'uploaded_images' not in st.session_state:
            st.session_state.uploaded_images = {}
        
        # Video recording state (use user_id as video_prefix for consistency)
        if 'video_prefix' not in st.session_state:
            if 'user_id' in st.session_state:
                st.session_state.video_prefix = st.session_state.user_id
            else:
                import uuid
                st.session_state.video_prefix = str(uuid.uuid4())
        
        if 'video_split_done' not in st.session_state:
            st.session_state.video_split_done = False
        
        if 'recorded_video_path' not in st.session_state:
            st.session_state.recorded_video_path = None
        
        if 'video_audio_path' not in st.session_state:
            st.session_state.video_audio_path = None
        
        if 'video_images_dir' not in st.session_state:
            st.session_state.video_images_dir = None
        
        # Task results
        if 'task_results' not in st.session_state:
            st.session_state.task_results = {}
        
        # Processing state
        if 'processing_task_id' not in st.session_state:
            st.session_state.processing_task_id = None
        
        # UI state
        if 'show_load_batch' not in st.session_state:
            st.session_state.show_load_batch = False
        
        # Clean up old temporary audio files
        self.cleanup_temp_audio_files()
        
        # Create video recording directories
        self.setup_video_directories()
    
    def cleanup_temp_audio_files(self):
        """Clean up old temporary audio files and video streams"""
        try:
            import tempfile
            from pathlib import Path
            import time
            
            # Clean up audio files
            temp_dir = Path(tempfile.gettempdir()) / "streamlit_audio"
            if temp_dir.exists():
                current_time = time.time()
                for file_path in temp_dir.glob("*.wav"):
                    # Remove files older than 1 hour
                    if current_time - file_path.stat().st_mtime > 3600:
                        file_path.unlink()
            
            # Clean up old video recordings
            record_dir = Path("./recordings")
            if record_dir.exists():
                current_time = time.time()
                for file_path in record_dir.glob("*.flv"):
                    # Remove video files older than 24 hours
                    if current_time - file_path.stat().st_mtime > 86400:
                        file_path.unlink()
                        
        except Exception:
            pass  # Ignore cleanup errors
    
    def cleanup_video_processing_files(self):
        """Clean up video processing files (audio and images) after task completion"""
        try:
            from pathlib import Path
            import shutil
            
            # Get user ID
            user_id = st.session_state.get('user_id', 'default_user')
            user_record_dir = Path(f"./recordings/{user_id}")
            
            if user_record_dir.exists():
                # Clean up audio files
                audio_dir = user_record_dir / "audio"
                if audio_dir.exists():
                    for audio_file in audio_dir.glob("*.wav"):
                        try:
                            audio_file.unlink()
                            st.info(f"🗑️ Deleted audio file: {audio_file.name}")
                        except Exception as e:
                            st.warning(f"⚠️ Failed to delete audio file: {audio_file.name} - {str(e)}")
                
                # Clean up image files
                images_dir = user_record_dir / "images"
                if images_dir.exists():
                    for image_file in images_dir.glob("*.png"):
                        try:
                            image_file.unlink()
                            st.info(f"🗑️ Deleted image file: {image_file.name}")
                        except Exception as e:
                            st.warning(f"⚠️ Failed to delete image file: {image_file.name} - {str(e)}")
                
                # Clean up FLV files
                for flv_file in user_record_dir.glob("*.flv"):
                    try:
                        flv_file.unlink()
                        st.info(f"🗑️ Deleted video file: {flv_file.name}")
                    except Exception as e:
                        st.warning(f"⚠️ Failed to delete video file: {flv_file.name} - {str(e)}")
                
                # Clean up MP4 files
                for mp4_file in user_record_dir.glob("*.mp4"):
                    try:
                        mp4_file.unlink()
                        st.info(f"🗑️ Deleted MP4 file: {mp4_file.name}")
                    except Exception as e:
                        st.warning(f"⚠️ Failed to delete MP4 file: {mp4_file.name} - {str(e)}")
                
                st.success("✅ Video processing files cleanup completed!")
                
        except Exception as e:
            st.error(f"❌ Error occurred during video processing files cleanup: {str(e)}")
    
    def save_temp_audio(self, uploaded_audio, task_id, is_recorded=False):
        """Save audio to temporary directory"""
        try:
            from pathlib import Path
            
            # Create temp_audio directory in current working directory
            temp_audio_dir = Path.cwd() / "temp_audio"
            temp_audio_dir.mkdir(parents=True, exist_ok=True)
            
            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if is_recorded:
                filename = f"task_{task_id}_recorded_{timestamp}.wav"
            else:
                filename = f"task_{task_id}_{uploaded_audio.name}"
            
            temp_path = temp_audio_dir / filename
            
            # Save audio file
            with open(temp_path, "wb") as f:
                f.write(uploaded_audio.getvalue())
            
            return str(temp_path)
            
        except Exception as e:
            st.error(f"Failed to save audio to temp directory: {str(e)}")
            return None
    
    def setup_video_directories(self):
        """Setup directories for video recording"""
        try:
            # Get user ID (or create default)
            user_id = st.session_state.get('user_id', 'default_user')
            
            # Create user-specific recordings directory
            user_record_dir = Path(f"./recordings/{user_id}")
            user_record_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories for user
            image_dir = user_record_dir / "images"
            audio_dir = user_record_dir / "audio"
            
            image_dir.mkdir(parents=True, exist_ok=True)
            audio_dir.mkdir(parents=True, exist_ok=True)
            
        except Exception as e:
            st.error(f"Failed to setup video directories: {str(e)}")
    
    def wait_for_file_complete(self, file_path, wait_seconds=5, check_interval=0.5):
        """Wait for file to be completely written"""
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
    
    def split_flv_to_audio_and_images(self, flv_path: str, audio_output_path: str, 
                                     images_output_dir: Path, prefix: str, fps: int = 1):
        """Split FLV video to audio and images using ffmpeg"""
        try:
            # First, check if the video has audio stream
            probe_cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_streams", flv_path
            ]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
            
            has_audio = False
            if probe_result.returncode == 0:
                import json
                try:
                    streams_info = json.loads(probe_result.stdout)
                    for stream in streams_info.get('streams', []):
                        if stream.get('codec_type') == 'audio':
                            has_audio = True
                            break
                except json.JSONDecodeError:
                    pass
            
            # Extract audio only if audio stream exists
            if has_audio:
                audio_cmd = [
                    "ffmpeg", "-i", flv_path, "-vn", 
                    "-ar", "48000", "-ac", "2", "-acodec", "pcm_s16le", 
                    audio_output_path, "-y"
                ]
                
                # Run audio extraction with detailed error handling
                audio_result = subprocess.run(audio_cmd, capture_output=True, text=True)
                if audio_result.returncode != 0:
                    st.warning(f"⚠️ Audio extraction warning: {audio_result.stderr}")
                    # Try alternative audio extraction method
                    alt_audio_cmd = [
                        "ffmpeg", "-i", flv_path, "-vn", 
                        "-acodec", "pcm_s16le", audio_output_path, "-y"
                    ]
                    alt_result = subprocess.run(alt_audio_cmd, capture_output=True, text=True)
                    if alt_result.returncode != 0:
                        st.error(f"❌ Audio extraction failed: {alt_result.stderr}")
                        return False
                else:
                    st.success("✅ Audio extracted successfully!")
            else:
                st.warning("⚠️ No audio stream found in video. Creating silent audio file.")
                # Create a silent audio file
                silent_audio_cmd = [
                    "ffmpeg", "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
                    "-t", "10", "-acodec", "pcm_s16le", audio_output_path, "-y"
                ]
                silent_result = subprocess.run(silent_audio_cmd, capture_output=True, text=True)
                if silent_result.returncode != 0:
                    st.warning("⚠️ Could not create silent audio file. Continuing without audio.")
            
            # Extract images
            images_pattern = str(images_output_dir / f"{prefix}_frame_%04d.png")
            image_cmd = [
                "ffmpeg", "-i", flv_path, "-vf", f"fps={fps}", 
                images_pattern, "-y"
            ]
            
            # Run image extraction with detailed error handling
            image_result = subprocess.run(image_cmd, capture_output=True, text=True)
            if image_result.returncode != 0:
                st.warning(f"⚠️ Image extraction warning: {image_result.stderr}")
                # Try alternative image extraction method
                alt_image_cmd = [
                    "ffmpeg", "-i", flv_path, "-vf", f"fps={fps}", 
                    "-q:v", "2", images_pattern, "-y"
                ]
                alt_img_result = subprocess.run(alt_image_cmd, capture_output=True, text=True)
                if alt_img_result.returncode != 0:
                    st.error(f"❌ Image extraction failed: {alt_img_result.stderr}")
                    return False
            
            st.success("✅ Images extracted successfully!")
            return True
            
        except Exception as e:
            st.error(f"❌ ffmpeg error during splitting: {str(e)}")
            return False
    

    
    def render_header(self):
        """Render the application header"""
        st.title("V³-Gemma3n")

        st.markdown(
            '<div style="font-size: 18px;">A Multimodal Depression Screener Based on <span style="color: green; font-weight: bold; font-size: 20px;">V</span>isual, <span style="color: green; font-weight: bold; font-size: 20px;">V</span>ocal, and <span style="color: green; font-weight: bold; font-size: 20px;">V</span>erbal Signals</div>', 
            unsafe_allow_html=True
        )
        # st.markdown("---")
        
                        # Small camera/microphone permission request UI (Audio feedback prevention)
        import streamlit.components.v1 as components
        html_code = """
        <script>
        async function requestMediaAccess() {
          try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
              video: true, 
              audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
                sampleRate: 48000,
                channelCount: 2
              }
            });
            document.getElementById('status').innerText = "✅ Camera/Microphone access granted";
            document.getElementById('status').style.color = "green";
            stream.getTracks().forEach(track => track.stop());
          } catch (err) {
            document.getElementById('status').innerText = "❌ Access denied or error: " + err.message;
            document.getElementById('status').style.color = "red";
          }
        }
        
        // Additional settings to prevent audio feedback
        function preventAudioFeedback() {
          // Set volume to 0 for all audio elements
          const audioElements = document.querySelectorAll('audio, video');
          audioElements.forEach(audio => {
            audio.volume = 0;
            audio.muted = true;
          });
          
          // Disable audio output for WebRTC streams
          if (window.webrtcStreams) {
            window.webrtcStreams.forEach(stream => {
              stream.getAudioTracks().forEach(track => {
                track.enabled = false;
              });
            });
          }
        }
        
        window.addEventListener('load', () => {
          requestMediaAccess();
          preventAudioFeedback();
        });
        
        // Prevent audio feedback on every page change
        document.addEventListener('DOMContentLoaded', preventAudioFeedback);
        </script>

        <div style="font-size:12px; color:#666; margin:5px 0; padding:5px; background-color:#f8f9fa; border-radius:4px;">
        📷🎙️ <span id="status">Requesting permissions...</span>
        </div>
        """
        components.html(html_code, height=40)
        
        
    
    def render_model_initialization(self):
        """Render simplified model status"""
        # Model status indicator
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Model Status:**")
        
        with col2:
            if st.session_state.model_initialized:
                st.success("🟢 Model Ready")
            else:
                st.error("🔴 Model Not Loaded")
                st.info("👈 Model in sidebar")
    
    def render_task_tab(self):
        """Render the Task tab for video recording and upload"""
        
        # Fixed image section with Mission Guidelines
        st.markdown("##### 🖼️ Mission Image")
        
        # Mission Guidelines and image in a single centered column
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Mission Guidelines above the image
            st.markdown("""
            <div style="text-align: center; margin: 15px 0;">
                <h4 style="color: #1f77b4; font-weight: bold; font-size: 14px;">🎯 Mission Guidelines</h4>
                <p style="font-size: 12px; color: #666; margin: 8px 0;">
                    <strong>Please describe what you see in the image above.</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
            # Use the actual mission image file with center alignment
            image_path = "data/files/images/fire.gif"
            if os.path.exists(image_path):
                # Check if file is GIF and use appropriate MIME type
                if image_path.lower().endswith('.gif'):
                    mime_type = "image/gif"
                else:
                    mime_type = "image/png"
                
                st.markdown("""
                <div style="text-align: center;">
                    <img src="data:{};base64,{}" style="max-width: 250px; width: 100%; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                </div>
                """.format(mime_type, base64.b64encode(open(image_path, "rb").read()).decode()), unsafe_allow_html=True)
            else:
                # Fallback to styled placeholder if image not found
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 10px;
                    padding: 20px;
                    text-align: center;
                    color: white;
                    font-size: 18px;
                    font-weight: bold;
                    margin: 10px 0;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                ">
                    🎯 Mission Image
                </div>
                """, unsafe_allow_html=True)
                st.info("Please add mission image file to data/files/images/ directory.")
        
        st.markdown("---")
        
        # Video recording and upload section
        
        # Generate unique task ID for this session
        if 'task_session_id' not in st.session_state:
            st.session_state.task_session_id = str(uuid.uuid4())
        
        task_id = st.session_state.task_session_id
        
        # Render video interface
        self.render_video_interface_for_task(task_id)
        
        st.markdown("---")
        
        # Task completion status
        if st.session_state.get(f'video_split_done_{task_id}', False):
            st.success("✅ Video processing completed! You can now proceed to the Screening tab.")
        else:
            st.warning("⚠️ Please record or upload a video to proceed.")
    
    def render_screening_tab(self):
        """Render the Screening tab for task execution"""

        
        # Check if video is ready from Task tab
        task_session_id = st.session_state.get('task_session_id')
        video_ready = task_session_id and st.session_state.get(f'video_split_done_{task_session_id}', False)
        
        if not video_ready:
            st.warning("⚠️ Please complete video recording/upload in the Task tab first.")
            st.info("👆 Go to the Task tab to record or upload a video for screening.")
            return
        
        if not st.session_state.model_initialized:
            st.error("❌ Model not initialized. Please initialize the model in the sidebar first.")
            return
        
        # Execute button at the top
        st.markdown("##### 🚀 Execute Screening")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🎯 Execute All Questions", type="primary", use_container_width=True):
                # Create a default LLM iteration task if none exists
                # Update existing task list or create new one
                if not st.session_state.task_list:
                    from config.questions import get_all_question_keys
                    question_keys = get_all_question_keys()
                    
                    default_task = {
                        "id": 0,
                        "name": "Default Screening",
                        "explanation": "Screening Questions Auto Execution",
                        "type": "screening",
                        "question_keys": question_keys,
                        "nI": 1,
                        "has_video": True,
                        "has_user_text": False,
                        "has_camera": False,
                        "has_audio": False,
                        "has_image_upload": False,
                        "prompt_text": "Screening Questions",
                        "prompt_texts": ["Screening Questions"]
                    }
                    st.session_state.task_list = [default_task]
                else:
                    # Update existing tasks to use new type
                    for task in st.session_state.task_list:
                        if task.get('type') == 'llm_iteration':
                            task['type'] = 'screening'
                
                # Execute all tasks
                self.execute_all_tasks()
        
        st.markdown("---")
        
        # Add LLM Interaction Task section at the bottom
        st.markdown("##### 🤖 Add LLM Interaction Task")
        with st.expander("➕ Add New LLM Task", expanded=False):
            st.markdown("#### Add Screening Task")
            st.info("This task automatically executes all question sets using LLM model with images or audio extracted from video.")
            
                        # Task name for screening task
            screening_task_name = st.text_input("Screening Task Name", 
                                        placeholder="e.g., Screening Questions", 
                                        key="screening_task_name")
            
            # Question sets info
            st.markdown("#### Question Set Information")
            st.info("All 15 question sets will be executed automatically. Each question will be answered with 1 (Yes) or 0 (No).")
            
            # Import question sets
            from config.questions import get_all_question_keys, get_question_count
            
            question_keys = get_all_question_keys()
            total_questions = sum(get_question_count(key) for key in question_keys)
            
            st.info(f"**Total Question Sets:** {len(question_keys)} (Q1-Q15)")
            st.info(f"**Total Questions:** {total_questions}")
            
            # Show all question sets preview
            with st.expander("Preview All Question Sets", expanded=False):
                for key in question_keys:
                    from config.questions import get_question_set, get_question_data_type
                    questions = get_question_set(key)
                    data_type = get_question_data_type(key)
                    question_count = get_question_count(key)
                    
                    st.markdown(f"**{key}** ({data_type}, {question_count} questions):")
                    for i, q in enumerate(questions):
                        st.markdown(f"  - {q}")
                    st.markdown("---")
            
            # Iteration count
            nI = st.number_input("Iteration Count", 
                                min_value=1, 
                                max_value=20, 
                                value=1, 
                                help="Set how many times to repeat the questions (default: 1)",
                                key="screening_nI")
            
            # Add screening task button
            if st.button("🤖 Add Screening Task", type="primary", use_container_width=True):
                if screening_task_name:
                    # Create screening task
                    screening_task = {
                        "id": len(st.session_state.task_list),
                        "name": screening_task_name,
                        "explanation": f"Screening Task - Automatic execution of all question sets",
                        "type": "screening",
                        "question_keys": question_keys,  # All question keys
                        "nI": nI,
                        "has_video": True,  # Screening task requires video
                        "has_user_text": False,
                        "has_camera": False,
                        "has_audio": False,
                        "has_image_upload": False,
                        "prompt_text": "Screening Task",  # Placeholder
                        "prompt_texts": ["Screening Task"]  # Placeholder
                    }
                    
                    st.session_state.task_list.append(screening_task)
                    
                    st.success(f"✅ Screening Task '{screening_task_name}' added successfully! (All question sets will be executed automatically)")
                    st.rerun()
            else:
                    st.error("❌ Task name is required!")
    

    def render_video_interface_for_task(self, task_id):
        """Render video recording interface for a specific task"""
        if not VIDEO_RECORDER_AVAILABLE:
            st.error("❌ Video recording is not available. Please install required libraries.")
            return
        
        # Generate unique prefix for this recording session
        if f'video_prefix_{task_id}' not in st.session_state:
            st.session_state[f'video_prefix_{task_id}'] = str(uuid.uuid4())
        
        prefix = st.session_state[f'video_prefix_{task_id}']
        
        # Setup file paths with user-specific directory
        user_id = st.session_state.get('user_id', 'default_user')
        user_record_dir = Path(f"./recordings/{user_id}")
        in_file = user_record_dir / f"{prefix}_input.flv"
        audio_output_path = user_record_dir / "audio" / f"{prefix}_audio.wav"
        images_output_dir = user_record_dir / "images"
        
        # Record Video section
        # Create tabs for recording and upload options
        st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 40px;
            padding: 8px 16px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🎥 Record Video", "📁 Upload MP4"])
        
        with tab1:
            # Video recorder factory
            def in_recorder_factory():
                return MediaRecorder(str(in_file), format="flv")
            
            # WebRTC streamer with sendback_audio=False to prevent audio feedback
            st.markdown("""
            <style>
            .stWebRTC {
                max-width: 250px !important;
                width: 100% !important;
                border-radius: 8px !important;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            webrtc_streamer(
                key=f"video_record_{task_id}",
                mode=WebRtcMode.SENDRECV,
                media_stream_constraints={
                    "video": True,
                    "audio": True,
                },
                in_recorder_factory=in_recorder_factory,
                sendback_audio=False,  # Disable audio feedback
            )
            
            # Process video when file exists
            if in_file.exists() and not st.session_state.get(f'video_split_done_{task_id}', False):
                try:
                    # Processing video silently
                        self.wait_for_file_complete(in_file)
                        
                        success = self.split_flv_to_audio_and_images(
                            flv_path=str(in_file),
                            audio_output_path=str(audio_output_path),
                            images_output_dir=images_output_dir,
                            prefix=prefix
                        )
                        
                        if success:
                            st.session_state[f'video_split_done_{task_id}'] = True
                            st.session_state[f'recorded_video_path_{task_id}'] = str(in_file)
                            st.session_state[f'video_audio_path_{task_id}'] = str(audio_output_path)
                            st.session_state[f'video_images_dir_{task_id}'] = str(images_output_dir)
                            
                            # Video processing completed - no additional logs needed
                            
                except Exception as e:
                    st.error(f"❌ Error processing video: {str(e)}")
            
            # Show download option if video exists
            if in_file.exists():
                with in_file.open("rb") as f:
                    st.download_button(
                        "📥 Download Recorded Video",
                        f,
                        f"task_{task_id}_video.flv",
                        help="Download the recorded video file"
                    )
        
        with tab2:
            uploaded_file = st.file_uploader(
                "Choose an MP4 file",
                type=['mp4'],
                key=f"mp4_upload_{task_id}",
                help="Upload an MP4 video file (max 100MB)"
            )
            
            if uploaded_file is not None:
                try:
                    # Save uploaded MP4 file
                    mp4_path = user_record_dir / f"{prefix}_uploaded.mp4"
                    with open(mp4_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    # MP4 file uploaded - no additional logs needed
                    
                    # Process uploaded MP4 file
                    if not st.session_state.get(f'video_split_done_{task_id}', False):
                        # Processing video silently
                            # Convert MP4 to FLV for processing
                            flv_path = user_record_dir / f"{prefix}_converted.flv"
                            
                            # Try to use ffmpeg to convert MP4 to FLV with better error handling
                            import subprocess
                            try:
                                # First, try to get video info
                                probe_cmd = [
                                    'ffprobe', '-v', 'quiet', '-print_format', 'json',
                                    '-show_format', '-show_streams', str(mp4_path)
                                ]
                                probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
                                
                                if probe_result.returncode != 0:
                                    st.error("❌ Cannot read MP4 file. Please check if the file is valid.")
                                    return
                                
                                # Convert MP4 to FLV with more compatible settings
                                convert_cmd = [
                                    'ffmpeg', '-i', str(mp4_path), 
                                    '-c:v', 'libx264', '-preset', 'fast',
                                    '-c:a', 'aac', '-b:a', '128k',
                                    '-f', 'flv', str(flv_path)
                                ]
                                
                                result = subprocess.run(convert_cmd, capture_output=True, text=True, timeout=120)
                                
                                if result.returncode == 0:
                                # MP4 converted to FLV successfully - no additional logs needed
                                    
                                    # Process the converted FLV file
                                    success = self.split_flv_to_audio_and_images(
                                        flv_path=str(flv_path),
                                        audio_output_path=str(audio_output_path),
                                        images_output_dir=images_output_dir,
                                        prefix=prefix
                                    )
                                    
                                    if success:
                                        st.session_state[f'video_split_done_{task_id}'] = True
                                        st.session_state[f'recorded_video_path_{task_id}'] = str(flv_path)
                                        st.session_state[f'video_audio_path_{task_id}'] = str(audio_output_path)
                                        st.session_state[f'video_images_dir_{task_id}'] = str(images_output_dir)
                                        
                                    # Uploaded video processing completed - no additional logs needed
                                    
                                else:
                                    st.error(f"❌ Error converting MP4 to FLV: {result.stderr}")
                                    st.info("💡 Try with a different MP4 file or check if the file is corrupted.")
                                    
                            except subprocess.TimeoutExpired:
                                st.error("❌ Video conversion timed out. Please try with a shorter video.")
                            except FileNotFoundError:
                                st.error("❌ FFmpeg not found. Please install FFmpeg to process MP4 files.")
                                st.info("💡 To install FFmpeg:")
                                st.info("• Windows: Download from https://ffmpeg.org/download.html")
                                st.info("• macOS: brew install ffmpeg")
                                st.info("• Ubuntu/Debian: sudo apt install ffmpeg")
                            except Exception as e:
                                st.error(f"❌ Error processing uploaded video: {str(e)}")
                    
                except Exception as e:
                    st.error(f"❌ Error uploading file: {str(e)}")
        
        # Info message below the tabs
        st.info("Please record a video according to the mission or upload an existing video.", icon="ℹ️")
    
    def execute_current_task(self, task, user_text):
        """Execute the current task with the model"""
        task_id = task['id']
        
        try:
            # Set processing state
            st.session_state.processing_task_id = task_id
            
            # Check if this is a screening task
            if task.get('type') == 'screening':
                return self.execute_screening_task(task, user_text)
            
            # Get all prompts for this task
            prompt_texts = task.get('prompt_texts', [task['prompt_text']])
            total_prompts = len(prompt_texts)
            
            # Show processing message
            if total_prompts > 1:
                st.info(f"🔄 Processing {total_prompts} prompts for this task...")
            
            # Create progress containers for multiple prompts
            if total_prompts > 1:
                progress_container = st.empty()
                status_container = st.empty()
            
            # Process each prompt
            for prompt_idx, prompt_text in enumerate(prompt_texts):
                
                # Update progress for multiple prompts
                if total_prompts > 1:
                    progress = (prompt_idx + 1) / total_prompts
                    progress_container.progress(progress, text=f"Processing prompt {prompt_idx + 1}/{total_prompts}")
                    status_container.info(f"🔄 Executing prompt {prompt_idx + 1}: {prompt_text[:50]}...")
                
                with st.spinner(f"🔄 Processing prompt {prompt_idx + 1}/{total_prompts} (no history)..." if total_prompts > 1 else "🔄 Processing task with Gemma3n model (no history)..."):
                    full_prompt = prompt_text
                    if user_text:
                        full_prompt += f"\n\nUser Response: {user_text}"
                    
                    # Prepare multimodal content
                    content_items = []
                    
                    # Add text content
                    content_items.append({
                        "type": "text",
                        "text": full_prompt
                    })
                    
                    # Add camera image if available
                    if task['has_camera'] and st.session_state.captured_camera_image:
                        content_items.append({
                            "type": "image",
                            "image": st.session_state.captured_camera_image.getvalue(),
                            "filename": "camera_image.jpg"
                        })
                    
                    # Add pre-uploaded image if available
                    if task['has_image_upload'] and task['uploaded_image']:
                        content_items.append({
                            "type": "image",
                            "image": task['uploaded_image'],
                            "filename": task['uploaded_image_name']
                        })
                    
                    # Add audio if available
                    if task['has_audio']:
                        audio_key = f"task_audio_{task_id}"
                        audio_name_key = f"task_audio_name_{task_id}"
                        
                        if audio_key in st.session_state and st.session_state[audio_key]:
                            audio_filename = st.session_state.get(audio_name_key, "recorded_audio.wav")
                            content_items.append({
                                "type": "audio",
                                "audio": st.session_state[audio_key],
                                "filename": audio_filename
                            })
                    
                    # Add video data if available
                    if task['has_video']:
                        video_audio_path = st.session_state.get(f'video_audio_path_{task_id}')
                        video_images_dir = st.session_state.get(f'video_images_dir_{task_id}')
                        
                        # Add video audio if available
                        if video_audio_path and Path(video_audio_path).exists():
                            with open(video_audio_path, 'rb') as f:
                                audio_data = f.read()
                                content_items.append({
                                    "type": "audio",
                                    "audio": audio_data,
                                    "filename": f"video_audio_{task_id}.wav"
                                })
                        
                        # Add video images if available
                        if video_images_dir and Path(video_images_dir).exists():
                            image_files = list(Path(video_images_dir).glob(f"*_frame_*.png"))
                            for i, img_path in enumerate(image_files[:5]):  # Limit to first 5 frames
                                with open(img_path, 'rb') as f:
                                    image_data = f.read()
                                    content_items.append({
                                        "type": "image",
                                        "image": image_data,
                                        "filename": f"video_frame_{task_id}_{i+1}.png"
                                    })
                    
                    # Format multimodal input
                    from src.services.model_service import MultimodalInputFormatter
                    formatted_message = MultimodalInputFormatter.format_multimodal_input(content_items)
                    
                    # Call the model
                    if self.app_controller.is_model_ready():
                        max_tokens = st.session_state.get("max_tokens", 256)
                        response = self.app_controller.model_service.process_prompt(formatted_message, max_tokens)
                    else:
                        response = "❌ Model not ready. Please initialize the model first."
                    
                    # Store result with prompt index
                    result_key = f"{task_id}_{prompt_idx}" if len(prompt_texts) > 1 else str(task_id)
                    
                    # Get audio data for results
                    audio_key = f"task_audio_{task_id}"
                    audio_name_key = f"task_audio_name_{task_id}"
                    task_audio_data = st.session_state.get(audio_key)
                    task_audio_name = st.session_state.get(audio_name_key, "recorded_audio.wav")
                    
                    st.session_state.task_results[result_key] = {
                        "task_name": f"{task['name']}" + (f" (Prompt {prompt_idx + 1})" if len(prompt_texts) > 1 else ""),
                        "prompt": full_prompt,
                        "user_text": user_text,
                        "has_camera": task['has_camera'],
                        "has_audio": task['has_audio'],
                        "has_video": task['has_video'],
                        "has_image_upload": task['has_image_upload'],
                        "camera_image": st.session_state.captured_camera_image.getvalue() if (task['has_camera'] and st.session_state.captured_camera_image) else None,
                        "uploaded_image": task['uploaded_image'] if task['has_image_upload'] else None,
                        "uploaded_image_name": task['uploaded_image_name'] if task['has_image_upload'] else None,
                        "task_audio": task_audio_data,
                        "task_audio_name": task_audio_name if task_audio_data else None,
                        "video_path": st.session_state.get(f'recorded_video_path_{task_id}'),
                        "video_audio_path": st.session_state.get(f'video_audio_path_{task_id}'),
                        "video_images_dir": st.session_state.get(f'video_images_dir_{task_id}'),
                        "timestamp": str(datetime.now()),
                        "response": response,
                        "success": True,
                        "prompt_index": prompt_idx,
                        "total_prompts": len(prompt_texts)
                    }
            
            # Clear processing state
            st.session_state.processing_task_id = None
            
            # Clear progress containers
            if total_prompts > 1:
                progress_container.empty()
                status_container.empty()
            
            # Mark task as completed
            st.session_state.task_completion_status[task_id] = True
            
            prompt_count = len(prompt_texts)
            st.success(f"✅ Task executed successfully! Processed {prompt_count} prompt{'s' if prompt_count > 1 else ''}.")
            st.rerun()
            
        except Exception as e:
            # Clear processing state
            st.session_state.processing_task_id = None
            
            # Clear progress containers if they exist
            try:
                if 'progress_container' in locals():
                    progress_container.empty()
                if 'status_container' in locals():
                    status_container.empty()
            except:
                pass
            
            # Store error result
            st.session_state.task_results[str(task_id)] = {
                "task_name": task['name'],
                "prompt": task.get('prompt_text', ''),
                "user_text": user_text,
                "timestamp": str(datetime.now()),
                "response": None,
                "error": str(e),
                "success": False
            }
            
            st.error(f"❌ Error executing task: {str(e)}")
            st.rerun()
    
    def execute_screening_task(self, task, user_text):
        """Execute screening task"""
        task_id = task['id']
        
        try:
            # Use task_session_id from Task tab for video data
            task_session_id = st.session_state.get('task_session_id')
            if not task_session_id:
                st.error("❌ Task 탭에서 비디오를 먼저 녹화하거나 업로드해주세요.")
                return
            
            # Validate prerequisites using task_session_id
            validation = self.screening_task.validate_prerequisites(task_session_id)
            if not validation["valid"]:
                st.error("❌ 태스크 실행을 위한 조건이 충족되지 않았습니다:")
                for error in validation["errors"]:
                    st.error(f"  - {error}")
                return
            
            # Get model service
            model_service = self.app_controller.model_service
            if not model_service or not model_service.is_model_ready():
                st.error("❌ LLM 모델이 초기화되지 않았습니다. 사이드바에서 모델을 초기화해주세요.")
                return
            
            # Get the chat state (which has the send_message method)
            chat_state = model_service.chat_state
            if not chat_state:
                st.error("❌ LLM 모델이 초기화되지 않았습니다. 사이드바에서 모델을 초기화해주세요.")
                return
            
            # Get task parameters
            question_keys = task.get('question_keys', ['Q1'])
            nI = task.get('nI', 5)
            
            # Check if question keys are set
            if not question_keys:
                st.error("❌ 질문 키가 설정되지 않았습니다. 설정 탭에서 질문 세트를 선택해주세요.")
                return
            
            # Show processing message
            with st.spinner(f"🤖 Executing Screening Task... ({nI} iterations, {len(question_keys)} question sets)"):
                # Execute all question sets
                all_results = []
                all_logs = []
                
                for i, question_key in enumerate(question_keys):
                    st.info(f"📝 Processing: {question_key} ({i+1}/{len(question_keys)})")
                    
                    # Prepare task parameters
                    task_params = {
                        'nI': nI
                    }
                    
                    # Execute screening task for this question set
                    result = self.screening_task.execute_task(
                        model=chat_state,
                        task_id=task_session_id,  # Use task_session_id for video data
                        question_key=question_key,
                        task_params=task_params
                    )
                    
                    if "error" in result:
                        st.error(f"❌ {question_key} 실행 중 오류: {result['error']}")
                        continue
                    
                    all_results.append(result)
                    all_logs.extend(result.get('log', []))
                
                # Combine all results
                if all_results:
                    # Use the first result as base and combine logs
                    combined_result = all_results[0].copy()
                    combined_result['log'] = all_logs
                    combined_result['question_keys'] = question_keys
                    combined_result['total_question_sets'] = len(question_keys)
                    
                    # Update final result with combined data
                    if 'final_result' in combined_result:
                        final_result = combined_result['final_result']
                        final_result['total_question_sets'] = len(question_keys)
                        final_result['processed_question_keys'] = question_keys
                    
                    result = combined_result
                else:
                    st.error("❌ 모든 질문 세트 실행에 실패했습니다.")
                    return
                
                if "error" in result:
                    st.error(f"❌ {result['error']}")
                    return
                
                # Store result with response field
                # Extract LLM responses for response field
                log = result.get('log', [])
                if log:
                    # Group logs by question_key for proper formatting
                    question_set_logs = {}
                    for iter_log in log:
                        question_key = iter_log.get('question_key', 'Unknown')
                        if question_key not in question_set_logs:
                            question_set_logs[question_key] = []
                        question_set_logs[question_key].append(iter_log)
                    
                    response_summary = []
                    for question_key, logs in question_set_logs.items():
                        from config.questions import get_question_set
                        questions = get_question_set(question_key)
                        
                        for i, iter_log in enumerate(logs):
                            iteration_text = f"({question_key}) Iteration {iter_log.get('iteration', i+1)}:"
                            question_responses = []
                            
                            # Group questions by their original order
                            question_responses_dict = {}
                            for q in iter_log.get('questions', []):
                                question_text = q.get('question_text', '')
                                response = q.get('response', -1)
                                question_responses_dict[question_text] = response
                            
                            # Display in order with proper indexing
                            for j, question_text in enumerate(questions):
                                response = question_responses_dict.get(question_text, -1)
                                if len(questions) == 1:
                                    index_label = ""
                                else:
                                    index_label = f"{chr(97 + j)})"
                                question_responses.append(f"{index_label} {response}")
                            
                            response_summary.append(f"{iteration_text}\n" + "\n".join([f"- {resp}" for resp in question_responses]))
                    
                    # Add response field with detailed format
                    result['response'] = '\n\n'.join(response_summary)
                else:
                    result['response'] = "No LLM responses available"
                
                # Add success field if not present
                if 'success' not in result:
                    result['success'] = True
                
                st.session_state.task_results[task_id] = result
                st.session_state.task_completion_status[task_id] = True
                
                # Display results
                st.success("✅ Screening Task completed!")
                
                # Task completed successfully
                
        except Exception as e:
            st.error(f"❌ Error occurred during Screening Task execution: {str(e)}")
            st.session_state.processing_task_id = None
    
    def execute_screening_task_single(self, task, user_text):
        """Execute single screening task for one question set"""
        task_id = task['id']
        question_key = task.get('question_key', 'Q1')
        
        # Use task_session_id from Task tab for video data
        task_session_id = st.session_state.get('task_session_id')
        if not task_session_id:
            return {"error": "Task 탭에서 비디오를 먼저 녹화하거나 업로드해주세요."}
        
        # Validate prerequisites using task_session_id
        validation = self.screening_task.validate_prerequisites(task_session_id)
        if not validation["valid"]:
            return {"error": f"태스크 실행을 위한 조건이 충족되지 않았습니다: {validation['errors']}"}
        
        # Get model service
        model_service = self.app_controller.model_service
        if not model_service or not model_service.is_model_ready():
            return {"error": "LLM 모델이 초기화되지 않았습니다."}
        
        # Get the chat state (which has the send_message method)
        chat_state = model_service.chat_state
        if not chat_state:
            return {"error": "LLM 모델이 초기화되지 않았습니다."}
        
        # Get task parameters
        nI = task.get('nI', 1)
        
        # Prepare task parameters
        task_params = {
            'nI': nI
        }
        
        # Execute screening task for this question set
        try:
            result = self.screening_task.execute_task(
                model=chat_state,
                task_id=task_session_id,
                question_key=question_key,
                task_params=task_params
            )
            
            if "error" in result:
                return {"error": f"{question_key} 실행 중 오류: {result['error']}"}
            
            # Ensure result has required fields
            if 'status' not in result:
                result['status'] = 'completed'
            if 'success' not in result:
                result['success'] = True
            
            return result
            
        except Exception as e:
            return {
                "error": f"Error occurred during {question_key} execution: {str(e)}",
                "status": "failed",
                "success": False
            }
    
    def execute_all_tasks(self):
        """Execute all tasks sequentially"""
        if not st.session_state.task_list:
            st.error("No tasks to execute!")
            return
        
        if not self.app_controller.is_model_ready():
            st.error("❌ Model not initialized. Please initialize the model in the sidebar first.")
            return
        
        # Create progress containers
        progress_container = st.empty()
        status_container = st.empty()
        detail_container = st.empty()
        time_container = st.empty()
        
        total_tasks = len(st.session_state.task_list)
        start_time = time.time()
        
        try:
            for i, task in enumerate(st.session_state.task_list):
                task_id = task['id']
                
                # Update progress
                progress = (i + 1) / total_tasks
                progress_container.progress(progress, text=f"Executing Task {i + 1}/{total_tasks}: {task['name']}")
                status_container.info(f"🔄 Processing: {task['name']}")
                
                # Set processing state
                st.session_state.processing_task_id = task_id
                
                # Get user text for this task if it was entered
                user_text = st.session_state.get(f"user_text_{task_id}", "")
                
                # Execute the task with detailed progress
                self.execute_task_silently_with_progress(task, user_text, detail_container, time_container, start_time)
                
                # Mark as completed
                st.session_state.task_completion_status[task_id] = True
            
            # Clear processing state
            st.session_state.processing_task_id = None
            
            # Show completion message
            progress_container.empty()
            status_container.success("🎉 All tasks completed successfully!")
            detail_container.empty()
            time_container.empty()
            
            total_time = time.time() - start_time
            st.success(f"✅ Successfully executed {total_tasks} tasks in {total_time:.1f} seconds!")
            st.rerun()
            
        except Exception as e:
            st.session_state.processing_task_id = None
            progress_container.empty()
            status_container.error(f"❌ Error during batch execution: {str(e)}")
            detail_container.empty()
            time_container.empty()
            st.error(f"Batch execution failed: {str(e)}")
    
    def execute_task_silently(self, task, user_text):
        """Execute a task without UI updates (for batch execution)"""
        task_id = task['id']
        
        try:
            # Check if this is a screening task
            if task.get('type') == 'screening':
                # Use task_session_id for video data
                task_session_id = st.session_state.get('task_session_id')
                if not task_session_id:
                    st.error("❌ Task 탭에서 비디오를 먼저 녹화하거나 업로드해주세요.")
                    return
                
                # Execute all question sets for screening task
                question_keys = task.get('question_keys', [])
                nI = task.get('nI', 1)
                
                for question_key in question_keys:
                    # Check if this question set is already completed
                    result_key = f"{task_id}_{question_key}"
                    if result_key in st.session_state.task_results:
                        existing_result = st.session_state.task_results[result_key]
                        if existing_result.get('status') == 'completed':
                            continue
                    
                    # Create a temporary task for each question set
                    temp_task = task.copy()
                    temp_task['question_key'] = question_key
                    
                    # Execute the screening task for this question set
                    result = self.execute_screening_task_single(temp_task, user_text)
                    
                    # Store individual question set result
                    if result:
                        if 'status' in result and result['status'] == 'completed':
                            result_key = f"{task_id}_{question_key}"
                            st.session_state.task_results[result_key] = result
                        elif 'error' in result:
                            # Store error result to prevent infinite loop
                            result_key = f"{task_id}_{question_key}"
                            st.session_state.task_results[result_key] = {
                                "task_name": task['name'],
                                "question_key": question_key,
                                "status": "failed",
                                "error": result['error'],
                                "timestamp": str(datetime.now())
                            }
                    else:
                        # Store empty result to prevent infinite loop
                        result_key = f"{task_id}_{question_key}"
                        st.session_state.task_results[result_key] = {
                            "task_name": task['name'],
                            "question_key": question_key,
                            "status": "failed",
                            "error": "No result returned",
                            "timestamp": str(datetime.now())
                        }
                return
            
            # Skip non-screening tasks for now (only screening tasks are supported)
            st.warning(f"⚠️ Regular task '{task['name']}' is not currently supported. Only Screening Tasks are available.")
            return
            
        except Exception as e:
            # Store error result for screening tasks
            if task.get('type') == 'screening':
                st.session_state.task_results[str(task_id)] = {
                    "task_name": task['name'],
                    "type": "screening",
                    "timestamp": str(datetime.now()),
                    "error": str(e),
                    "success": False
                }
            else:
                st.error(f"❌ Task execution failed: {str(e)}")
    
    def execute_task_silently_with_progress(self, task, user_text, detail_container, time_container, start_time):
        """Execute a task with detailed progress updates"""
        task_id = task['id']
        
        try:
            # Check if this is a screening task
            if task.get('type') == 'screening':
                # Use task_session_id for video data
                task_session_id = st.session_state.get('task_session_id')
                if not task_session_id:
                    st.error("❌ Task 탭에서 비디오를 먼저 녹화하거나 업로드해주세요.")
                    return
                # Execute all question sets for screening task
                question_keys = task.get('question_keys', [])
                nI = task.get('nI', 1)
                
                # Import question sets for progress tracking
                from config.questions import get_all_question_keys, get_question_count
                
                if not question_keys:
                    question_keys = get_all_question_keys()
                
                total_questions = len(question_keys)
                completed_questions = 0
                
                detail_container.info(f"🤖 Starting LLM Iteration Task: {total_questions} question sets to process")
                
                for i, question_key in enumerate(question_keys):
                    # Check if this question set is already completed
                    result_key = f"{task_id}_{question_key}"
                    if result_key in st.session_state.task_results:
                        existing_result = st.session_state.task_results[result_key]
                        if existing_result.get('status') == 'completed':
                            detail_container.info(f"⏭️ {question_key} already completed, skipping...")
                            completed_questions += 1
                            continue
                    
                    # Update progress
                    completed_questions += 1
                    elapsed_time = time.time() - start_time
                    
                    # Calculate estimated time
                    if completed_questions > 0:
                        avg_time_per_question = elapsed_time / completed_questions
                        remaining_questions = total_questions - completed_questions
                        estimated_remaining = avg_time_per_question * remaining_questions
                        estimated_total = elapsed_time + estimated_remaining
                    else:
                        estimated_remaining = 0
                        estimated_total = 0
                    
                    # Update detail display
                    detail_container.info(
                        f"📊 Progress: {completed_questions}/{total_questions} question sets completed\n"
                        f"🔄 Currently processing: {question_key}\n"
                        f"⏱️ Elapsed time: {elapsed_time:.1f} seconds\n"
                        f"⏳ Estimated remaining time: {estimated_remaining:.1f} seconds\n"
                        f"🎯 Estimated total time: {estimated_total:.1f} seconds"
                    )
                    
                    # Update time display
                    time_container.info(
                        f"⏰ Time Information:\n"
                        f"• Elapsed: {elapsed_time:.1f} seconds\n"
                        f"• Remaining: {estimated_remaining:.1f} seconds\n"
                        f"• Total estimated: {estimated_total:.1f} seconds"
                    )
                    
                    # Create a temporary task for each question set
                    temp_task = task.copy()
                    temp_task['question_key'] = question_key
                    
                    # Execute the LLM iteration task for this question set
                    result = self.execute_screening_task_single(temp_task, user_text)
                    
                    # Store individual question set result
                    if result:
                        if 'status' in result and result['status'] == 'completed':
                            result_key = f"{task_id}_{question_key}"
                            st.session_state.task_results[result_key] = result
                            detail_container.success(f"✅ {question_key} completed successfully")
                        elif 'error' in result:
                            detail_container.error(f"❌ {question_key} failed: {result['error']}")
                            # Store error result to prevent infinite loop
                            result_key = f"{task_id}_{question_key}"
                            st.session_state.task_results[result_key] = {
                                "task_name": task['name'],
                                "question_key": question_key,
                                "status": "failed",
                                "error": result['error'],
                                "timestamp": str(datetime.now())
                            }
                    else:
                        detail_container.error(f"❌ {question_key} returned no result")
                        # Store empty result to prevent infinite loop
                        result_key = f"{task_id}_{question_key}"
                        st.session_state.task_results[result_key] = {
                            "task_name": task['name'],
                            "question_key": question_key,
                            "status": "failed",
                            "error": "No result returned",
                            "timestamp": str(datetime.now())
                        }
                    
                    # Small delay to update UI
                    time.sleep(0.1)
                
                detail_container.success(f"✅ 모든 질문 세트 완료! 총 {total_questions}개 질문 세트 처리 완료")
                time_container.empty()
                return
            
            # Skip non-LLM tasks for now (only LLM iteration tasks are supported)
            detail_container.warning(f"⚠️ Regular task '{task['name']}' is not currently supported. Only LLM Iteration Tasks are available.")
            return
            
        except Exception as e:
            detail_container.error(f"❌ Task execution failed: {str(e)}")
            time_container.error(f"⏰ 실행 중 오류 발생: {str(e)}")
    
    def render_sidebar(self):
        """Render the sidebar with configuration and management controls"""
        with st.sidebar:
            st.markdown("# ⚙️ Settings & Management")
            st.markdown("---")

            # User Configuration
            with st.expander("👤 User Configuration", expanded=True):
                st.markdown("### User Settings")
                
                # Display auto-generated user ID (same as video_prefix)
                user_id = st.session_state.get('user_id', st.session_state.get('video_prefix', 'Not set'))
                st.info(f"**User ID:** {user_id}")
                st.caption("비디오 녹화와 동일한 UUID를 사용합니다. 세션 내에서 일관된 식별자가 유지됩니다.")
                
                # Copy user ID button
                if st.button("📋 Copy User ID", use_container_width=True, key="sidebar_copy_user_btn"):
                    st.write("User ID copied to clipboard!")
                    st.code(user_id)
            
            # File Management
            with st.expander("🗑️ File Management", expanded=False):
                st.markdown("### Video Processing Files")
                st.caption("비디오 처리 후 생성된 오디오, 이미지, 비디오 파일들을 정리합니다.")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    if st.button("🧹 Clean Files", use_container_width=True, key="sidebar_clean_files_btn"):
                        with st.spinner("Cleaning files..."):
                            self.cleanup_video_processing_files()
                
                with col2:
                    if st.button("📊 Show Files", use_container_width=True, key="sidebar_show_files_btn"):
                        from pathlib import Path
                        user_id = st.session_state.get('user_id', 'default_user')
                        user_record_dir = Path(f"./recordings/{user_id}")
                        
                        if user_record_dir.exists():
                            st.info(f"📁 Directory: {user_record_dir}")
                            
                            # Count files
                            audio_files = list(user_record_dir.glob("audio/*.wav"))
                            image_files = list(user_record_dir.glob("images/*.png"))
                            video_files = list(user_record_dir.glob("*.flv")) + list(user_record_dir.glob("*.mp4"))
                            
                            st.write(f"🎵 Audio files: {len(audio_files)}")
                            st.write(f"🖼️ Image files: {len(image_files)}")
                            st.write(f"🎬 Video files: {len(video_files)}")
                        else:
                            st.info("📁 No video processing directory found.")
            
            # Model Configuration
            with st.expander("🤖 Model Configuration", expanded=True):
                st.markdown("### Model Setup")
                
                model_path = st.selectbox(
                    "Select Model",
                    options=[
                        "google/gemma-3n-E2B-it",
                        "google/gemma-3n-E4B-it"
                    ],
                    index=0,
                    help="Choose the Gemma 3n model variant to use",
                    key="sidebar_model_selectbox"
                )
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    if st.button("🚀 Initialize", disabled=st.session_state.model_initialized, use_container_width=True, key="sidebar_initialize_btn"):
                        with st.spinner("Loading model..."):
                            try:
                                success = self.app_controller.initialize_model(model_path)
                                if success:
                                    st.session_state.model_initialized = True
                                    st.success("Model initialized!")
                                    st.rerun()
                                else:
                                    st.error("Failed to initialize model.")
                            except Exception as e:
                                st.error(f"Failed: {str(e)}")
                
                with col2:
                    if st.button("🔄 Reset", disabled=not st.session_state.model_initialized, use_container_width=True, key="sidebar_reset_btn"):
                        with st.spinner("Resetting..."):
                            try:
                                success = self.app_controller.reset_model()
                                st.session_state.model_initialized = False
                                st.info("Please reinitialize to continue.")
                                st.rerun()
                            except Exception as e:
                                st.session_state.model_initialized = False
                                st.error(f"Error: {str(e)}")
                                st.rerun()
                
                # Model status
                if st.session_state.model_initialized:
                    st.success("🟢 Model Ready")
                else:
                    st.error("🔴 Model Not Loaded")
            
            # 카메라/오디오 허용 체크
            with st.expander("📷🎙️ Camera/Audio Permission", expanded=False):
                st.markdown("### Media Access Status")
                
                # Small camera/microphone permission request UI
                import streamlit.components.v1 as components
                html_code = """
                <script>
                async function requestMediaAccess() {
                  try {
                    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
                    document.getElementById('status').innerText = "✅ Camera/Microphone access granted";
                    document.getElementById('status').style.color = "green";
                    stream.getTracks().forEach(track => track.stop());
                  } catch (err) {
                    document.getElementById('status').innerText = "❌ Access denied or error: " + err.message;
                    document.getElementById('status').style.color = "red";
                  }
                }
                window.addEventListener('load', requestMediaAccess);
                </script>

                <div style="font-size:14px; color:#666; margin:10px 0; padding:10px; background-color:#f8f9fa; border-radius:4px;">
                📷🎙️ <span id="status">Requesting permissions...</span>
                </div>
                """
                components.html(html_code, height=60)
                
                st.info("💡 Camera/microphone access is only available in HTTPS environment.")
            
            
            
            # Model Cache Management
            with st.expander("💾 Model Cache Management", expanded=False):
                st.markdown("### Cached Models")
                
                try:
                    if not hasattr(self.app_controller, 'get_cache_stats'):
                        st.error("Cache management not available.")
                        if st.button("🔄 Restart App Controller", use_container_width=True, key="sidebar_restart_cache_btn"):
                            from src.app_controller import clear_app_controller_cache
                            clear_app_controller_cache()
                            st.success("Cache cleared. Please refresh.")
                            st.rerun()
                        return
                    
                    cache_stats = self.app_controller.get_cache_stats()
                    
                    if cache_stats.get("cache_exists", False):
                        st.metric("Cached Models", cache_stats.get("total_models", 0))
                        st.metric("Total Size", f"{cache_stats.get('total_size_mb', 0):.1f} MB")
                        
                        # Show cached models
                        cached_models = cache_stats.get("models", [])
                        if cached_models:
                            for model in cached_models:
                                status_icon = "✅" if model.get("is_valid", False) else "❌"
                                st.text(f"{status_icon} {model.get('model_name', 'Unknown')}")
                                st.text(f"   Size: {model.get('size_mb', 0):.1f} MB")
                        
                        # Clear all cache button
                        if st.button("🗑️ Clear All Cache", type="secondary", use_container_width=True, key="sidebar_clear_all_cache_btn"):
                            if self.app_controller.clear_model_cache():
                                st.success("All caches cleared!")
                                st.rerun()
                            else:
                                st.error("Failed to clear caches")
                    else:
                        st.info("No model cache found.")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            
            # Token Settings
            with st.expander("🔤 Token Settings", expanded=False):
                st.markdown("### Generation Settings")
                max_tokens = st.slider("Max Tokens", 50, 1000, 512, key="sidebar_max_tokens_slider")
                st.session_state.max_tokens = max_tokens
                st.info(f"Current: {max_tokens} tokens")
            
            # GPU Memory Management
            with st.expander("🖥️ GPU Memory Management", expanded=False):
                st.markdown("### GPU Status")
                
                try:
                    if not hasattr(self.app_controller, 'get_gpu_memory_info'):
                        st.error("GPU monitoring not available.")
                        if st.button("🔄 Restart for GPU Monitor", use_container_width=True, key="sidebar_restart_gpu_btn"):
                            from src.app_controller import clear_app_controller_cache
                            clear_app_controller_cache()
                            st.success("Cache cleared. Please refresh.")
                            st.rerun()
                        return
                    
                    memory_info = self.app_controller.get_gpu_memory_info()
                    
                    if memory_info.get("gpu_available", False):
                        st.metric("Total GPU Memory", f"{memory_info.get('total_memory', 0):.1f} GB")
                        st.metric("Used Memory", f"{memory_info.get('cached_memory', 0):.1f} GB")
                        st.metric("Free Memory", f"{memory_info.get('free_memory', 0):.1f} GB")
                        
                        # Memory usage bar
                        usage_percent = memory_info.get('memory_usage_percent', 0)
                        st.progress(usage_percent / 100, text=f"Usage: {usage_percent:.1f}%")
                        
                        # Force cleanup button
                        if st.button("🧹 Force GPU Cleanup", type="secondary", use_container_width=True, key="sidebar_force_gpu_cleanup_btn"):
                            with st.spinner("Cleaning GPU memory..."):
                                try:
                                    if hasattr(self.app_controller.model_service, 'force_cleanup_gpu_memory'):
                                        cleanup_results = self.app_controller.model_service.force_cleanup_gpu_memory()
                                        
                                        if cleanup_results["success"]:
                                            memory_freed = cleanup_results.get("memory_freed", 0)
                                            st.success(f"✅ Freed: {memory_freed:.2f} GB")
                                        else:
                                            st.error("❌ Cleanup failed")
                                        
                                        st.rerun()
                                    else:
                                        st.error("Cleanup function not available")
                                except Exception as cleanup_error:
                                    st.error(f"Error: {str(cleanup_error)}")
                    else:
                        st.info("🚫 No GPU detected")
                        try:
                            import torch
                            st.text(f"PyTorch: {torch.__version__}")
                            st.text(f"CUDA: {torch.cuda.is_available()}")
                        except:
                            st.text("PyTorch info unavailable")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            
            st.markdown("---")
            st.markdown("### 🔄 Quick Actions")
            
            if st.button("🔄 Refresh All", use_container_width=True, key="sidebar_refresh_all_btn"):
                st.rerun()
            
            if st.button("🧹 Clear App Cache", use_container_width=True, key="sidebar_clear_app_cache_btn"):
                try:
                    from src.app_controller import clear_app_controller_cache
                    clear_app_controller_cache()
                    st.success("App cache cleared!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to clear cache: {str(e)}")
    
    def render_results_tab(self):
        """Render the Results tab showing task execution results"""
        st.markdown("## 📊 Task Results")
        
        # Check if any tasks have been executed
        if not st.session_state.task_results:
            st.info("No results yet. Execute tasks in the Screening tab to see results here.")
            return
        
        # Results summary
        total_tasks = len(st.session_state.task_list)
        executed_tasks = len(st.session_state.task_results)
        successful_tasks = sum(1 for result in st.session_state.task_results.values() if result.get('success', False))
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Tasks", total_tasks)
        with col2:
            st.metric("Executed", executed_tasks)
        with col3:
            st.metric("Successful", successful_tasks)
        with col4:
            success_rate = (successful_tasks / executed_tasks * 100) if executed_tasks > 0 else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        # Show processing status if any task is being processed
        if st.session_state.processing_task_id is not None:
            processing_task = next((task for task in st.session_state.task_list if task['id'] == st.session_state.processing_task_id), None)
            if processing_task:
                st.warning(f"🔄 Currently processing: **{processing_task['name']}**")
        
        st.markdown("---")
        
        # Check if all question sets (Q1-Q15) are completed with safe condition
        required_question_keys = [f"Q{i}" for i in range(1, 16)]  # Q1 to Q15
        available_results = []
        
        for result_key, result_data in st.session_state.task_results.items():
            if result_data.get('status') == 'completed' and 'final_result' in result_data:
                question_key = result_data.get('question_key')
                if question_key in required_question_keys:
                    available_results.append(question_key)
        
        # Clear All Results button at the top
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🗑️ Clear All Results", type="secondary", use_container_width=True):
                # Clear task results
                st.session_state.task_results = {}
                st.session_state.task_completion_status = {}
                
                # Reset screening state
                st.session_state.screening_started = False
                st.session_state.current_task_index = 0
                st.session_state.processing_task_id = None
                
                st.success("✅ All results cleared and screening reset!")
                st.rerun()
        
        st.markdown("---")
        
        # Display Task Results in expander format
        for result_key, result_data in st.session_state.task_results.items():
            if result_data.get('status') == 'completed':
                task_name = result_data.get('task_name', 'Default Screening')
                task_id = result_data.get('task_id', 0)
                
                with st.expander(f"{task_name} (Task ID: {task_id})", expanded=False):
                    # Task completion status
                    st.success("✅ Task completed successfully")
                    
                    # Model Response section
                    st.markdown("**Model Response:**")
                    
                    # Response data in text area
                    response = result_data.get('response', '')
                    if response:
                        st.text_area("Response", value=response, height=200, disabled=True, label_visibility="collapsed")
                    
                    # Screening Scores section (separate expander)
                    if 'final_result' in result_data:
                        final_result = result_data['final_result']
                        if 'processed_question_keys' in final_result:
                            # Create all_results structure for screening calculation
                            # Collect all question results from session state
                            all_results = {}
                            for result_key, result in st.session_state.task_results.items():
                                if result.get('status') == 'completed' and 'question_key' in result:
                                    question_key = result['question_key']
                                    all_results[question_key] = result
                            
                            # Calculate screening scores
                            from src.services.llm_iteration_task import LLMIterationTask
                            screening_task = LLMIterationTask()
                            screening_scores = screening_task.calculate_screening_scores(all_results)
                            diagnosis = screening_task.calculate_diagnosis(all_results)
                            
                            # Display Screening Scores in separate expander
                            with st.expander("🧠 Screening Assessment", expanded=False):
                                st.markdown("#### 🧠 Screening Assessment")
                                st.markdown("**📋 Screening Items:**")
                                
                                screening_items = [
                                    ("Interest Loss", "interest_loss"),
                                    ("Depression", "depression"), 
                                    ("Sleep", "sleep"),
                                    ("Fatigue", "fatigue"),
                                    ("Appetite", "appetite"),
                                    ("Negative Thoughts", "negative_thoughts"),
                                    ("Concentration", "concentration"),
                                    ("Slowness", "slowness"),
                                    ("Suicidal Thoughts", "suicidal_thoughts")
                                ]
                                
                                # Display scores in 3-column layout with colored icons
                                col1, col2, col3 = st.columns(3)
                                
                                for i, (item_name, item_key) in enumerate(screening_items):
                                    score = screening_scores.get(item_key, 0)
                                    
                                    # Choose column based on index
                                    if i % 3 == 0:
                                        with col1:
                                            if score == 1:
                                                st.markdown(f"🔴 **{item_name}:** {score}", help=f"Score: {score}")
                                            else:
                                                st.markdown(f"🟢 **{item_name}:** {score}", help=f"Score: {score}")
                                    elif i % 3 == 1:
                                        with col2:
                                            if score == 1:
                                                st.markdown(f"🔴 **{item_name}:** {score}", help=f"Score: {score}")
                                            else:
                                                st.markdown(f"🟢 **{item_name}:** {score}", help=f"Score: {score}")
                                    else:
                                        with col3:
                                            if score == 1:
                                                st.markdown(f"🔴 **{item_name}:** {score}", help=f"Score: {score}")
                                            else:
                                                st.markdown(f"🟢 **{item_name}:** {score}", help=f"Score: {score}")
                            
                            st.markdown("---")
                            screening_total = sum(screening_scores.values())
                            st.markdown(f"**📊 Screening Total Score: {screening_total}/9**", help=f"Total score: {screening_total} out of 9")
                    

        
        # Export Results
        st.markdown("---")
        st.markdown("### 💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Export as JSON", use_container_width=True):
                self.export_results_as_json()
        
        with col2:
            if st.button("📝 Export as Text", use_container_width=True):
                self.export_results_as_text()

        

    
    def show_detailed_calculation_results(self, all_results, screening_scores, diagnosis):
        """Show detailed calculation results in a separate section"""
        st.markdown("---")
        st.markdown("## 🔍 Detailed Calculation Results")
        
        # Create tabs for different types of analysis
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Raw Results", "🧮 Screening Calculation", "🎯 Diagnosis Logic", "📋 Summary"])
        
        with tab1:
            st.markdown("### 📊 Raw Question Set Results")
            
            # Display raw results for each question set
            for question_key in sorted(all_results.keys()):
                result_data = all_results[question_key]
                final_result = result_data.get('final_result', {})
                
                with st.expander(f"**{question_key}** - {result_data.get('data_type', 'N/A')}", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Feature Scores:**")
                        feature_scores = final_result.get('feature_scores', {})
                        for feature, score in feature_scores.items():
                            st.markdown(f"• **{feature}:** {score}")
        
        with col2:
                        st.markdown("**Question Responses:**")
                        log = result_data.get('log', [])
                        if log:
                            for iter_log in log:
                                questions = iter_log.get('questions', [])
                                for q in questions:
                                    response = q.get('response', 'N/A')
                                    question_text = q.get('question_text', 'N/A')
                                    st.markdown(f"• **Q{q.get('question_num', 'N/A')}:** {response}")
        
        with tab2:
            st.markdown("### 🧮 Screening Score Calculation Details")
            
            # Show how each screening item was calculated
            screening_items = [
                ("Interest Loss", "interest_loss"),
                ("Depression", "depression"), 
                ("Sleep", "sleep"),
                ("Fatigue", "fatigue"),
                ("Appetite", "appetite"),
                ("Negative Thoughts", "negative_thoughts"),
                ("Concentration", "concentration"),
                ("Slowness", "slowness"),
                ("Suicidal Thoughts", "suicidal_thoughts")
            ]
            
            for item_name, item_key in screening_items:
                score = screening_scores.get(item_key, 0)
                with st.expander(f"**{item_name}** (Score: {score})", expanded=False):
                    # Show calculation logic for each item
                    if item_key == "interest_loss":
                        st.markdown("**Calculation Logic:** Q1 OR Q3 = 9 → Score = 1")
                    elif item_key == "depression":
                        st.markdown("**Calculation Logic:** image_score OR audio_score = 1 → Score = 1")
                    elif item_key == "sleep":
                        st.markdown("**Calculation Logic:** Fixed value = 0")
                    elif item_key == "fatigue":
                        st.markdown("**Calculation Logic:** Q3 OR Q4 OR Q8 OR Q9 = 9 → Score = 1")
                    elif item_key == "appetite":
                        st.markdown("**Calculation Logic:** Q6 = 9 → Score = 1")
                    elif item_key == "negative_thoughts":
                        st.markdown("**Calculation Logic:** text_feature != 'optimism' AND (Q1 OR Q7 = 9) → Score = 1")
                    elif item_key == "concentration":
                        st.markdown("**Calculation Logic:** Q12 OR Q13 = 9 → Score = 1")
                    elif item_key == "slowness":
                        st.markdown("**Calculation Logic:** Q9 = 9 → Score = 1")
                    elif item_key == "suicidal_thoughts":
                        st.markdown("**Calculation Logic:** Q15_result = 9 → Score = 1")
                    
                    # Show relevant question responses
                    st.markdown("**Relevant Question Responses:**")
                    # This would need to be implemented based on the specific logic
        
        with tab3:
            st.markdown("### 🎯 Diagnosis Logic Details")
            
            # Show the diagnosis calculation logic
            st.markdown("**Diagnosis Priority Logic:**")
            st.markdown("1. **text_feature == 'identification' OR Q15_result == 9** → **DEPRESSIVE**")
            st.markdown("2. **text_feature == 'optimism'** → **NORMAL**")
            st.markdown("3. **image_score == 1 AND audio_score == 1** → **DEPRESSIVE**")
            st.markdown("4. **image_score == 0 AND audio_score == 0** → **NORMAL**")
            st.markdown("5. **screening_total >= 4** → **DEPRESSIVE**")
            st.markdown("6. **Otherwise** → **NORMAL**")
            
            st.markdown("---")
            st.markdown(f"**Final Diagnosis:** {diagnosis.upper()}")
            
            # Show the values used in diagnosis
            st.markdown("**Values Used in Diagnosis:**")
            
            # Extract values from all_results
            text_feature = "N/A"
            Q15_result = "N/A"
            image_score = "N/A"
            audio_score = "N/A"
            
            for question_key, result_data in all_results.items():
                final_result = result_data.get('final_result', {})
                feature_scores = final_result.get('feature_scores', {})
                
                if question_key == "Q14":
                    text_feature = feature_scores.get('text_feature', 'N/A')
                elif question_key == "Q15":
                    Q15_result = feature_scores.get('Q15_result', 'N/A')
                
                # Get image_score and audio_score from any result
                if image_score == "N/A":
                    image_score = feature_scores.get('image_score', 'N/A')
                if audio_score == "N/A":
                    audio_score = feature_scores.get('audio_score', 'N/A')
            
            st.markdown(f"• **text_feature:** {text_feature}")
            st.markdown(f"• **Q15_result:** {Q15_result}")
            st.markdown(f"• **image_score:** {image_score}")
            st.markdown(f"• **audio_score:** {audio_score}")
            st.markdown(f"• **screening_total:** {sum(screening_scores.values())}")
        
        with tab4:
            st.markdown("### 📋 Complete Summary")
            
            # Overall summary
            st.markdown("**📊 Overall Results:**")
            st.markdown(f"• **Total Question Sets:** {len(all_results)}/15")
            st.markdown(f"• **Screening Total Score:** {sum(screening_scores.values())}/9")
            st.markdown(f"• **Final Diagnosis:** {diagnosis.upper()}")
            
            # Screening breakdown
            st.markdown("**🧮 Screening Breakdown:**")
            for item_name, item_key in screening_items:
                score = screening_scores.get(item_key, 0)
                status = "🔴 Positive" if score == 1 else "🟢 Negative"
                st.markdown(f"• **{item_name}:** {score} ({status})")
            
            # Question set completion status
            st.markdown("**📋 Question Set Status:**")
            completed_sets = sorted(all_results.keys())
            st.markdown(f"• **Completed:** {', '.join(completed_sets)}")
            
            # Export detailed results
            st.markdown("---")
            st.markdown("**💾 Export Detailed Results:**")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📄 Export Detailed JSON", key="export_detailed_json"):
                    self.export_detailed_results_as_json(all_results, screening_scores, diagnosis)
            
            with col2:
                if st.button("📝 Export Detailed Text", key="export_detailed_text"):
                    self.export_detailed_results_as_text(all_results, screening_scores, diagnosis)
    

    
    def render_screening_result(self, result, result_key):
        """Render screening task result"""
        
        # Success/Failure indicator
        if result.get('status') == 'completed':
            st.success("✅ LLM Iteration Task completed!")
        else:
            st.error("❌ LLM Iteration Task failed")
            if 'error' in result:
                st.error(f"Error: {result['error']}")
            return
        
        # Task details
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 📋 태스크 정보")
            
            # Show task details
            st.markdown(f"**Task Name:** {result.get('task_name', 'N/A')}")
            st.markdown(f"**Question Set:** {result.get('question_key', 'N/A')}")
            st.markdown(f"**Data Type:** {result.get('data_type', 'N/A')}")
            st.markdown(f"**Iteration Count:** {result.get('iterations', 'N/A')}")
            st.markdown(f"**Processed File:** {result.get('file_path', 'N/A')}")
            
            # Show questions
            questions = result.get('questions', [])
            if questions:
                st.markdown("**질문 목록:**")
                for i, q in enumerate(questions):
                    if q:  # Only show non-empty questions
                        st.markdown(f"• Q{i+1}: {q}")
        
        with col2:
            st.markdown("#### 📊 최종 분석 결과")
            
            # Show final processed result
            final_result = result.get('final_result', {})
            if final_result:
                # Summary statistics
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Total Iterations", final_result.get('total_iterations', 0))
                with col_b:
                    st.metric("Total Responses", final_result.get('total_responses', 0))
                with col_c:
                    st.metric("Question Count", final_result.get('total_questions', 0))
                
                # Response rates
                st.markdown("**Response Rates:**")
                response_rates = final_result.get('response_rates', {})
                for response, rate in response_rates.items():
                    percentage = rate * 100
                    if response == 9:
                        st.markdown(f"✅ Response 9: **{percentage:.1f}%**")
                    elif response == 0:
                        st.markdown(f"❌ Response 0: **{percentage:.1f}%**")
                    else:
                        st.markdown(f"⚠️ Response {response}: **{percentage:.1f}%**")
                
                # Analysis summary
                analysis = final_result.get('analysis_summary', {})
                if analysis:
                    st.markdown("**📋 Analysis Summary:**")
                    st.info(f"0 Response Rate: {analysis.get('zero_rate', 0):.1%}")
                    st.info(f"9 Response Rate: {analysis.get('nine_rate', 0):.1%}")
                
                # Feature scores
                feature_scores = final_result.get('feature_scores', {})
                st.markdown("**🎯 Feature Scores:**")
                
                if feature_scores:
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        image_feature = feature_scores.get('image_feature', 0)
                        st.metric("Image Feature", image_feature)
                    with col_b:
                        audio_feature = feature_scores.get('audio_feature', 0)
                        st.metric("Audio Feature", audio_feature)
                    with col_c:
                        text_feature = feature_scores.get('text_feature', '')
                        if text_feature:
                            st.metric("Text Feature", text_feature)
                        else:
                            st.metric("Text Feature", "N/A")
                    
                    # Q15 결과 표시
                    Q15_result = feature_scores.get('Q15_result')
                    if Q15_result is not None:
                        st.info(f"**Q15 Result:** {Q15_result}")
                    else:
                        st.info("**Q15 Result:** N/A")
                        
                    # 디버깅용 정보 표시
                    with st.expander("🔍 Debugging Information", expanded=False):
                        st.json(feature_scores)
                else:
                    st.warning("⚠️ No Feature Scores available.")
                    st.info("final_result content:")
                    st.json(final_result)
            else:
                st.warning("⚠️ No final analysis results available.")
        

        
        # Show detailed iteration results (collapsed by default)
        st.markdown("---")
        st.markdown("#### 🔍 Detailed Log View")
        
        log = result.get('log', [])
        if log:
            with st.expander("Detailed Execution Log", expanded=False):
                for i, iter_log in enumerate(log):
                    with st.expander(f"Iteration {i+1}", expanded=False):
                        st.markdown(f"**Iteration {iter_log['iteration']}:**")
                        
                        # Show frame info if available
                        if 'frame_index' in iter_log:
                            st.markdown(f"**Frame:** {iter_log['frame_index']}")
                        if 'frame_path' in iter_log:
                            st.markdown(f"**File:** {Path(iter_log['frame_path']).name}")
                        
                        for q in iter_log.get('questions', []):
                            question_text = q.get('question_text', 'N/A')
                            response = q.get('response', -1)
                            response_text = "Yes (9)" if response == 9 else "No (0)" if response == 0 else "Unknown"
                            
                            # Color coding for responses
                            if response == 9:
                                st.markdown(f"✅ **Q{q.get('question_num', '?')}:** {question_text}")
                                st.markdown(f"   **Answer:** {response_text}")
                            elif response == 0:
                                st.markdown(f"❌ **Q{q.get('question_num', '?')}:** {question_text}")
                                st.markdown(f"   **Answer:** {response_text}")
                                st.info("🛑 Skipping next question due to 'No' answer.")
                            else:
                                st.markdown(f"⚠️ **Q{q.get('question_num', '?')}:** {question_text}")
                                st.markdown(f"   **Answer:** {response_text}")
        else:
            st.warning("⚠️ No detailed result data available.")
    

    
    def save_batch_to_json(self):
        """Save current task list to JSON file"""
        try:
            import json
            import base64
            from datetime import datetime
            
            # Prepare data for JSON serialization
            batch_data = {
                "version": "1.0",
                "created_at": str(datetime.now()),
                "task_count": len(st.session_state.task_list),
                "tasks": []
            }
            
            for task in st.session_state.task_list:
                task_data = {
                    "id": task["id"],
                    "name": task["name"],
                    "explanation": task.get("explanation", ""),
                    "prompt_text": task["prompt_text"],
                    "prompt_texts": task.get("prompt_texts", [task["prompt_text"]]),
                    "has_user_text": task["has_user_text"],
                    "has_camera": task["has_camera"],
                    "has_audio": task["has_audio"],
                    "has_video": task["has_video"],
                    "has_image_upload": task["has_image_upload"],
                    "uploaded_image_name": task.get("uploaded_image_name")
                }
                
                # Convert binary image data to base64 for JSON serialization
                if task.get("uploaded_image"):
                    task_data["uploaded_image_base64"] = base64.b64encode(task["uploaded_image"]).decode('utf-8')
                
                batch_data["tasks"].append(task_data)
            
            # Convert to JSON string
            json_str = json.dumps(batch_data, indent=2, ensure_ascii=False)
            
            # Create download button
            st.download_button(
                label="📥 Download Batch JSON",
                data=json_str,
                file_name=f"task_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
            
            st.success(f"✅ Batch with {len(st.session_state.task_list)} tasks ready for download!")
            
        except Exception as e:
            st.error(f"❌ Error saving batch: {str(e)}")
    
    def load_batch_from_json(self):
        """Load task list from JSON file"""
        st.markdown("---")
        st.markdown("### 📥 Load Batch from JSON")
        
        # Close button
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("❌ Close", key="close_load_batch"):
                st.session_state.show_load_batch = False
                st.rerun()
        
        uploaded_file = st.file_uploader(
            "Choose a JSON batch file",
            type=['json'],
            help="Upload a previously saved task batch JSON file",
            key="batch_file_uploader"
        )
        
        if uploaded_file is not None:
            try:
                # Read and parse JSON
                json_str = uploaded_file.read().decode('utf-8')
                batch_data = json.loads(json_str)
                
                # Validate JSON structure
                if "tasks" not in batch_data:
                    st.error("❌ Invalid batch file format: missing 'tasks' field")
                    return
                
                # Show batch info
                st.info(f"📊 Batch Info:")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Version", batch_data.get("version", "Unknown"))
                with col2:
                    st.metric("Task Count", batch_data.get("task_count", len(batch_data["tasks"])))
                with col3:
                    created_at = batch_data.get("created_at", "Unknown")
                    if created_at != "Unknown":
                        try:
                            created_date = datetime.fromisoformat(created_at).strftime("%Y-%m-%d %H:%M")
                            st.metric("Created", created_date)
                        except:
                            st.metric("Created", "Unknown")
                    else:
                        st.metric("Created", "Unknown")
                
                # Show task preview
                st.markdown("**Tasks to be loaded:**")
                for i, task_data in enumerate(batch_data["tasks"]):
                    prompt_count = len(task_data.get("prompt_texts", [task_data.get("prompt_text", "")]))
                    components = []
                    if task_data.get("has_user_text"): components.append("User Text")
                    if task_data.get("has_camera"): components.append("Camera")
                    if task_data.get("has_audio"): components.append("Audio")
                    if task_data.get("has_video"): components.append("Video")
                    if task_data.get("has_image_upload"): components.append("Image Upload")
                    
                    st.markdown(f"**{i+1}. {task_data.get('name', 'Unnamed Task')}**")
                    st.markdown(f"   - Prompts: {prompt_count}")
                    st.markdown(f"   - Components: {', '.join(components) if components else 'Prompt Text Only'}")
                
                # Load options
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🔄 Replace Current Tasks", type="primary", use_container_width=True):
                        self.load_tasks_from_data(batch_data, replace=True)
                
                with col2:
                    if st.button("➕ Add to Current Tasks", use_container_width=True):
                        self.load_tasks_from_data(batch_data, replace=False)
                
            except json.JSONDecodeError as e:
                st.error(f"❌ Invalid JSON file: {str(e)}")
            except Exception as e:
                st.error(f"❌ Error loading batch: {str(e)}")
    
    def load_tasks_from_data(self, batch_data, replace=True):
        """Load tasks from batch data"""
        try:
            if replace:
                st.session_state.task_list = []
                st.session_state.task_completion_status = {}
                st.session_state.task_results = {}
            
            # Reset screening state when task list is updated
            st.session_state.screening_started = False
            st.session_state.current_task_index = 0
            st.session_state.processing_task_id = None
            
            # Load tasks
            loaded_count = 0
            for task_data in batch_data["tasks"]:
                new_task = {
                    "id": len(st.session_state.task_list),  # Reassign ID to avoid conflicts
                    "name": task_data.get("name", "Unnamed Task"),
                    "explanation": task_data.get("explanation", ""),
                    "prompt_text": task_data.get("prompt_text", ""),
                    "prompt_texts": task_data.get("prompt_texts", [task_data.get("prompt_text", "")]),
                    "has_user_text": task_data.get("has_user_text", False),
                    "has_camera": task_data.get("has_camera", False),
                    "has_audio": task_data.get("has_audio", False),
                    "has_video": task_data.get("has_video", False),
                    "has_image_upload": task_data.get("has_image_upload", False),
                    "uploaded_image_name": task_data.get("uploaded_image_name"),
                    "uploaded_image": None
                }
                
                # Decode base64 image data if present
                if task_data.get("uploaded_image_base64"):
                    try:
                        new_task["uploaded_image"] = base64.b64decode(task_data["uploaded_image_base64"])
                    except Exception as e:
                        st.warning(f"⚠️ Could not load image for task '{new_task['name']}': {str(e)}")
                
                st.session_state.task_list.append(new_task)
                loaded_count += 1
            
            # Clear current prompt texts
            st.session_state.current_prompt_texts = [""]
            
            # Close load batch interface
            st.session_state.show_load_batch = False
            
            action = "replaced" if replace else "added"
            st.success(f"✅ Successfully {action} {loaded_count} tasks!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error loading tasks: {str(e)}")
    
    def run(self):
        """Main method to run the interface"""
        # Configure page
        st.set_page_config(
            page_title="Screening",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Render components
        self.render_sidebar()  # Render sidebar first
        self.render_header()
        self.render_model_initialization()
        
        # Main content with tabs
        # Custom CSS to make tabs larger and more prominent
        st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 60px;
            padding: 0px 24px;
            background-color: #f0f2f6;
            border-radius: 8px 8px 0px 0px;
            font-size: 18px;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background-color: #ffffff;
            border-bottom: 3px solid #ff6b6b;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Create tabs with larger size
        tab1, tab2, tab3 = st.tabs(["📋 Task", "Screening", "📊 Results"])
        
        with tab1:
            self.render_task_tab()
        
        with tab2:
            self.render_screening_tab()
        
        with tab3:
            self.render_results_tab()
    

    
    def export_results_as_json(self):
        """Export results as JSON"""
        try:
            import json
            import base64
            from datetime import datetime
            
            # Create a JSON-serializable copy of task_results
            serializable_task_results = {}
            for task_id, result in st.session_state.task_results.items():
                serializable_result = result.copy()
                
                # Handle LLM iteration task results specially
                if result.get('task_name') == 'LLM Iteration Task':
                    # Extract LLM responses and create a proper response field
                    log = result.get('log', [])
                    if log:
                        # Group logs by question_key for proper formatting
                        question_set_logs = {}
                        for iter_log in log:
                            question_key = iter_log.get('question_key', 'Unknown')
                            if question_key not in question_set_logs:
                                question_set_logs[question_key] = []
                            question_set_logs[question_key].append(iter_log)
                        
                        response_summary = []
                        for question_key, logs in question_set_logs.items():
                            from config.questions import get_question_set
                            questions = get_question_set(question_key)
                            
                            for i, iter_log in enumerate(logs):
                                iteration_text = f"({question_key}) Iteration {iter_log.get('iteration', i+1)}:"
                                question_responses = []
                                
                                # Group questions by their original order
                                question_responses_dict = {}
                                for q in iter_log.get('questions', []):
                                    question_text = q.get('question_text', '')
                                    response = q.get('response', -1)
                                    question_responses_dict[question_text] = response
                                
                                # Display in order with proper indexing
                                for j, question_text in enumerate(questions):
                                    response = question_responses_dict.get(question_text, -1)
                                    if len(questions) == 1:
                                        index_label = ""
                                    else:
                                        index_label = f"{chr(97 + j)})"
                                    question_responses.append(f"{index_label} {response}")
                                
                                response_summary.append(f"{iteration_text}\n" + "\n".join([f"- {resp}" for resp in question_responses]))
                        
                        # Add response field with detailed format
                        serializable_result['response'] = '\n\n'.join(response_summary)
                    else:
                        serializable_result['response'] = "No LLM responses available"
                    
                    # Add execution time and timestamp if not present
                    if 'execution_time' not in serializable_result:
                        serializable_result['execution_time'] = 0.0
                    if 'timestamp' not in serializable_result:
                        serializable_result['timestamp'] = datetime.now().isoformat()
                    if 'success' not in serializable_result:
                        serializable_result['success'] = True
                
                # Convert any bytes objects to base64 strings
                for key, value in serializable_result.items():
                    if isinstance(value, bytes):
                        serializable_result[key] = base64.b64encode(value).decode('utf-8')
                    elif isinstance(value, dict):
                        # Handle nested dictionaries
                        for nested_key, nested_value in value.items():
                            if isinstance(nested_value, bytes):
                                value[nested_key] = base64.b64encode(nested_value).decode('utf-8')
                
                serializable_task_results[task_id] = serializable_result
            
            # Create a JSON-serializable copy of task_list
            serializable_task_list = []
            for task in st.session_state.task_list:
                serializable_task = task.copy()
                
                # Convert uploaded_image bytes to base64 if present
                if 'uploaded_image' in serializable_task and isinstance(serializable_task['uploaded_image'], bytes):
                    serializable_task['uploaded_image'] = base64.b64encode(serializable_task['uploaded_image']).decode('utf-8')
                
                serializable_task_list.append(serializable_task)
            
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "task_results": serializable_task_results,
                "task_list": serializable_task_list,
                "completion_status": st.session_state.task_completion_status
            }
            
            json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
            
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=f"screening_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
        except Exception as e:
            st.error(f"Failed to export JSON: {str(e)}")
    
    def export_detailed_results_as_json(self, all_results, screening_scores, diagnosis):
        """Export detailed calculation results as JSON"""
        try:
            import json
            from datetime import datetime
            
            # Create detailed export data
            detailed_data = {
                "export_timestamp": datetime.now().isoformat(),
                "export_type": "detailed_calculation_results",
                "all_results": all_results,
                "screening_scores": screening_scores,
                "diagnosis": diagnosis,
                "screening_total": sum(screening_scores.values()),
                "completed_question_sets": list(all_results.keys()),
                "total_question_sets": len(all_results)
            }
            
            json_str = json.dumps(detailed_data, indent=2, ensure_ascii=False)
            
            st.download_button(
                label="📥 Download Detailed JSON",
                data=json_str,
                file_name=f"detailed_calculation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
        except Exception as e:
            st.error(f"Failed to export detailed JSON: {str(e)}")
    
    def export_detailed_results_as_text(self, all_results, screening_scores, diagnosis):
        """Export detailed calculation results as text"""
        try:
            from datetime import datetime
            
            text_content = f"Detailed Calculation Results\n"
            text_content += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            text_content += "=" * 50 + "\n\n"
            
            # Overall summary
            text_content += f"OVERALL SUMMARY\n"
            text_content += f"Total Question Sets: {len(all_results)}/15\n"
            text_content += f"Screening Total Score: {sum(screening_scores.values())}/9\n"
            text_content += f"Final Diagnosis: {diagnosis.upper()}\n\n"
            
            # Screening scores
            text_content += f"SCREENING SCORES\n"
            screening_items = [
                ("Interest Loss", "interest_loss"),
                ("Depression", "depression"), 
                ("Sleep", "sleep"),
                ("Fatigue", "fatigue"),
                ("Appetite", "appetite"),
                ("Negative Thoughts", "negative_thoughts"),
                ("Concentration", "concentration"),
                ("Slowness", "slowness"),
                ("Suicidal Thoughts", "suicidal_thoughts")
            ]
            
            for item_name, item_key in screening_items:
                score = screening_scores.get(item_key, 0)
                text_content += f"{item_name}: {score}\n"
            
            text_content += "\n"
            
            # Raw results
            text_content += f"RAW QUESTION SET RESULTS\n"
            for question_key in sorted(all_results.keys()):
                result_data = all_results[question_key]
                final_result = result_data.get('final_result', {})
                feature_scores = final_result.get('feature_scores', {})
                
                text_content += f"\n{question_key}:\n"
                text_content += f"  Data Type: {result_data.get('data_type', 'N/A')}\n"
                text_content += f"  Feature Scores:\n"
                for feature, score in feature_scores.items():
                    text_content += f"    {feature}: {score}\n"
                
                # Question responses
                log = result_data.get('log', [])
                if log:
                    text_content += f"  Question Responses:\n"
                    for iter_log in log:
                        questions = iter_log.get('questions', [])
                        for q in questions:
                            response = q.get('response', 'N/A')
                            text_content += f"    Q{q.get('question_num', 'N/A')}: {response}\n"
            
            st.download_button(
                label="📥 Download Detailed Text",
                data=text_content,
                file_name=f"detailed_calculation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
            
        except Exception as e:
            st.error(f"Failed to export detailed text: {str(e)}")
    
    def export_results_as_text(self):
        """Export results as text"""
        try:
            from datetime import datetime
            
            text_content = f"Screening Results\n"
            text_content += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            text_content += "=" * 50 + "\n\n"
            
            for task_id, result in st.session_state.task_results.items():
                # Find task name
                task_name = "Unknown Task"
                for task in st.session_state.task_list:
                    if task['id'] == task_id:
                        task_name = task['name']
                        break
                
                text_content += f"Task: {task_name} (ID: {task_id})\n"
                text_content += "-" * 30 + "\n"
                
                # Check if this is an LLM iteration task
                if result.get('task_name') == 'LLM Iteration Task':
                    text_content += f"Status: SUCCESS\n"
                    
                    # Process LLM iteration results with detailed format
                    log = result.get('log', [])
                    if log:
                        # Group logs by question_key for proper formatting
                        question_set_logs = {}
                        for iter_log in log:
                            question_key = iter_log.get('question_key', 'Unknown')
                            if question_key not in question_set_logs:
                                question_set_logs[question_key] = []
                            question_set_logs[question_key].append(iter_log)
                        
                        text_content += f"LLM 반복 질문 결과:\n"
                        for question_key, logs in question_set_logs.items():
                            from config.questions import get_question_set
                            questions = get_question_set(question_key)
                            
                            for i, iter_log in enumerate(logs):
                                text_content += f"  ({question_key}) Iteration {iter_log.get('iteration', i+1)}:\n"
                                
                                # Group questions by their original order
                                question_responses_dict = {}
                                for q in iter_log.get('questions', []):
                                    question_text = q.get('question_text', '')
                                    response = q.get('response', -1)
                                    question_responses_dict[question_text] = response
                                
                                # Display in order with proper indexing
                                for j, question_text in enumerate(questions):
                                    response = question_responses_dict.get(question_text, -1)
                                    if len(questions) == 1:
                                        index_label = ""
                                    else:
                                        index_label = f"{chr(97 + j)})"
                                    text_content += f"    - {index_label} {response}\n"
                    else:
                        text_content += f"Response: No LLM responses available\n"
                    
                    # Show final result if available
                    final_result = result.get('final_result', {})
                    if final_result:
                        text_content += f"최종 분석 결과:\n"
                        feature_scores = final_result.get('feature_scores', {})
                        if feature_scores:
                            text_content += f"  Image Feature: {feature_scores.get('image_feature', 0)}\n"
                            text_content += f"  Audio Feature: {feature_scores.get('audio_feature', 0)}\n"
                            text_content += f"  Text Feature: {feature_scores.get('text_feature', 'N/A')}\n"
                            text_content += f"  Q15 Result: {feature_scores.get('Q15_result', 'N/A')}\n"
                    
                    if 'execution_time' in result:
                        text_content += f"Execution Time: {result['execution_time']:.2f} seconds\n"
                    
                    if 'timestamp' in result:
                        text_content += f"Timestamp: {result['timestamp']}\n"
                else:
                    # Handle other task types
                    if result.get('success', False):
                        text_content += f"Status: SUCCESS\n"
                        text_content += f"Response: {result.get('response', 'No response')}\n"
                        
                        if 'execution_time' in result:
                            text_content += f"Execution Time: {result['execution_time']:.2f} seconds\n"
                        
                        if 'timestamp' in result:
                            text_content += f"Timestamp: {result['timestamp']}\n"
                    else:
                        text_content += f"Status: FAILED\n"
                        text_content += f"Error: {result.get('error', 'Unknown error')}\n"
                
                text_content += "\n"
            
            st.download_button(
                label="📥 Download Text",
                data=text_content,
                file_name=f"screening_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
            
        except Exception as e:
            st.error(f"Failed to export text: {str(e)}")
    

    
    def execute_all_tasks(self):
        """Execute all tasks in sequence"""
        try:
            if not st.session_state.model_initialized:
                st.error("❌ Model not initialized. Please initialize the model first.")
                return
            
            if not st.session_state.task_list:
                st.error("❌ No tasks to execute.")
                return
            
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_tasks = len(st.session_state.task_list)
            
            for i, task in enumerate(st.session_state.task_list):
                task_id = task['id']
                
                # Update progress
                progress = (i + 1) / total_tasks
                progress_bar.progress(progress)
                status_text.text(f"Executing task {i + 1}/{total_tasks}: {task['name']}")
                
                # Get user text for this task if available
                user_text_key = f"user_text_{task_id}"
                user_text = st.session_state.get(user_text_key, "")
                
                # Execute task based on type
                if task.get('type') == 'screening':
                    # Execute screening task
                    self.execute_screening_task(task, user_text)
                else:
                    # Execute regular task
                    self.execute_current_task(task, user_text)
                
                # Small delay between tasks
                time.sleep(0.5)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            st.success(f"✅ All {total_tasks} tasks completed successfully!")
            
        except Exception as e:
            st.error(f"❌ Batch execution failed: {str(e)}")
    
    def save_batch_to_json(self):
        """Save current task batch to JSON file"""
        try:
            import json
            from datetime import datetime
            
            if not st.session_state.task_list:
                st.warning("⚠️ No tasks to save.")
                return
            
            # Prepare batch data
            batch_data = {
                "created_at": datetime.now().isoformat(),
                "task_count": len(st.session_state.task_list),
                "tasks": []
            }
            
            # Process each task
            for task in st.session_state.task_list:
                task_data = task.copy()
                
                # Convert uploaded image to base64 if present
                if task_data.get('uploaded_image'):
                    import base64
                    task_data['uploaded_image'] = base64.b64encode(task_data['uploaded_image']).decode('utf-8')
                
                batch_data["tasks"].append(task_data)
            
            # Create JSON string
            json_str = json.dumps(batch_data, indent=2, ensure_ascii=False)
            
            # Offer download
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"task_batch_{timestamp}.json"
            
            st.download_button(
                label="📥 Download Task Batch",
                data=json_str,
                file_name=filename,
                mime="application/json"
            )
            
            st.success(f"✅ Task batch ready for download as '{filename}'")
            
        except Exception as e:
            st.error(f"❌ Failed to save batch: {str(e)}")
    
    def load_batch_from_json(self):
        """Load task batch from JSON file"""
        st.markdown("### 📂 Load Task Batch")
        
        uploaded_file = st.file_uploader(
            "Choose a JSON batch file",
            type=['json'],
            help="Upload a previously saved task batch file"
        )
        
        if uploaded_file is not None:
            try:
                import json
                import base64
                
                # Parse JSON
                batch_data = json.loads(uploaded_file.getvalue().decode('utf-8'))
                
                # Validate structure
                if 'tasks' not in batch_data:
                    st.error("❌ Invalid batch file format.")
                    return
                
                # Show batch info
                st.info(f"📋 Batch contains {len(batch_data['tasks'])} tasks")
                if 'created_at' in batch_data:
                    st.info(f"📅 Created: {batch_data['created_at']}")
                
                # Load options
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🔄 Replace Current Tasks", type="primary"):
                        # Clear current tasks
                        st.session_state.task_list = []
                        st.session_state.task_completion_status = {}
                        st.session_state.task_results = {}
                        
                        # Reset screening state when task list is updated
                        st.session_state.screening_started = False
                        st.session_state.current_task_index = 0
                        st.session_state.processing_task_id = None
                        
                        # Load new tasks
                        for task_data in batch_data['tasks']:
                            # Convert base64 image back to bytes if present
                            if task_data.get('uploaded_image'):
                                task_data['uploaded_image'] = base64.b64decode(task_data['uploaded_image'])
                            
                            st.session_state.task_list.append(task_data)
                        
                        st.session_state.show_load_batch = False
                        st.success(f"✅ Loaded {len(batch_data['tasks'])} tasks successfully!")
                        st.rerun()
                
                with col2:
                    if st.button("➕ Add to Current Tasks"):
                        # Add to existing tasks
                        start_id = len(st.session_state.task_list)
                        
                        # Reset screening state when task list is updated
                        st.session_state.screening_started = False
                        st.session_state.current_task_index = 0
                        st.session_state.processing_task_id = None
                        
                        for i, task_data in enumerate(batch_data['tasks']):
                            # Update task ID to avoid conflicts
                            task_data['id'] = start_id + i
                            
                            # Convert base64 image back to bytes if present
                            if task_data.get('uploaded_image'):
                                task_data['uploaded_image'] = base64.b64decode(task_data['uploaded_image'])
                            
                            st.session_state.task_list.append(task_data)
                        
                        st.session_state.show_load_batch = False
                        st.success(f"✅ Added {len(batch_data['tasks'])} tasks to current batch!")
                        st.rerun()
                
                # Cancel button
                if st.button("❌ Cancel"):
                    st.session_state.show_load_batch = False
                    st.rerun()
                
            except Exception as e:
                st.error(f"❌ Failed to load batch: {str(e)}") 
   
    def render_sidebar(self):
        """Render the sidebar with model controls"""
        with st.sidebar:
            st.markdown("## 🤖 Model Control")
            
            # Model initialization section
            st.markdown("### Model Initialization")
            
            # Model selection
            model_options = [
                "google/gemma-3n-E2B-it",
                "google/gemma-3n-E4B-it"
            ]
            
            selected_model = st.selectbox(
                "Select Model:",
                model_options,
                index=0,
                help="Choose the Gemma model variant to use"
            )
            
            # Model status
            if st.session_state.model_initialized:
                st.success("🟢 Model Ready")
                model_info = self.app_controller.get_model_info()
                if model_info.get('initialized'):
                    st.info(f"📍 Current: {model_info.get('model_path', 'Unknown')}")
                    st.info(f"🖥️ Device: {model_info.get('device', 'Unknown')}")
            else:
                st.error("🔴 Model Not Loaded")
            
            # Model control buttons
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🚀 Initialize", use_container_width=True, disabled=st.session_state.model_initialized):
                    with st.spinner("Initializing model..."):
                        success = self.app_controller.initialize_model(selected_model)
                        if success:
                            st.session_state.model_initialized = True
                            st.success("✅ Model initialized!")
                            st.rerun()
                        else:
                            st.error("❌ Model initialization failed!")
            
            with col2:
                if st.button("🔄 Reset", use_container_width=True, disabled=not st.session_state.model_initialized):
                    with st.spinner("Resetting model..."):
                        success = self.app_controller.reset_model()
                        if success:
                            st.session_state.model_initialized = False
                            st.success("✅ Model reset!")
                            st.rerun()
                        else:
                            st.error("❌ Model reset failed!")
            
            # GPU Memory Information
            st.markdown("---")
            st.markdown("### 💾 GPU Memory")
            
            memory_info = self.app_controller.get_gpu_memory_info()
            
            if memory_info.get('gpu_available'):
                # Memory usage bar
                usage_percent = memory_info.get('memory_usage_percent', 0)
                st.metric("Memory Usage", f"{usage_percent}%")
                st.progress(usage_percent / 100)
                
                # Detailed memory info
                with st.expander("📊 Memory Details"):
                    st.text(f"Total: {memory_info.get('total_memory', 0):.2f} GB")
                    st.text(f"Allocated: {memory_info.get('allocated_memory', 0):.2f} GB")
                    st.text(f"Cached: {memory_info.get('cached_memory', 0):.2f} GB")
                    st.text(f"Free: {memory_info.get('free_memory', 0):.2f} GB")
            else:
                st.info("🖥️ No GPU detected")
            
            # Model Cache Management
            st.markdown("---")
            st.markdown("### 🗂️ Model Cache")
            
            cache_stats = self.app_controller.get_cache_stats()
            
            if cache_stats.get('cache_exists'):
                st.metric("Cached Models", cache_stats.get('total_models', 0))
                st.metric("Cache Size", f"{cache_stats.get('total_size_mb', 0):.1f} MB")
                
                # Cache management buttons
                if st.button("🗑️ Clear Cache", use_container_width=True):
                    if self.app_controller.clear_model_cache():
                        st.success("✅ Cache cleared!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to clear cache!")
                
                # Show cached models
                with st.expander("📋 Cached Models"):
                    cached_models = self.app_controller.get_cached_models()
                    for model in cached_models:
                        st.text(f"• {model['model_name']}")
                        st.caption(f"  Size: {model['size_mb']:.1f} MB")
            else:
                st.info("📂 No cached models")
            
            # Session Management
            st.markdown("---")
            st.markdown("### 💾 Session")
            
            # Quick stats
            task_count = len(st.session_state.task_list)
            completed_count = sum(1 for status in st.session_state.task_completion_status.values() if status)
            
            st.metric("Tasks", f"{completed_count}/{task_count}")
            
            if st.session_state.screening_started:
                st.success("🟢 Screening Active")
            else:
                st.info("⚪ Screening Inactive")