# V³-Gemma: Multimodal Depression Screener

This project is a web-based application that performs comprehensive depression screening through multimodal analysis of user-recorded videos. It analyzes **V**isual, **V**ocal, and **V**erbal signals from video recordings using Google's Gemma 3n model to evaluate 15 standardized screening questions (Q1-Q15) and provides automated risk assessment and clinical insights.

🚀 **Live Demo**
Try the application live on our cloud service!

➡️ **[V³-Gemma Depression Screener Demo](https://moai.ai.kr) or (https://moai.ai.kr.ngrok.app/)**


This project is intended as a comprehensive screening tool for researchers and healthcare professionals. The instructions below provide a step-by-step guide to get the application running locally or access it through our cloud service.

## Project Structure

This project is built with Streamlit and organized as follows:

- `streamlit_app.py`: The main Streamlit application entry point
- `src/app_controller.py`: Core application controller that manages all services
- `src/ui/main_interface.py`: Main Streamlit interface with 4 tabs (Task, Screening, Results, Reporting)
- `src/services/model_service.py`: Gemma 3n model integration and inference logic
- `src/services/llm_iteration_task.py`: LLM iteration task processing for Q1-Q15 questions
- `src/services/storage_service.py`: Session and file management
- `config/questions.py`: Q1-Q15 screening questions configuration
- `function/function.py`: Core model iteration functions
- `data/`: Storage directory for sessions and extracted media files

## Getting Started

This guide assumes you have a working knowledge of Python development and access to a CUDA-compatible GPU for optimal performance.

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (8GB+ VRAM recommended)
- FFmpeg for video processing
- Camera and microphone access

### Hugging Face Access

The `google/gemma-3n-E2B-it` model requires Hugging Face authentication. Visit the model page, accept the license terms, and log in:

```bash
pip install huggingface_hub
huggingface-cli login
```

### Installation & Setup

1. **Clone and Navigate:**
   ```bash
   git clone <repository-url>
   cd V3_gemma
   ```

2. **Install Dependencies:**
   It is highly recommended to use a Python virtual environment:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install FFmpeg:**
   ```bash
   # Ubuntu/Debian
   sudo apt update && sudo apt install ffmpeg
   
   # macOS
   brew install ffmpeg
   
   # Windows: Download from https://ffmpeg.org/download.html
   ```

4. **Run the Application:**
   ```bash
   streamlit run streamlit_app.py
   ```
   The application will be accessible at `http://localhost:8501`.

   The first time you run this, the script will download the Gemma model, which may take some time.

## Core Features

### 🎥 Video Recording & Processing
- Real-time video recording with camera and microphone
- Automatic extraction of audio and image frames
- FFmpeg-based video processing pipeline

### 🤖 AI-Powered Screening
- **15 Comprehensive Questions (Q1-Q15):** Systematic evaluation using standardized screening criteria
- **Visual Analysis (Q1-Q7):** Appearance, posture, facial expressions, body language assessment
- **Audio Analysis (Q8-Q13):** Voice characteristics, speech patterns, silence detection
- **Content Analysis (Q14-Q15):** Verbal responses to emotional stimuli (house fire scenario)

### 📊 Intelligent Results & Reporting
- **LLM-Powered Scoring:** Advanced algorithms using Gemma 3n for contextual analysis
- **Automated Report Generation:** Comprehensive screening reports with clinical insights
- **Risk Assessment:** Automated categorization (Low/Medium/High risk levels)
- **Export Functionality:** JSON export for further analysis

### 💾 Session Management
- Auto-save screening sessions during processing
- Load and review previous screening results
- Comprehensive session history management

## Application Interface

The application features 4 main tabs:

1. **📋 Task Tab:** Model configuration and initialization
2. **🎬 Screening Tab:** Video recording and automatic Q1-Q15 question processing
3. **📊 Results Tab:** Detailed screening results with scoring and analysis
4. **📈 Reporting Tab:** Comprehensive reports and export functionality

## Quick Usage Guide

1. **Initialize Model:** Configure and load the Gemma 3n model in the Task tab
2. **Record Video:** Navigate to Screening tab, record a video describing the house fire image
3. **Automatic Processing:** The system automatically processes all 15 questions (Q1-Q15)
4. **View Results:** Check Results tab for detailed scoring and analysis
5. **Generate Reports:** Use Reporting tab for comprehensive assessment reports

## Configuration

### Environment Variables

Create a `.env` file for custom configuration:

```env
STORAGE_DIR=data
DEFAULT_MAX_TOKENS=256
AUTO_SAVE_SESSIONS=true
SESSION_CLEANUP_DAYS=30
MAX_VIDEO_SIZE_MB=100
```

### Model Variants

- `google/gemma-3n-E2B-it` (default, faster, 2B parameters)
- `google/gemma-3n-E4B-it` (larger, more capable, 4B parameters)

## Technical Requirements

- **GPU Memory:** 8GB+ VRAM for optimal performance
- **System RAM:** 16GB+ recommended
- **Storage:** 10GB+ for model and data files
- **Network:** Required for initial model download

## Clinical Considerations

⚠️ **Important:** This application is for screening purposes only and is not a diagnostic tool. Results should be interpreted by qualified healthcare professionals following appropriate clinical guidelines.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google for the Gemma 3n model
- Streamlit team for the web framework
- Hugging Face for the transformers library
- Chonnam National University Hospital
- Harvard/MGH
- GIST

## Support

For support and questions:
- GitHub:
- Youtube link: 
- Contact: torot383@naver.com
---

**Advancing mental health screening through AI! 🧠💚**