# 🤖 Multimodal AI Chat with Gemma 3n

A powerful Streamlit application that enables multimodal AI interactions using Google's Gemma 3n model. Create prompts with text, images, and audio, process them in batches, and save your sessions for later use.

## ✨ Features

- **Multimodal Input Support**: Combine text, images (PNG, JPG, JPEG), audio (MP3, WAV, M4A), and video recording in your prompts
- **Batch Processing**: Create and execute multiple prompts efficiently with progress tracking
- **Shared Media**: Use shared images or audio across multiple text prompts in a batch
- **Session Management**: Save and load your chat sessions with full conversation history
- **Real-time Progress**: Track batch execution progress with detailed status updates
- **File Management**: Automatic file storage and cleanup for uploaded media
- **Responsive UI**: Clean, intuitive interface built with Streamlit

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for model inference)
- At least 8GB RAM (16GB+ recommended)
- FFmpeg (required for video processing)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd multimodal-streamlit-app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install FFmpeg** (required for video processing)
   ```bash
   # Ubuntu/Debian
   sudo apt update && sudo apt install ffmpeg
   
   # macOS
   brew install ffmpeg
   
   # Windows
   # Download from https://ffmpeg.org/download.html
   ```

4. **Configure the application**
   ```bash
   cp .env.example .env
   # Edit .env file with your preferred settings
   ```

5. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

6. **Open your browser**
   Navigate to `http://localhost:8501`

## 📖 Usage Guide

### 1. Initialize the Model

- Click "🔧 Model Configuration" to expand the setup section
- Select your preferred Gemma 3n model variant
- Click "🚀 Initialize Model" and wait for loading to complete
- The status indicator will show "🟢 Model Ready" when initialization is successful

### 2. Create Prompts

#### Single Modality
- **Text Only**: Enter your text in the "💬 Text" tab
- **Image Only**: Upload an image in the "🖼️ Image" tab
- **Audio Only**: Upload an audio file in the "🎵 Audio" tab
- **Video Only**: Record video with camera and microphone in the "🎥 Video" tab

#### Multimodal Combinations
- Use multiple tabs to combine different input types
- Example: Add text description + image for visual analysis
- Example: Add text question + audio for speech analysis
- Example: Add text question + video for behavioral analysis

#### Shared Media (Optional)
- Expand "🔗 Shared Media" section
- Upload images/audio that will be shared across all text prompts in the batch
- Useful for analyzing the same media with different text questions

### 3. Build and Execute Batches

1. **Add Prompts**: Click "➕ Add Prompt to Batch" after creating each prompt
2. **Review Batch**: Check the "📋 Prompt Batch" section to see all added prompts
3. **Execute**: Click "🚀 Execute Batch" to process all prompts
4. **Monitor Progress**: Watch the real-time progress indicator
5. **View Results**: Check the "📊 Results" section for responses

### 4. Manage Sessions

- **Save Session**: Use the sidebar to save your current batch and results
- **Load Session**: Restore previously saved sessions
- **Session History**: Browse and manage your saved sessions

## 🛠️ Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```env
# Storage Configuration
STORAGE_DIR=data                    # Directory for storing sessions and files

# Input Limits
MAX_TEXT_LENGTH=10000              # Maximum characters in text input
MAX_IMAGE_SIZE_MB=10               # Maximum image file size
MAX_AUDIO_SIZE_MB=25               # Maximum audio file size
MAX_IMAGE_WIDTH=2048               # Maximum image width (auto-resize)
MAX_IMAGE_HEIGHT=2048              # Maximum image height (auto-resize)

# Model Parameters
DEFAULT_MAX_TOKENS=256             # Default response length
MIN_MAX_TOKENS=50                  # Minimum response length
MAX_MAX_TOKENS=1000                # Maximum response length

# Session Management
AUTO_SAVE_SESSIONS=true            # Automatically save sessions
SESSION_CLEANUP_DAYS=30            # Days before old sessions are cleaned up
```

### Model Configuration

The application supports these Gemma 3n model variants:
- `google/gemma-3n-E2B-it` (default, smaller, faster)
- `google/gemma-3n-E4B-it` (larger, more capable)

## 📁 Project Structure

```
multimodal-streamlit-app/
├── src/
│   ├── models/
│   │   └── data_models.py          # Data structures (PromptData, SessionResult)
│   ├── services/
│   │   ├── input_handlers.py       # Text, image, audio processing
│   │   ├── model_service.py        # Gemma 3n integration
│   │   ├── storage_service.py      # Session and file management
│   │   └── batch_service.py        # Batch processing and execution
│   ├── ui/
│   │   └── main_interface.py       # Streamlit UI components
│   └── app_controller.py           # Main application controller
├── tests/                          # Unit tests
├── data/                          # Storage directory (created automatically)
├── streamlit_app.py               # Application entry point
├── config.py                      # Configuration management
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variables template
└── README.md                      # This file
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_data_models.py

# Run with verbose output
pytest -v
```

## 🔧 Development

### Adding New Features

1. **Input Handlers**: Extend `src/services/input_handlers.py` for new input types
2. **Model Integration**: Modify `src/services/model_service.py` for model changes
3. **UI Components**: Add new interface elements in `src/ui/main_interface.py`
4. **Storage**: Extend `src/services/storage_service.py` for new data types

### Code Style

- Follow PEP 8 guidelines
- Use type hints for function parameters and return values
- Add docstrings for all classes and methods
- Write unit tests for new functionality

## 🐛 Troubleshooting

### Common Issues

**Model Loading Fails**
- Ensure you have sufficient GPU memory (8GB+ recommended)
- Check internet connection for model download
- Verify CUDA installation if using GPU

**Out of Memory Errors**
- Reduce batch size
- Lower max_tokens setting
- Use smaller model variant (E2B instead of E4B)
- Resize large images before upload

**File Upload Issues**
- Check file size limits in configuration
- Ensure file formats are supported
- Verify storage directory permissions

**Session Loading Fails**
- Check if session files exist in storage directory
- Verify file permissions
- Look for corrupted session files

### Performance Optimization

- **GPU Usage**: Ensure CUDA is properly configured for faster inference
- **Memory Management**: Process smaller batches to avoid memory issues
- **File Storage**: Regularly clean up old sessions and files
- **Image Optimization**: Resize large images before upload

## 📊 Monitoring and Maintenance

### Storage Management

The application automatically manages file storage:
- Session data stored in JSON format
- Media files organized by type (images/, audio/)
- Automatic cleanup of orphaned files
- Session cleanup after configurable days

### Performance Monitoring

Monitor these metrics:
- Model inference time per prompt
- Memory usage during batch processing
- Storage space utilization
- Session load/save performance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google for the Gemma 3n model
- Streamlit team for the excellent web framework
- Hugging Face for the transformers library
- Contributors and testers

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section above
- Review the configuration options

---

**Happy chatting with your multimodal AI! 🚀**