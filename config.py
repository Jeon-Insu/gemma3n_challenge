"""
Configuration management for the multimodal Streamlit app
"""
import os
from pathlib import Path
from typing import Dict, Any
import json
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


class Config:
    """Configuration class for the application"""
    
    # Model Configuration
    DEFAULT_MODEL_PATH = "google/gemma-3n-E2B-it"
    AVAILABLE_MODELS = [
        "google/gemma-3n-E2B-it",
        "google/gemma-3n-E4B-it"
    ]
    
    # Storage Configuration
    STORAGE_DIR = os.getenv("STORAGE_DIR", "data")
    
    # Input Limits
    MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "10000"))
    MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
    MAX_AUDIO_SIZE_MB = int(os.getenv("MAX_AUDIO_SIZE_MB", "25"))
    MAX_IMAGE_WIDTH = int(os.getenv("MAX_IMAGE_WIDTH", "2048"))
    MAX_IMAGE_HEIGHT = int(os.getenv("MAX_IMAGE_HEIGHT", "2048"))
    
    # Model Parameters
    DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "256"))
    MIN_MAX_TOKENS = int(os.getenv("MIN_MAX_TOKENS", "50"))
    MAX_MAX_TOKENS = int(os.getenv("MAX_MAX_TOKENS", "1000"))
    
    # Supported File Formats
    SUPPORTED_IMAGE_FORMATS = ["png", "jpg", "jpeg"]
    SUPPORTED_AUDIO_FORMATS = ["mp3", "wav", "m4a"]
    
    # Session Management
    AUTO_SAVE_SESSIONS = os.getenv("AUTO_SAVE_SESSIONS", "true").lower() == "true"
    SESSION_CLEANUP_DAYS = int(os.getenv("SESSION_CLEANUP_DAYS", "30"))
    
    # UI Configuration
    PAGE_TITLE = os.getenv("PAGE_TITLE", "Multimodal AI Chat")
    PAGE_ICON = os.getenv("PAGE_ICON", "🤖")
    LAYOUT = os.getenv("LAYOUT", "wide")
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "app.log")
    
    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """Get model configuration"""
        return {
            "default_model_path": cls.DEFAULT_MODEL_PATH,
            "available_models": cls.AVAILABLE_MODELS,
            "default_max_tokens": cls.DEFAULT_MAX_TOKENS,
            "min_max_tokens": cls.MIN_MAX_TOKENS,
            "max_max_tokens": cls.MAX_MAX_TOKENS
        }
    
    @classmethod
    def get_storage_config(cls) -> Dict[str, Any]:
        """Get storage configuration"""
        return {
            "storage_dir": cls.STORAGE_DIR,
            "auto_save_sessions": cls.AUTO_SAVE_SESSIONS,
            "session_cleanup_days": cls.SESSION_CLEANUP_DAYS
        }
    
    @classmethod
    def get_input_limits(cls) -> Dict[str, Any]:
        """Get input validation limits"""
        return {
            "max_text_length": cls.MAX_TEXT_LENGTH,
            "max_image_size_mb": cls.MAX_IMAGE_SIZE_MB,
            "max_audio_size_mb": cls.MAX_AUDIO_SIZE_MB,
            "max_image_width": cls.MAX_IMAGE_WIDTH,
            "max_image_height": cls.MAX_IMAGE_HEIGHT,
            "supported_image_formats": cls.SUPPORTED_IMAGE_FORMATS,
            "supported_audio_formats": cls.SUPPORTED_AUDIO_FORMATS
        }
    
    @classmethod
    def get_ui_config(cls) -> Dict[str, Any]:
        """Get UI configuration"""
        return {
            "page_title": cls.PAGE_TITLE,
            "page_icon": cls.PAGE_ICON,
            "layout": cls.LAYOUT
        }
    
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """Validate configuration and return any issues"""
        issues = []
        
        # Check storage directory
        storage_path = Path(cls.STORAGE_DIR)
        try:
            storage_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            issues.append(f"Cannot create storage directory {cls.STORAGE_DIR}: {e}")
        
        # Check numeric values
        if cls.MAX_TEXT_LENGTH <= 0:
            issues.append("MAX_TEXT_LENGTH must be positive")
        
        if cls.MAX_IMAGE_SIZE_MB <= 0:
            issues.append("MAX_IMAGE_SIZE_MB must be positive")
        
        if cls.MAX_AUDIO_SIZE_MB <= 0:
            issues.append("MAX_AUDIO_SIZE_MB must be positive")
        
        if cls.DEFAULT_MAX_TOKENS < cls.MIN_MAX_TOKENS or cls.DEFAULT_MAX_TOKENS > cls.MAX_MAX_TOKENS:
            issues.append("DEFAULT_MAX_TOKENS must be between MIN_MAX_TOKENS and MAX_MAX_TOKENS")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues
        }
    
    @classmethod
    def save_config_to_file(cls, filepath: str) -> None:
        """Save current configuration to a JSON file"""
        config_data = {
            "model": cls.get_model_config(),
            "storage": cls.get_storage_config(),
            "input_limits": cls.get_input_limits(),
            "ui": cls.get_ui_config()
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    @classmethod
    def load_config_from_file(cls, filepath: str) -> None:
        """Load configuration from a JSON file (for future use)"""
        # This would update class variables from file
        # Implementation depends on specific requirements
        pass


# Create a global config instance
config = Config()

# Validate configuration on import
validation_result = config.validate_config()
if not validation_result["valid"]:
    print("Configuration validation issues:")
    for issue in validation_result["issues"]:
        print(f"  - {issue}")