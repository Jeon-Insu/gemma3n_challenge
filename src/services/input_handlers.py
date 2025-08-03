"""
Input handlers for different modalities
"""
from typing import Dict, Any, Optional
import re
import io
from PIL import Image
import base64


class TextInputHandler:
    """Handler for text input processing and validation"""
    
    def __init__(self, max_length: int = 10000):
        self.max_length = max_length
    
    def handle_text_input(self, text: str) -> Dict[str, Any]:
        """
        Process and validate text input
        
        Args:
            text: Raw text input from user
            
        Returns:
            Dictionary with processed text and metadata
        """
        if not text:
            return {
                "valid": False,
                "error": "Text input cannot be empty",
                "processed_text": None
            }
        
        # Clean and validate text
        cleaned_text = self._clean_text(text)
        
        if len(cleaned_text) > self.max_length:
            return {
                "valid": False,
                "error": f"Text exceeds maximum length of {self.max_length} characters",
                "processed_text": None
            }
        
        return {
            "valid": True,
            "processed_text": cleaned_text,
            "original_length": len(text),
            "processed_length": len(cleaned_text),
            "content_type": "text"
        }
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text input by removing excessive whitespace and normalizing
        
        Args:
            text: Raw text input
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        # Remove any potentially harmful characters (basic sanitization)
        cleaned = re.sub(r'[^\w\s\.,!?;:\-\(\)\[\]\'\"@#$%&*+=<>/\\|`~]', '', cleaned)
        
        return cleaned
    
    def validate_text(self, text: str) -> bool:
        """
        Simple validation for text input
        
        Args:
            text: Text to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not text or not text.strip():
            return False
        
        if len(text) > self.max_length:
            return False
        
        return True
    
    def format_for_model(self, text: str) -> Dict[str, Any]:
        """
        Format text for model input
        
        Args:
            text: Processed text
            
        Returns:
            Dictionary formatted for model consumption
        """
        return {
            "type": "text",
            "text": text
        }


class ImageInputHandler:
    """Handler for image input processing and validation"""
    
    def __init__(self, max_size_mb: int = 10, max_width: int = 2048, max_height: int = 2048):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_width = max_width
        self.max_height = max_height
        self.supported_formats = {'PNG', 'JPEG', 'JPG'}
    
    def handle_image_input(self, uploaded_file) -> Dict[str, Any]:
        """
        Process and validate image input
        
        Args:
            uploaded_file: Streamlit uploaded file object or file-like object
            
        Returns:
            Dictionary with processed image data and metadata
        """
        if uploaded_file is None:
            return {
                "valid": False,
                "error": "No image file provided",
                "processed_image": None
            }
        
        try:
            # Get file info
            file_size = len(uploaded_file.getvalue()) if hasattr(uploaded_file, 'getvalue') else uploaded_file.size
            filename = getattr(uploaded_file, 'name', 'unknown')
            
            # Validate file size
            if file_size > self.max_size_bytes:
                return {
                    "valid": False,
                    "error": f"Image file too large. Maximum size: {self.max_size_bytes // (1024*1024)}MB",
                    "processed_image": None
                }
            
            # Read image data
            if hasattr(uploaded_file, 'getvalue'):
                image_bytes = uploaded_file.getvalue()
            else:
                uploaded_file.seek(0)
                image_bytes = uploaded_file.read()
            
            # Validate and process image
            validation_result = self._validate_and_process_image(image_bytes, filename)
            
            if not validation_result["valid"]:
                return validation_result
            
            return {
                "valid": True,
                "processed_image": validation_result["processed_bytes"],
                "original_size": file_size,
                "filename": filename,
                "format": validation_result["format"],
                "dimensions": validation_result["dimensions"],
                "content_type": "image"
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"Error processing image: {str(e)}",
                "processed_image": None
            }
    
    def _validate_and_process_image(self, image_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        Validate image format and process for model compatibility
        
        Args:
            image_bytes: Raw image bytes
            filename: Original filename
            
        Returns:
            Dictionary with validation results and processed image
        """
        try:
            # Open image with PIL
            image = Image.open(io.BytesIO(image_bytes))
            
            # Validate format
            if image.format not in self.supported_formats:
                return {
                    "valid": False,
                    "error": f"Unsupported image format: {image.format}. Supported: {', '.join(self.supported_formats)}"
                }
            
            # Get original dimensions
            width, height = image.size
            
            # Resize if too large
            if width > self.max_width or height > self.max_height:
                # Calculate new dimensions maintaining aspect ratio
                ratio = min(self.max_width / width, self.max_height / height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to RGB if necessary (for JPEG compatibility)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Save processed image to bytes
            output_buffer = io.BytesIO()
            image.save(output_buffer, format='JPEG', quality=85)
            processed_bytes = output_buffer.getvalue()
            
            return {
                "valid": True,
                "processed_bytes": processed_bytes,
                "format": image.format,
                "dimensions": image.size,
                "original_dimensions": (width, height)
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"Invalid image file: {str(e)}"
            }
    
    def validate_image_format(self, filename: str) -> bool:
        """
        Validate image format based on filename extension
        
        Args:
            filename: Name of the file
            
        Returns:
            True if format is supported, False otherwise
        """
        if not filename:
            return False
        
        extension = filename.split('.')[-1].upper()
        return extension in self.supported_formats
    
    def format_for_model(self, image_bytes: bytes, filename: str = None) -> Dict[str, Any]:
        """
        Format image for model input
        
        Args:
            image_bytes: Processed image bytes
            filename: Optional filename
            
        Returns:
            Dictionary formatted for model consumption
        """
        # Convert to base64 for model input (if needed) or keep as bytes
        return {
            "type": "image",
            "image": image_bytes,
            "filename": filename
        }


class AudioInputHandler:
    """Handler for audio input processing and validation"""
    
    def __init__(self, max_size_mb: int = 25, max_duration_seconds: int = 300):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_duration_seconds = max_duration_seconds
        self.supported_formats = {'MP3', 'WAV', 'M4A'}
    
    def handle_audio_input(self, uploaded_file) -> Dict[str, Any]:
        """
        Process and validate audio input
        
        Args:
            uploaded_file: Streamlit uploaded file object or file-like object
            
        Returns:
            Dictionary with processed audio data and metadata
        """
        if uploaded_file is None:
            return {
                "valid": False,
                "error": "No audio file provided",
                "processed_audio": None
            }
        
        try:
            # Get file info
            file_size = len(uploaded_file.getvalue()) if hasattr(uploaded_file, 'getvalue') else uploaded_file.size
            filename = getattr(uploaded_file, 'name', 'unknown')
            
            # Validate file size
            if file_size > self.max_size_bytes:
                return {
                    "valid": False,
                    "error": f"Audio file too large. Maximum size: {self.max_size_bytes // (1024*1024)}MB",
                    "processed_audio": None
                }
            
            # Validate format
            if not self.validate_audio_format(filename):
                return {
                    "valid": False,
                    "error": f"Unsupported audio format. Supported: {', '.join(self.supported_formats)}",
                    "processed_audio": None
                }
            
            # Read audio data
            if hasattr(uploaded_file, 'getvalue'):
                audio_bytes = uploaded_file.getvalue()
            else:
                uploaded_file.seek(0)
                audio_bytes = uploaded_file.read()
            
            # Basic validation of audio data
            validation_result = self._validate_audio_data(audio_bytes, filename)
            
            if not validation_result["valid"]:
                return validation_result
            
            return {
                "valid": True,
                "processed_audio": audio_bytes,
                "original_size": file_size,
                "filename": filename,
                "format": self._get_audio_format(filename),
                "content_type": "audio"
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"Error processing audio: {str(e)}",
                "processed_audio": None
            }
    
    def _validate_audio_data(self, audio_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        Basic validation of audio data
        
        Args:
            audio_bytes: Raw audio bytes
            filename: Original filename
            
        Returns:
            Dictionary with validation results
        """
        try:
            # Basic file signature validation
            if filename.lower().endswith('.mp3'):
                # Check for MP3 header (ID3 or frame sync)
                if not (audio_bytes.startswith(b'ID3') or 
                       audio_bytes.startswith(b'\xff\xfb') or 
                       audio_bytes.startswith(b'\xff\xfa')):
                    return {
                        "valid": False,
                        "error": "Invalid MP3 file format"
                    }
            elif filename.lower().endswith('.wav'):
                # Check for WAV header
                if not audio_bytes.startswith(b'RIFF'):
                    return {
                        "valid": False,
                        "error": "Invalid WAV file format"
                    }
            
            # Additional validation could be added here using audio libraries
            # For now, we'll do basic checks
            
            return {
                "valid": True
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"Audio validation error: {str(e)}"
            }
    
    def validate_audio_format(self, filename: str) -> bool:
        """
        Validate audio format based on filename extension
        
        Args:
            filename: Name of the file
            
        Returns:
            True if format is supported, False otherwise
        """
        if not filename:
            return False
        
        extension = filename.split('.')[-1].upper()
        return extension in self.supported_formats
    
    def _get_audio_format(self, filename: str) -> str:
        """
        Get audio format from filename
        
        Args:
            filename: Name of the file
            
        Returns:
            Audio format string
        """
        if not filename:
            return "unknown"
        
        return filename.split('.')[-1].upper()
    
    def format_for_model(self, audio_bytes: bytes, filename: str = None) -> Dict[str, Any]:
        """
        Format audio for model input
        
        Args:
            audio_bytes: Processed audio bytes
            filename: Optional filename
            
        Returns:
            Dictionary formatted for model consumption
        """
        return {
            "type": "audio",
            "audio": audio_bytes,
            "filename": filename
        }