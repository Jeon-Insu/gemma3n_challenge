"""
Application controller that wires together all components
"""
import streamlit as st
from typing import Optional
import os
from pathlib import Path
import torch

from src.services.model_service import ModelService
from src.services.storage_service import StorageService, FileManager
from src.services.batch_service import BatchExecutor
from src.services.input_handlers import TextInputHandler, ImageInputHandler, AudioInputHandler


class AppController:
    """Main application controller"""
    
    def __init__(self):
        self.model_service: Optional[ModelService] = None
        self.storage_service: Optional[StorageService] = None
        self.file_manager: Optional[FileManager] = None
        self.batch_executor: Optional[BatchExecutor] = None
        
        # Input handlers
        self.text_handler = TextInputHandler()
        self.image_handler = ImageInputHandler()
        self.audio_handler = AudioInputHandler()
        
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize core services"""
        try:
            # Initialize storage service
            storage_dir = os.getenv("STORAGE_DIR", "data")
            self.storage_service = StorageService(storage_dir)
            
            # Initialize file manager
            self.file_manager = FileManager(self.storage_service)
            
            # Initialize model service (but don't load model yet)
            self.model_service = ModelService()
            
            # Initialize batch executor
            self.batch_executor = BatchExecutor(
                self.model_service,
                self.storage_service,
                self.file_manager
            )
            
        except Exception as e:
            st.error(f"Failed to initialize services: {str(e)}")
            raise
    
    def initialize_model(self, model_path: str = "google/gemma-3n-E2B-it") -> bool:
        """
        Initialize the AI model
        
        Args:
            model_path: Path to the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.model_service.initialize_model(model_path)
            return True
        except Exception as e:
            st.error(f"Model initialization failed: {str(e)}")
            return False
    
    def reset_model(self) -> bool:
        """
        Reset the model and free GPU memory
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.model_service:
                # Use the enhanced cleanup method
                cleanup_results = self.model_service.force_cleanup_gpu_memory()
                
                if cleanup_results["success"]:
                    # Show detailed cleanup information
                    if cleanup_results["gpu_available"]:
                        memory_freed = cleanup_results.get("memory_freed", 0)
                        st.success(f"✅ Model reset successful! GPU memory freed: {memory_freed:.2f} GB")
                        
                        # Show actions taken
                        actions = cleanup_results.get("actions_taken", [])
                        if actions:
                            with st.expander("🔍 Cleanup Details", expanded=False):
                                for action in actions:
                                    st.text(f"• {action}")
                    else:
                        st.success("✅ Model reset successful! (No GPU detected)")
                    
                    return True
                else:
                    st.warning("⚠️ Model reset completed with some issues. Check cleanup details.")
                    actions = cleanup_results.get("actions_taken", [])
                    if actions:
                        with st.expander("⚠️ Cleanup Issues", expanded=True):
                            for action in actions:
                                st.text(f"• {action}")
                    return False
            else:
                st.info("No model service to reset.")
                return True
                
        except Exception as e:
            st.error(f"Model reset failed: {str(e)}")
            return False
    
    def get_gpu_memory_info(self) -> dict:
        """Get GPU memory information"""
        if self.model_service:
            return self.model_service.get_gpu_memory_info()
        return {"gpu_available": False}
    
    def is_model_ready(self) -> bool:
        """Check if model is ready for use"""
        return self.model_service and self.model_service.is_model_ready()
    
    def get_model_info(self) -> dict:
        """Get model information"""
        if self.model_service:
            return self.model_service.get_model_info()
        return {"initialized": False}
    
    def process_text_input(self, text: str) -> dict:
        """Process text input"""
        return self.text_handler.handle_text_input(text)
    
    def process_image_input(self, uploaded_file) -> dict:
        """Process image input"""
        return self.image_handler.handle_image_input(uploaded_file)
    
    def process_audio_input(self, uploaded_file) -> dict:
        """Process audio input"""
        return self.audio_handler.handle_audio_input(uploaded_file)
    
    def execute_batch(self, batch, max_tokens: int = 256, save_results: bool = True) -> dict:
        """
        Execute a batch of prompts
        
        Args:
            batch: PromptBatch to execute
            max_tokens: Maximum tokens per response
            save_results: Whether to save results
            
        Returns:
            Execution results
        """
        if not self.is_model_ready():
            raise RuntimeError("Model is not ready")
        
        return self.batch_executor.execute_batch(batch, max_tokens, save_results)
    
    def set_batch_progress_callback(self, callback):
        """Set progress callback for batch execution"""
        if self.batch_executor:
            self.batch_executor.set_progress_callback(callback)
    
    def save_session(self, session_data: dict) -> str:
        """Save session data"""
        return self.storage_service.save_session(session_data)
    
    def load_session(self, session_id: str) -> dict:
        """Load session data"""
        return self.storage_service.load_session(session_id)
    
    def list_sessions(self) -> list:
        """List all saved sessions"""
        return self.storage_service.list_sessions()
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        return self.storage_service.delete_session(session_id)
    
    def get_storage_stats(self) -> dict:
        """Get storage statistics"""
        return self.storage_service.get_storage_stats()
    
    def cleanup_old_sessions(self, days_old: int = 30) -> int:
        """Clean up old sessions"""
        return self.storage_service.cleanup_old_sessions(days_old)
    
    def get_cached_models(self) -> list:
        """Get list of cached models"""
        if self.model_service:
            return self.model_service.get_cached_models()
        return []
    
    def clear_model_cache(self, model_path: str = None) -> bool:
        """Clear model cache"""
        if self.model_service:
            return self.model_service.clear_model_cache(model_path)
        return False
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        if self.model_service:
            return self.model_service.get_cache_stats()
        return {}


# Global app controller instance
@st.cache_resource
def get_app_controller() -> AppController:
    """Get or create the global app controller instance"""
    return AppController()

def clear_app_controller_cache():
    """Clear the cached app controller instance"""
    get_app_controller.clear()