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


class AppController:
    """Main application controller"""
    
    def __init__(self):
        self.model_service: Optional[ModelService] = None
        self.storage_service: Optional[StorageService] = None
        self.file_manager: Optional[FileManager] = None
        
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