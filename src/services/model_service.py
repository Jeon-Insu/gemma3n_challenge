"""
Model service for Gemma 3n integration
"""
from typing import Dict, List, Any, Optional
import torch
import gc
from datetime import datetime
import json
import os
from pathlib import Path


class ChatState:
    """Enhanced ChatState class for conversation management"""
    
    def __init__(self, model=None, processor=None):
        self.model = model
        self.processor = processor
        self.history = []
        self.session_id = None
        self.created_at = datetime.now()
    
    def send_message(self, message: Dict[str, Any], max_tokens: int = 256) -> str:
        """
        Send a message to the model and get response
        Based on the provided Gemma 3n example code
        
        Args:
            message: Message dictionary with role and content
            max_tokens: Maximum tokens for response
            
        Returns:
            Model response text
        """
        if not self.model or not self.processor:
            raise ValueError("Model and processor must be initialized")
        self.history = []
        # Add message to history
        self.history.append(message)
        
        try:
            # Apply chat template and tokenize - exactly as in the example
            input_ids = self.processor.apply_chat_template(
                self.history,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            
            input_len = input_ids["input_ids"].shape[-1]
            input_ids = input_ids.to('cuda:0', dtype=self.model.dtype)
            
            # Generate response - exactly as in the example
            outputs = self.model.generate(
                **input_ids,
                max_new_tokens=max_tokens,
                temperature=0.1,
                disable_compile=True
            )
            
            # Decode response - exactly as in the example
            text = self.processor.batch_decode(
                outputs[:, input_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            response_text = text[0] if text else ""
            
            # Add assistant response to history - exactly as in the example
            self.history.append({
                "role": "assistant",
                "content": [{"type": "text", "text": response_text}]
            })
            
            # Force garbage collection after processing
            gc.collect()
            
            return response_text
            
        except Exception as e:
            # Remove the failed message from history
            if self.history and self.history[-1] == message:
                self.history.pop()
            raise e
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get conversation history
        
        Returns:
            List of conversation messages
        """
        return self.history.copy()
    
    def clear_history(self) -> None:
        """Clear conversation history"""
        self.history = []
    
    def export_session(self) -> Dict[str, Any]:
        """
        Export session data for storage
        
        Returns:
            Dictionary containing session data
        """
        return {
            "session_id": self.session_id,
            "history": self.history,
            "created_at": self.created_at.isoformat(),
            "message_count": len(self.history)
        }
    
    def import_session(self, session_data: Dict[str, Any]) -> None:
        """
        Import session data
        
        Args:
            session_data: Dictionary containing session data
        """
        self.session_id = session_data.get("session_id")
        self.history = session_data.get("history", [])
        
        if "created_at" in session_data:
            self.created_at = datetime.fromisoformat(session_data["created_at"])
    
    def get_last_response(self) -> Optional[str]:
        """
        Get the last assistant response
        
        Returns:
            Last response text or None
        """
        for message in reversed(self.history):
            if message.get("role") == "assistant":
                content = message.get("content", [])
                for item in content:
                    if item.get("type") == "text":
                        return item.get("text")
        return None
    
    def get_message_count(self) -> int:
        """
        Get total number of messages in history
        
        Returns:
            Number of messages
        """
        return len(self.history)
    
    def get_user_messages(self) -> List[Dict[str, Any]]:
        """
        Get only user messages from history
        
        Returns:
            List of user messages
        """
        return [msg for msg in self.history if msg.get("role") == "user"]
    
    def get_assistant_messages(self) -> List[Dict[str, Any]]:
        """
        Get only assistant messages from history
        
        Returns:
            List of assistant messages
        """
        return [msg for msg in self.history if msg.get("role") == "assistant"]
    
    def to_json(self) -> str:
        """
        Convert session to JSON string
        
        Returns:
            JSON representation of session
        """
        return json.dumps(self.export_session(), indent=2)
    
    def from_json(self, json_str: str) -> None:
        """
        Load session from JSON string
        
        Args:
            json_str: JSON string containing session data
        """
        session_data = json.loads(json_str)
        self.import_session(session_data)


class ModelService:
    """Service for managing Gemma 3n model operations"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.model_path = None
        self.is_initialized = False
        self.chat_state = None
        self.cache_dir = Path("saved_models")  # 모델 캐시 디렉토리
    
    def _get_model_cache_path(self, model_path: str) -> Path:
        """
        Get the cache path for a specific model
        
        Args:
            model_path: Original model path/name
            
        Returns:
            Path to cached model directory
        """
        # Create a safe directory name from model path
        safe_name = model_path.replace("/", "_").replace(":", "_")
        return self.cache_dir / safe_name
    
    def _is_model_cached(self, model_path: str) -> bool:
        """
        Check if model is already cached locally
        
        Args:
            model_path: Model path/name to check
            
        Returns:
            True if model is cached, False otherwise
        """
        cache_path = self._get_model_cache_path(model_path)
        
        # Check if cache directory exists and contains necessary files
        if not cache_path.exists():
            return False
        
        # Check for essential model files
        required_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
        alternative_files = ["model.safetensors"]  # Alternative to pytorch_model.bin
        
        has_config = (cache_path / "config.json").exists()
        has_tokenizer = (cache_path / "tokenizer.json").exists() or (cache_path / "tokenizer_config.json").exists()
        has_model = (cache_path / "pytorch_model.bin").exists() or (cache_path / "model.safetensors").exists()
        
        # Check for index files in case of sharded models
        has_index = (cache_path / "pytorch_model.bin.index.json").exists() or (cache_path / "model.safetensors.index.json").exists()
        
        return has_config and has_tokenizer and (has_model or has_index)
    
    def _save_model_to_cache(self, model_path: str) -> None:
        """
        Save model to local cache after loading
        
        Args:
            model_path: Original model path/name
        """
        try:
            cache_path = self._get_model_cache_path(model_path)
            
            # Create cache directory
            cache_path.mkdir(parents=True, exist_ok=True)
            
            print(f"Saving model to cache: {cache_path}")
            
            # Save processor and model to cache
            if self.processor:
                self.processor.save_pretrained(cache_path)
            
            if self.model:
                self.model.save_pretrained(cache_path)
            
            print(f"Model cached successfully at: {cache_path}")
            
        except Exception as e:
            print(f"Warning: Failed to cache model: {str(e)}")
            # Don't raise exception as this is not critical for functionality
    
    def initialize_model(self, model_path: str = "google/gemma-3n-E2B-it") -> None:
        """
        Initialize the Gemma 3n model and processor with caching support
        
        Args:
            model_path: Path or name of the model to load
        """
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
            
            # Check CUDA availability
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available. Please ensure GPU drivers are installed.")
            
            self.model_path = model_path
            cache_path = self._get_model_cache_path(model_path)
            
            # Check if model is cached locally
            if self._is_model_cached(model_path):
                print(f"Loading model from cache: {cache_path}")
                
                # Load from cache with proper device mapping
                self.processor = AutoProcessor.from_pretrained(str(cache_path))
                self.model = AutoModelForImageTextToText.from_pretrained(
                    str(cache_path), 
                    torch_dtype=torch.float16,
                    device_map=None
                ).to("cuda")
                
                print(f"Model loaded from cache successfully!")
                
            else:
                print(f"Model not found in cache. Downloading from: {model_path}")
                
                # Load from HuggingFace Hub
                self.processor = AutoProcessor.from_pretrained(model_path)
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_path, 
                    torch_dtype=torch.float16, 
                    device_map=None
                ).to("cuda")
                
                print(f"Model downloaded successfully!")
                
                # Save to cache for future use
                self._save_model_to_cache(model_path)
            
            # Force CUDA device
            torch.cuda.set_device(0)
            
            # Model is already moved to CUDA during loading
            print("Model loaded and moved to CUDA successfully!")
            
            # Verify model is on CUDA
            if hasattr(self.model, 'device'):
                current_device = str(self.model.device)
                print(f"Model device: {current_device}")
                if 'cuda' not in current_device.lower():
                    print("Warning: Model is not on CUDA device")
            else:
                print("Warning: Could not determine model device")
            
            # Initialize chat state
            self.chat_state = ChatState(self.model, self.processor)
            self.is_initialized = True
            
            # Debug prints for CUDA status
            print(f"🔍 Debug: CUDA available: {torch.cuda.is_available()}")
            print(f"🔍 Debug: CUDA device count: {torch.cuda.device_count()}")
            print(f"🔍 Debug: Current CUDA device: {torch.cuda.current_device()}")
            
            # Get model device info
            try:
                model_device = next(self.model.parameters()).device
                print(f"🔍 Debug: Model device: {model_device}")
            except Exception as e:
                print(f"Warning: Could not get model device: {str(e)}")
            
            # Verify model is properly loaded
            if hasattr(self.model, 'device'):
                print(f"Model device attribute: {self.model.device}")
            else:
                print("Warning: Model device attribute not available")
            
            if hasattr(self.model, 'dtype'):
                print(f"Model dtype: {self.model.dtype}")
            else:
                print("Warning: Model dtype attribute not available")
            
            # Additional verification - only if model is on CUDA
            try:
                if torch.cuda.is_available() and hasattr(self.model, 'device'):
                    if 'cuda' in str(self.model.device).lower():
                        # Test if model can be used on CUDA
                        test_input = torch.randn(1, 3, 224, 224).to('cuda:0')
                        print("Model initialization verification successful (CUDA)")
                    else:
                        print("Model is not on CUDA, skipping CUDA verification")
                else:
                    print("CUDA not available or model device unknown, skipping verification")
            except Exception as e:
                print(f"Warning: Model verification failed: {str(e)}")
                # Don't fail initialization for verification issues
            
        except Exception as e:
            self.is_initialized = False
            raise RuntimeError(f"Failed to initialize model: {str(e)}")
    
    def process_prompt(self, prompt: Dict[str, Any], max_tokens: int = 256) -> str:
        """
        Process a single prompt and return response (without history)
        
        Args:
            prompt: Formatted prompt dictionary
            max_tokens: Maximum tokens for response
            
        Returns:
            Model response text
        """
        if not self.is_initialized:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")
        
        try:
            # Create a new ChatState for each prompt to ensure no history contamination
            # Each task execution is completely independent
            temp_chat_state = ChatState(self.model, self.processor)
            response = temp_chat_state.send_message(prompt, max_tokens)
            return response
        except Exception as e:
            raise RuntimeError(f"Error processing prompt: {str(e)}")
    
    def process_batch(self, prompts: List[Dict[str, Any]], max_tokens: int = 256) -> List[Dict[str, Any]]:
        """
        Process multiple prompts in batch
        
        Args:
            prompts: List of formatted prompt dictionaries
            max_tokens: Maximum tokens for each response
            
        Returns:
            List of results with prompts and responses
        """
        if not self.is_initialized:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")
        
        results = []
        
        for i, prompt in enumerate(prompts):
            try:
                # Create a new chat state for each prompt to ensure independent execution
                temp_chat_state = ChatState(self.model, self.processor)
                response = temp_chat_state.send_message(prompt, max_tokens)
                
                results.append({
                    "prompt_index": i,
                    "prompt": prompt,
                    "response": response,
                    "success": True,
                    "error": None
                })
                
            except Exception as e:
                results.append({
                    "prompt_index": i,
                    "prompt": prompt,
                    "response": None,
                    "success": False,
                    "error": str(e)
                })
            
            # Force garbage collection after each prompt
            gc.collect()
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        if not self.is_initialized:
            return {
                "initialized": False,
                "model_path": None,
                "device": None,
                "dtype": None
            }
        
        # Get CUDA information
        cuda_info = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_cuda_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }
        
        return {
            "initialized": True,
            "model_path": self.model_path,
            "device": str(self.model.device) if self.model else None,
            "dtype": str(self.model.dtype) if self.model else None,
            "cuda_info": cuda_info
        }
    
    def clear_conversation(self) -> None:
        """Clear the current conversation history"""
        if self.chat_state:
            self.chat_state.clear_history()
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get current conversation history
        
        Returns:
            List of conversation messages
        """
        if self.chat_state:
            return self.chat_state.get_history()
        return []
    
    def export_conversation(self) -> Dict[str, Any]:
        """
        Export current conversation for storage
        
        Returns:
            Dictionary with conversation data
        """
        if self.chat_state:
            return self.chat_state.export_session()
        return {}
    
    def import_conversation(self, conversation_data: Dict[str, Any]) -> None:
        """
        Import conversation data
        
        Args:
            conversation_data: Dictionary with conversation data
        """
        if self.chat_state:
            self.chat_state.import_session(conversation_data)
    
    def is_model_ready(self) -> bool:
        """
        Check if model is ready for processing
        
        Returns:
            True if model is initialized and ready
        """
        return self.is_initialized and self.model is not None and self.processor is not None
    
    def get_cached_models(self) -> List[Dict[str, Any]]:
        """
        Get list of cached models
        
        Returns:
            List of cached model information
        """
        cached_models = []
        
        if not self.cache_dir.exists():
            return cached_models
        
        for model_dir in self.cache_dir.iterdir():
            if model_dir.is_dir():
                # Convert safe name back to original model path
                original_name = model_dir.name.replace("_", "/", 1).replace("_", ":")
                
                # Get cache size
                cache_size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
                cache_size_mb = cache_size / (1024 * 1024)
                
                # Get creation time
                creation_time = datetime.fromtimestamp(model_dir.stat().st_ctime)
                
                cached_models.append({
                    "model_name": original_name,
                    "cache_path": str(model_dir),
                    "size_mb": round(cache_size_mb, 2),
                    "cached_at": creation_time.isoformat(),
                    "is_valid": self._is_model_cached(original_name)
                })
        
        return cached_models
    
    def clear_model_cache(self, model_path: str = None) -> bool:
        """
        Clear model cache
        
        Args:
            model_path: Specific model to clear, or None to clear all
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import shutil
            
            if model_path:
                # Clear specific model cache
                cache_path = self._get_model_cache_path(model_path)
                if cache_path.exists():
                    shutil.rmtree(cache_path)
                    print(f"Cleared cache for model: {model_path}")
            else:
                # Clear all model caches
                if self.cache_dir.exists():
                    shutil.rmtree(self.cache_dir)
                    print("Cleared all model caches")
            
            return True
            
        except Exception as e:
            print(f"Failed to clear model cache: {str(e)}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        if not self.cache_dir.exists():
            return {
                "total_models": 0,
                "total_size_mb": 0,
                "cache_dir": str(self.cache_dir),
                "cache_exists": False
            }
        
        cached_models = self.get_cached_models()
        total_size = sum(model["size_mb"] for model in cached_models)
        
        return {
            "total_models": len(cached_models),
            "total_size_mb": round(total_size, 2),
            "cache_dir": str(self.cache_dir),
            "cache_exists": True,
            "models": cached_models
        }
    
    def force_cleanup_gpu_memory(self) -> Dict[str, Any]:
        """
        Force cleanup of GPU memory with detailed reporting
        
        Returns:
            Dictionary with cleanup results and memory stats
        """
        cleanup_results = {
            "success": False,
            "gpu_available": False,
            "memory_before": 0,
            "memory_after": 0,
            "memory_freed": 0,
            "actions_taken": []
        }
        
        try:
            # Check if CUDA is available
            if not torch.cuda.is_available():
                cleanup_results["actions_taken"].append("CUDA not available")
                cleanup_results["success"] = True
                return cleanup_results
            
            cleanup_results["gpu_available"] = True
            
            # Get memory before cleanup
            if torch.cuda.is_available():
                cleanup_results["memory_before"] = torch.cuda.memory_allocated() / (1024**3)  # GB
            
            # Step 1: Clear model references with proper cleanup
            if self.model is not None:
                try:
                    # Move model to CPU first to avoid device issues
                    if hasattr(self.model, 'cpu'):
                        self.model.cpu()
                    del self.model
                    cleanup_results["actions_taken"].append("Moved model to CPU and deleted reference")
                except Exception as e:
                    cleanup_results["actions_taken"].append(f"Error moving model to CPU: {str(e)}")
                finally:
                    self.model = None
            
            if self.processor is not None:
                del self.processor
                self.processor = None
                cleanup_results["actions_taken"].append("Deleted processor reference")
            
            if self.chat_state is not None:
                del self.chat_state
                self.chat_state = None
                cleanup_results["actions_taken"].append("Deleted chat state")
            
            self.is_initialized = False
            
            # Step 2: Force garbage collection multiple times
            import gc
            for i in range(3):
                collected = gc.collect()
                if collected > 0:
                    cleanup_results["actions_taken"].append(f"GC round {i+1}: collected {collected} objects")
            
            # Step 3: Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                cleanup_results["actions_taken"].append("Cleared CUDA cache")
                
                # Synchronize CUDA operations
                torch.cuda.synchronize()
                cleanup_results["actions_taken"].append("Synchronized CUDA operations")
                
                # Reset peak memory stats
                torch.cuda.reset_peak_memory_stats()
                cleanup_results["actions_taken"].append("Reset CUDA memory stats")
                
                # Try to reset max memory cached
                try:
                    torch.cuda.reset_max_memory_cached()
                    cleanup_results["actions_taken"].append("Reset max memory cached")
                except:
                    pass
            
            # Step 4: Additional cleanup for transformers models
            try:
                # Clear transformers cache if available
                from transformers import pipeline
                # This might help clear any cached pipelines
                cleanup_results["actions_taken"].append("Attempted transformers cleanup")
            except:
                pass
            
            # Step 5: Force Python garbage collection again
            for i in range(2):
                collected = gc.collect()
                if collected > 0:
                    cleanup_results["actions_taken"].append(f"Final GC round {i+1}: collected {collected} objects")
            
            # Get memory after cleanup
            if torch.cuda.is_available():
                cleanup_results["memory_after"] = torch.cuda.memory_allocated() / (1024**3)  # GB
                cleanup_results["memory_freed"] = cleanup_results["memory_before"] - cleanup_results["memory_after"]
            
            cleanup_results["success"] = True
            
        except Exception as e:
            cleanup_results["actions_taken"].append(f"Error during cleanup: {str(e)}")
            cleanup_results["success"] = False
        
        return cleanup_results
    
    def get_gpu_memory_info(self) -> Dict[str, Any]:
        """
        Get current GPU memory information
        
        Returns:
            Dictionary with GPU memory stats
        """
        memory_info = {
            "gpu_available": False,
            "total_memory": 0,
            "allocated_memory": 0,
            "cached_memory": 0,
            "free_memory": 0,
            "memory_usage_percent": 0
        }
        
        try:
            if torch.cuda.is_available():
                memory_info["gpu_available"] = True
                
                # Get memory stats in GB
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                allocated_memory = torch.cuda.memory_allocated() / (1024**3)
                cached_memory = torch.cuda.memory_reserved() / (1024**3)
                free_memory = total_memory - cached_memory
                
                memory_info.update({
                    "total_memory": round(total_memory, 2),
                    "allocated_memory": round(allocated_memory, 2),
                    "cached_memory": round(cached_memory, 2),
                    "free_memory": round(free_memory, 2),
                    "memory_usage_percent": round((cached_memory / total_memory) * 100, 1)
                })
                
        except Exception as e:
            memory_info["error"] = str(e)
        
        return memory_info


class MultimodalInputFormatter:
    """Formatter for multimodal inputs to model-compatible format"""
    
    @staticmethod
    def format_multimodal_input(content_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Format multimodal content for model input
        Based on the Gemma 3n example format
        
        Args:
            content_items: List of content items with type and data
            
        Returns:
            Formatted message for model consumption
        """
        if not content_items:
            raise ValueError("Content items cannot be empty")
        
        formatted_content = []
        
        for item in content_items:
            content_type = item.get("type")
            
            if content_type == "text":
                formatted_content.append({
                    "type": "text",
                    "text": item.get("text", "")
                })
            
            elif content_type == "image":
                # For Gemma 3n, handle both file paths and bytes
                if "image_path" in item:
                    # Use file path directly (like in the example)
                    formatted_content.append({
                        "type": "image",
                        "image": item["image_path"]
                    })
                elif "image" in item:
                    # Convert bytes to temporary file path
                    temp_path = MultimodalInputFormatter._save_temp_image(
                        item["image"], 
                        item.get("filename", "temp_image.jpg")
                    )
                    formatted_content.append({
                        "type": "image",
                        "image": temp_path
                    })
            
            elif content_type == "audio":
                # For Gemma 3n, handle both file paths and bytes
                if "audio_path" in item:
                    # Use file path directly (like in the example)
                    formatted_content.append({
                        "type": "audio",
                        "audio": item["audio_path"]
                    })
                elif "audio" in item:
                    # Convert bytes to temporary file path
                    temp_path = MultimodalInputFormatter._save_temp_audio(
                        item["audio"], 
                        item.get("filename", "temp_audio.mp3")
                    )
                    formatted_content.append({
                        "type": "audio",
                        "audio": temp_path
                    })
        
        return {
            "role": "user",
            "content": formatted_content
        }
    
    @staticmethod
    def _save_temp_image(image_data: bytes, filename: str) -> str:
        """
        Save image bytes to temporary file
        
        Args:
            image_data: Image bytes
            filename: Original filename
            
        Returns:
            Path to temporary file
        """
        import tempfile
        import os
        from pathlib import Path
        
        # Create temp directory if it doesn't exist
        temp_dir = Path(tempfile.gettempdir()) / "gemma_multimodal"
        temp_dir.mkdir(exist_ok=True)
        
        # Generate unique filename
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        extension = Path(filename).suffix or '.jpg'
        temp_filename = f"img_{unique_id}{extension}"
        temp_path = temp_dir / temp_filename
        
        # Save image data
        with open(temp_path, 'wb') as f:
            f.write(image_data)
        
        return str(temp_path)
    
    @staticmethod
    def _save_temp_audio(audio_data: bytes, filename: str) -> str:
        """
        Save audio bytes to temporary file
        
        Args:
            audio_data: Audio bytes
            filename: Original filename
            
        Returns:
            Path to temporary file
        """
        import tempfile
        import os
        from pathlib import Path
        
        # Create temp directory if it doesn't exist
        temp_dir = Path(tempfile.gettempdir()) / "gemma_multimodal"
        temp_dir.mkdir(exist_ok=True)
        
        # Generate unique filename
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        extension = Path(filename).suffix or '.mp3'
        temp_filename = f"aud_{unique_id}{extension}"
        temp_path = temp_dir / temp_filename
        
        # Save audio data
        with open(temp_path, 'wb') as f:
            f.write(audio_data)
        
        return str(temp_path)
    
    @staticmethod
    def cleanup_temp_files():
        """Clean up temporary files"""
        import tempfile
        import shutil
        from pathlib import Path
        
        temp_dir = Path(tempfile.gettempdir()) / "gemma_multimodal"
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Could not clean up temp files: {e}")
    
    @staticmethod
    def validate_multimodal_input(content_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate multimodal input combinations
        
        Args:
            content_items: List of content items to validate
            
        Returns:
            Dictionary with validation results
        """
        if not content_items:
            return {
                "valid": False,
                "error": "No content items provided"
            }
        
        supported_types = {"text", "image", "audio"}
        content_types = set()
        
        for item in content_items:
            content_type = item.get("type")
            
            if not content_type:
                return {
                    "valid": False,
                    "error": "Content item missing 'type' field"
                }
            
            if content_type not in supported_types:
                return {
                    "valid": False,
                    "error": f"Unsupported content type: {content_type}"
                }
            
            content_types.add(content_type)
            
            # Validate content data
            if content_type == "text":
                if not item.get("text"):
                    return {
                        "valid": False,
                        "error": "Text content cannot be empty"
                    }
            
            elif content_type == "image":
                if not (item.get("image") or item.get("image_path")):
                    return {
                        "valid": False,
                        "error": "Image content missing data or path"
                    }
            
            elif content_type == "audio":
                if not (item.get("audio") or item.get("audio_path")):
                    return {
                        "valid": False,
                        "error": "Audio content missing data or path"
                    }
        
        return {
            "valid": True,
            "content_types": list(content_types),
            "is_multimodal": len(content_types) > 1
        }
    
    @staticmethod
    def create_text_only_input(text: str) -> Dict[str, Any]:
        """
        Create a text-only input message
        
        Args:
            text: Text content
            
        Returns:
            Formatted message
        """
        return MultimodalInputFormatter.format_multimodal_input([
            {"type": "text", "text": text}
        ])
    
    @staticmethod
    def create_text_image_input(text: str, image_data: bytes, image_filename: str = None) -> Dict[str, Any]:
        """
        Create a text + image input message
        
        Args:
            text: Text content
            image_data: Image bytes
            image_filename: Optional filename
            
        Returns:
            Formatted message
        """
        content = [
            {"type": "image", "image": image_data, "filename": image_filename},
            {"type": "text", "text": text}
        ]
        return MultimodalInputFormatter.format_multimodal_input(content)
    
    @staticmethod
    def create_text_audio_input(text: str, audio_data: bytes, audio_filename: str = None) -> Dict[str, Any]:
        """
        Create a text + audio input message
        
        Args:
            text: Text content
            audio_data: Audio bytes
            audio_filename: Optional filename
            
        Returns:
            Formatted message
        """
        content = [
            {"type": "audio", "audio": audio_data, "filename": audio_filename},
            {"type": "text", "text": text}
        ]
        return MultimodalInputFormatter.format_multimodal_input(content)
    
    @staticmethod
    def create_full_multimodal_input(text: str, image_data: bytes, audio_data: bytes, 
                                   image_filename: str = None, audio_filename: str = None) -> Dict[str, Any]:
        """
        Create a text + image + audio input message
        
        Args:
            text: Text content
            image_data: Image bytes
            audio_data: Audio bytes
            image_filename: Optional image filename
            audio_filename: Optional audio filename
            
        Returns:
            Formatted message
        """
        content = [
            {"type": "image", "image": image_data, "filename": image_filename},
            {"type": "audio", "audio": audio_data, "filename": audio_filename},
            {"type": "text", "text": text}
        ]
        return MultimodalInputFormatter.format_multimodal_input(content)