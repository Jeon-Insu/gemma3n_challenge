"""
Batch processing service for managing multiple prompts
"""
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
from src.models.data_models import PromptData


class PromptBatch:
    """Class for managing multiple prompts in a batch"""
    
    def __init__(self, batch_id: str = None):
        self.batch_id = batch_id or str(uuid.uuid4())
        self.prompts: List[PromptData] = []
        self.created_at = datetime.now()
        self.shared_image = None
        self.shared_audio = None
        self.shared_image_filename = None
        self.shared_audio_filename = None
    
    def add_prompt(self, text: str = None, image: bytes = None, audio: bytes = None,
                   image_filename: str = None, audio_filename: str = None) -> str:
        """
        Add a prompt to the batch
        
        Args:
            text: Text content
            image: Image bytes (optional)
            audio: Audio bytes (optional)
            image_filename: Image filename (optional)
            audio_filename: Audio filename (optional)
            
        Returns:
            ID of the added prompt
        """
        prompt = PromptData(
            id=str(uuid.uuid4()),
            text=text,
            image=image,
            audio=audio,
            image_filename=image_filename,
            audio_filename=audio_filename
        )
        
        self.prompts.append(prompt)
        return prompt.id
    
    def add_text_prompt(self, text: str) -> str:
        """
        Add a text-only prompt to the batch
        
        Args:
            text: Text content
            
        Returns:
            ID of the added prompt
        """
        return self.add_prompt(text=text)
    
    def remove_prompt(self, prompt_id: str) -> bool:
        """
        Remove a prompt from the batch
        
        Args:
            prompt_id: ID of prompt to remove
            
        Returns:
            True if prompt was removed, False if not found
        """
        for i, prompt in enumerate(self.prompts):
            if prompt.id == prompt_id:
                self.prompts.pop(i)
                return True
        return False
    
    def get_prompt(self, prompt_id: str) -> Optional[PromptData]:
        """
        Get a specific prompt by ID
        
        Args:
            prompt_id: ID of prompt to retrieve
            
        Returns:
            PromptData object or None if not found
        """
        for prompt in self.prompts:
            if prompt.id == prompt_id:
                return prompt
        return None
    
    def get_all_prompts(self) -> List[PromptData]:
        """
        Get all prompts in the batch
        
        Returns:
            List of PromptData objects
        """
        return self.prompts.copy()
    
    def set_shared_image(self, image: bytes, filename: str = None) -> None:
        """
        Set a shared image for all prompts in the batch
        
        Args:
            image: Image bytes
            filename: Image filename (optional)
        """
        self.shared_image = image
        self.shared_image_filename = filename
    
    def set_shared_audio(self, audio: bytes, filename: str = None) -> None:
        """
        Set a shared audio for all prompts in the batch
        
        Args:
            audio: Audio bytes
            filename: Audio filename (optional)
        """
        self.shared_audio = audio
        self.shared_audio_filename = filename
    
    def clear_shared_media(self) -> None:
        """Clear shared image and audio"""
        self.shared_image = None
        self.shared_audio = None
        self.shared_image_filename = None
        self.shared_audio_filename = None
    
    def get_combined_prompts(self) -> List[PromptData]:
        """
        Get prompts combined with shared media
        
        Returns:
            List of PromptData objects with shared media applied
        """
        combined_prompts = []
        
        for prompt in self.prompts:
            # Create a copy of the prompt
            combined_prompt = PromptData(
                id=prompt.id,
                text=prompt.text,
                image=prompt.image or self.shared_image,
                audio=prompt.audio or self.shared_audio,
                image_filename=prompt.image_filename or self.shared_image_filename,
                audio_filename=prompt.audio_filename or self.shared_audio_filename,
                created_at=prompt.created_at
            )
            
            combined_prompts.append(combined_prompt)
        
        return combined_prompts
    
    def validate_batch(self) -> Dict[str, Any]:
        """
        Validate the batch and all its prompts
        
        Returns:
            Dictionary with validation results
        """
        if not self.prompts:
            return {
                "valid": False,
                "error": "Batch contains no prompts"
            }
        
        invalid_prompts = []
        for prompt in self.prompts:
            if not prompt.validate():
                invalid_prompts.append(prompt.id)
        
        if invalid_prompts:
            return {
                "valid": False,
                "error": f"Invalid prompts found: {', '.join(invalid_prompts)}"
            }
        
        return {
            "valid": True,
            "prompt_count": len(self.prompts),
            "has_shared_image": self.shared_image is not None,
            "has_shared_audio": self.shared_audio is not None
        }
    
    def get_batch_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the batch
        
        Returns:
            Dictionary with batch statistics
        """
        text_only_count = 0
        image_count = 0
        audio_count = 0
        multimodal_count = 0
        
        combined_prompts = self.get_combined_prompts()
        
        for prompt in combined_prompts:
            content_types = prompt.get_content_types()
            
            if len(content_types) == 1 and "text" in content_types:
                text_only_count += 1
            elif len(content_types) > 1:
                multimodal_count += 1
            
            if "image" in content_types:
                image_count += 1
            if "audio" in content_types:
                audio_count += 1
        
        return {
            "batch_id": self.batch_id,
            "total_prompts": len(self.prompts),
            "text_only_prompts": text_only_count,
            "prompts_with_image": image_count,
            "prompts_with_audio": audio_count,
            "multimodal_prompts": multimodal_count,
            "has_shared_image": self.shared_image is not None,
            "has_shared_audio": self.shared_audio is not None,
            "created_at": self.created_at.isoformat()
        }
    
    def clear_batch(self) -> None:
        """Clear all prompts and shared media from the batch"""
        self.prompts.clear()
        self.clear_shared_media()
    
    def duplicate_prompt(self, prompt_id: str) -> Optional[str]:
        """
        Duplicate an existing prompt in the batch
        
        Args:
            prompt_id: ID of prompt to duplicate
            
        Returns:
            ID of the new duplicated prompt, or None if original not found
        """
        original_prompt = self.get_prompt(prompt_id)
        if not original_prompt:
            return None
        
        new_prompt = PromptData(
            id=str(uuid.uuid4()),
            text=original_prompt.text,
            image=original_prompt.image,
            audio=original_prompt.audio,
            image_filename=original_prompt.image_filename,
            audio_filename=original_prompt.audio_filename
        )
        
        self.prompts.append(new_prompt)
        return new_prompt.id
    
    def reorder_prompts(self, new_order: List[str]) -> bool:
        """
        Reorder prompts based on provided ID list
        
        Args:
            new_order: List of prompt IDs in desired order
            
        Returns:
            True if reordering was successful, False otherwise
        """
        if len(new_order) != len(self.prompts):
            return False
        
        # Check that all IDs are valid
        current_ids = {prompt.id for prompt in self.prompts}
        if set(new_order) != current_ids:
            return False
        
        # Create new prompt list in specified order
        reordered_prompts = []
        for prompt_id in new_order:
            prompt = self.get_prompt(prompt_id)
            if prompt:
                reordered_prompts.append(prompt)
        
        self.prompts = reordered_prompts
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert batch to dictionary for serialization
        
        Returns:
            Dictionary representation of the batch
        """
        return {
            "batch_id": self.batch_id,
            "prompts": [prompt.to_dict() for prompt in self.prompts],
            "created_at": self.created_at.isoformat(),
            "shared_image": self.shared_image,
            "shared_audio": self.shared_audio,
            "shared_image_filename": self.shared_image_filename,
            "shared_audio_filename": self.shared_audio_filename
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptBatch':
        """
        Create PromptBatch from dictionary
        
        Args:
            data: Dictionary containing batch data
            
        Returns:
            PromptBatch instance
        """
        batch = cls(batch_id=data.get("batch_id"))
        
        if "created_at" in data:
            batch.created_at = datetime.fromisoformat(data["created_at"])
        
        # Load prompts
        for prompt_data in data.get("prompts", []):
            prompt = PromptData.from_dict(prompt_data)
            batch.prompts.append(prompt)
        
        # Load shared media
        batch.shared_image = data.get("shared_image")
        batch.shared_audio = data.get("shared_audio")
        batch.shared_image_filename = data.get("shared_image_filename")
        batch.shared_audio_filename = data.get("shared_audio_filename")
        
        return batch


class BatchExecutor:
    """Executor for processing batches of prompts with progress tracking"""
    
    def __init__(self, model_service, storage_service=None, file_manager=None):
        self.model_service = model_service
        self.storage_service = storage_service
        self.file_manager = file_manager
        self.current_batch = None
        self.execution_results = []
        self.progress_callback = None
    
    def set_progress_callback(self, callback) -> None:
        """
        Set callback function for progress updates
        
        Args:
            callback: Function that takes (current, total, message) parameters
        """
        self.progress_callback = callback
    
    def _update_progress(self, current: int, total: int, message: str = "") -> None:
        """
        Update progress if callback is set
        
        Args:
            current: Current progress count
            total: Total items to process
            message: Progress message
        """
        if self.progress_callback:
            self.progress_callback(current, total, message)
    
    def execute_batch(self, batch: PromptBatch, max_tokens: int = 256, 
                     save_results: bool = True) -> Dict[str, Any]:
        """
        Execute a batch of prompts
        
        Args:
            batch: PromptBatch to execute
            max_tokens: Maximum tokens per response
            save_results: Whether to save results to storage
            
        Returns:
            Dictionary with execution results
        """
        if not self.model_service.is_model_ready():
            raise RuntimeError("Model service is not ready")
        
        # Validate batch
        validation = batch.validate_batch()
        if not validation["valid"]:
            raise ValueError(f"Invalid batch: {validation['error']}")
        
        self.current_batch = batch
        self.execution_results = []
        
        # Get combined prompts (with shared media applied)
        combined_prompts = batch.get_combined_prompts()
        total_prompts = len(combined_prompts)
        
        self._update_progress(0, total_prompts, "Starting batch execution...")
        
        start_time = datetime.now()
        successful_executions = 0
        failed_executions = 0
        
        try:
            # Process each prompt
            for i, prompt_data in enumerate(combined_prompts):
                try:
                    self._update_progress(
                        i, total_prompts, 
                        f"Processing prompt {i + 1} of {total_prompts}..."
                    )
                    
                    # Convert PromptData to model input format
                    model_input = self._convert_prompt_to_model_input(prompt_data)
                    
                    # Execute prompt
                    response = self.model_service.process_prompt(model_input, max_tokens)
                    
                    # Store result
                    result = {
                        "prompt_id": prompt_data.id,
                        "prompt_data": prompt_data,
                        "response": response,
                        "success": True,
                        "error": None,
                        "execution_time": datetime.now()
                    }
                    
                    self.execution_results.append(result)
                    successful_executions += 1
                    
                except Exception as e:
                    # Handle individual prompt failure
                    error_result = {
                        "prompt_id": prompt_data.id,
                        "prompt_data": prompt_data,
                        "response": None,
                        "success": False,
                        "error": str(e),
                        "execution_time": datetime.now()
                    }
                    
                    self.execution_results.append(error_result)
                    failed_executions += 1
        
        finally:
            # Clean up temporary files after batch execution
            try:
                from src.services.model_service import MultimodalInputFormatter
                MultimodalInputFormatter.cleanup_temp_files()
            except Exception as e:
                print(f"Warning: Could not clean up temporary files: {e}")
        
        end_time = datetime.now()
        execution_duration = (end_time - start_time).total_seconds()
        
        self._update_progress(
            total_prompts, total_prompts, 
            f"Batch execution completed! {successful_executions} successful, {failed_executions} failed"
        )
        
        # Prepare execution summary
        execution_summary = {
            "batch_id": batch.batch_id,
            "total_prompts": total_prompts,
            "successful_executions": successful_executions,
            "failed_executions": failed_executions,
            "execution_duration_seconds": execution_duration,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "results": self.execution_results
        }
        
        # Save results if requested
        if save_results and self.storage_service:
            try:
                session_id = self._save_execution_results(execution_summary)
                execution_summary["saved_session_id"] = session_id
            except Exception as e:
                execution_summary["save_error"] = str(e)
        
        return execution_summary
    
    def _convert_prompt_to_model_input(self, prompt_data: PromptData) -> Dict[str, Any]:
        """
        Convert PromptData to model input format
        
        Args:
            prompt_data: PromptData object
            
        Returns:
            Dictionary formatted for model input
        """
        from src.services.model_service import MultimodalInputFormatter
        
        content_items = []
        
        # Add text content
        if prompt_data.text:
            content_items.append({
                "type": "text",
                "text": prompt_data.text
            })
        
        # Add image content
        if prompt_data.image:
            content_items.append({
                "type": "image",
                "image": prompt_data.image,
                "filename": prompt_data.image_filename
            })
        
        # Add audio content
        if prompt_data.audio:
            content_items.append({
                "type": "audio",
                "audio": prompt_data.audio,
                "filename": prompt_data.audio_filename
            })
        
        return MultimodalInputFormatter.format_multimodal_input(content_items)
    
    def _save_execution_results(self, execution_summary: Dict[str, Any]) -> str:
        """
        Save execution results to storage
        
        Args:
            execution_summary: Summary of batch execution
            
        Returns:
            Session ID of saved results
        """
        from src.models.data_models import SessionResult
        
        # Extract prompts and responses
        prompts = []
        responses = []
        
        for result in execution_summary["results"]:
            prompts.append(result["prompt_data"])
            responses.append(result["response"] if result["success"] else f"ERROR: {result['error']}")
        
        # Create SessionResult
        session_result = SessionResult(
            session_id=f"batch_{execution_summary['batch_id']}_{int(datetime.now().timestamp())}",
            prompts=prompts,
            responses=responses,
            created_at=datetime.fromisoformat(execution_summary["start_time"]),
            model_config={
                "batch_execution": True,
                "execution_duration": execution_summary["execution_duration_seconds"],
                "successful_executions": execution_summary["successful_executions"],
                "failed_executions": execution_summary["failed_executions"]
            }
        )
        
        return self.storage_service.save_session_result(session_result)
    
    def get_last_execution_results(self) -> List[Dict[str, Any]]:
        """
        Get results from the last batch execution
        
        Returns:
            List of execution results
        """
        return self.execution_results.copy()
    
    def cancel_execution(self) -> bool:
        """
        Cancel current batch execution (placeholder for future implementation)
        
        Returns:
            True if cancellation was successful
        """
        # This is a placeholder - actual cancellation would require
        # threading or async implementation
        return False
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the last execution
        
        Returns:
            Dictionary with execution statistics
        """
        if not self.execution_results:
            return {"no_executions": True}
        
        successful = sum(1 for result in self.execution_results if result["success"])
        failed = len(self.execution_results) - successful
        
        # Calculate average response length for successful executions
        successful_responses = [
            result["response"] for result in self.execution_results 
            if result["success"] and result["response"]
        ]
        
        avg_response_length = 0
        if successful_responses:
            avg_response_length = sum(len(response) for response in successful_responses) / len(successful_responses)
        
        return {
            "total_executions": len(self.execution_results),
            "successful_executions": successful,
            "failed_executions": failed,
            "success_rate": successful / len(self.execution_results) if self.execution_results else 0,
            "average_response_length": round(avg_response_length, 2),
            "batch_id": self.current_batch.batch_id if self.current_batch else None
        }