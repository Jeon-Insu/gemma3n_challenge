"""
Storage service for session persistence
"""
import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from src.models.data_models import SessionResult, PromptData


class StorageService:
    """Service for managing session storage and retrieval"""
    
    def __init__(self, storage_dir: str = "sessions"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.sessions_dir = self.storage_dir / "sessions"
        self.files_dir = self.storage_dir / "files"
        self.sessions_dir.mkdir(exist_ok=True)
        self.files_dir.mkdir(exist_ok=True)
    
    def save_session(self, session_data: Dict[str, Any]) -> str:
        """
        Save session data to storage
        
        Args:
            session_data: Dictionary containing session information
            
        Returns:
            Session ID of saved session
        """
        try:
            # Generate session ID if not provided
            session_id = session_data.get("session_id", str(uuid.uuid4()))
            session_data["session_id"] = session_id
            
            # Add timestamp if not present
            if "timestamp" not in session_data:
                session_data["timestamp"] = datetime.now().isoformat()
            
            # Create session file path
            session_file = self.sessions_dir / f"{session_id}.json"
            
            # Save session data
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            
            return session_id
            
        except Exception as e:
            raise RuntimeError(f"Failed to save session: {str(e)}")
    
    def load_session(self, session_id: str) -> Dict[str, Any]:
        """
        Load session data from storage
        
        Args:
            session_id: ID of session to load
            
        Returns:
            Dictionary containing session data
        """
        try:
            session_file = self.sessions_dir / f"{session_id}.json"
            
            if not session_file.exists():
                raise FileNotFoundError(f"Session {session_id} not found")
            
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            return session_data
            
        except FileNotFoundError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to load session {session_id}: {str(e)}")
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all saved sessions with metadata
        
        Returns:
            List of session metadata dictionaries
        """
        try:
            sessions = []
            
            for session_file in self.sessions_dir.glob("*.json"):
                try:
                    with open(session_file, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)
                    
                    # Extract metadata
                    metadata = {
                        "session_id": session_data.get("session_id", session_file.stem),
                        "timestamp": session_data.get("timestamp"),
                        "prompt_count": len(session_data.get("prompts", [])),
                        "response_count": len(session_data.get("responses", [])),
                        "model_config": session_data.get("model_config", {}),
                        "file_path": str(session_file)
                    }
                    
                    sessions.append(metadata)
                    
                except Exception as e:
                    # Skip corrupted session files
                    print(f"Warning: Skipping corrupted session file {session_file}: {e}")
                    continue
            
            # Sort by timestamp (newest first)
            sessions.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            return sessions
            
        except Exception as e:
            raise RuntimeError(f"Failed to list sessions: {str(e)}")
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and its associated files
        
        Args:
            session_id: ID of session to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            session_file = self.sessions_dir / f"{session_id}.json"
            
            if not session_file.exists():
                return False
            
            # Load session to get associated files
            try:
                session_data = self.load_session(session_id)
                
                # Delete associated files
                prompts = session_data.get("prompts", [])
                for prompt in prompts:
                    if isinstance(prompt, dict):
                        # Delete image files
                        if "image_path" in prompt:
                            image_path = Path(prompt["image_path"])
                            if image_path.exists():
                                image_path.unlink()
                        
                        # Delete audio files
                        if "audio_path" in prompt:
                            audio_path = Path(prompt["audio_path"])
                            if audio_path.exists():
                                audio_path.unlink()
                
            except Exception as e:
                print(f"Warning: Could not clean up associated files for session {session_id}: {e}")
            
            # Delete session file
            session_file.unlink()
            return True
            
        except Exception as e:
            print(f"Error deleting session {session_id}: {e}")
            return False
    
    def save_session_result(self, session_result: SessionResult) -> str:
        """
        Save a SessionResult object to storage
        
        Args:
            session_result: SessionResult object to save
            
        Returns:
            Session ID of saved session
        """
        if not session_result.validate():
            raise ValueError("Invalid session result data")
        
        session_data = session_result.to_dict()
        return self.save_session(session_data)
    
    def load_session_result(self, session_id: str) -> SessionResult:
        """
        Load a SessionResult object from storage
        
        Args:
            session_id: ID of session to load
            
        Returns:
            SessionResult object
        """
        session_data = self.load_session(session_id)
        return SessionResult.from_dict(session_data)
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics
        
        Returns:
            Dictionary with storage statistics
        """
        try:
            session_count = len(list(self.sessions_dir.glob("*.json")))
            
            # Calculate total storage size
            total_size = 0
            for file_path in self.storage_dir.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            
            return {
                "session_count": session_count,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "storage_dir": str(self.storage_dir),
                "sessions_dir": str(self.sessions_dir),
                "files_dir": str(self.files_dir)
            }
            
        except Exception as e:
            return {
                "error": f"Failed to get storage stats: {str(e)}"
            }
    
    def cleanup_old_sessions(self, days_old: int = 30) -> int:
        """
        Clean up sessions older than specified days
        
        Args:
            days_old: Number of days after which sessions are considered old
            
        Returns:
            Number of sessions deleted
        """
        try:
            from datetime import timedelta
            
            cutoff_date = datetime.now() - timedelta(days=days_old)
            deleted_count = 0
            
            sessions = self.list_sessions()
            
            for session in sessions:
                try:
                    if session.get("timestamp"):
                        session_date = datetime.fromisoformat(session["timestamp"])
                        if session_date < cutoff_date:
                            if self.delete_session(session["session_id"]):
                                deleted_count += 1
                except Exception as e:
                    print(f"Warning: Could not process session {session.get('session_id')}: {e}")
                    continue
            
            return deleted_count
            
        except Exception as e:
            raise RuntimeError(f"Failed to cleanup old sessions: {str(e)}")
    
    def export_session_to_json(self, session_id: str, output_path: str) -> bool:
        """
        Export session to a JSON file
        
        Args:
            session_id: ID of session to export
            output_path: Path where to save the exported file
            
        Returns:
            True if export was successful, False otherwise
        """
        try:
            session_data = self.load_session(session_id)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"Error exporting session {session_id}: {e}")
            return False


class FileManager:
    """Manager for multimodal file storage and retrieval"""
    
    def __init__(self, storage_service: StorageService):
        self.storage_service = storage_service
        self.files_dir = storage_service.files_dir
        
        # Create subdirectories for different file types
        self.images_dir = self.files_dir / "images"
        self.audio_dir = self.files_dir / "audio"
        self.images_dir.mkdir(exist_ok=True)
        self.audio_dir.mkdir(exist_ok=True)
    
    def save_image_file(self, image_data: bytes, filename: str = None, session_id: str = None) -> str:
        """
        Save image data to file storage
        
        Args:
            image_data: Image bytes
            filename: Original filename (optional)
            session_id: Associated session ID (optional)
            
        Returns:
            Path to saved image file
        """
        try:
            # Generate unique filename
            file_id = str(uuid.uuid4())
            
            # Determine file extension
            if filename:
                extension = Path(filename).suffix.lower()
                if not extension:
                    extension = '.jpg'  # Default extension
            else:
                extension = '.jpg'
            
            # Create filename with session prefix if provided
            if session_id:
                saved_filename = f"{session_id}_{file_id}{extension}"
            else:
                saved_filename = f"{file_id}{extension}"
            
            # Save file
            file_path = self.images_dir / saved_filename
            with open(file_path, 'wb') as f:
                f.write(image_data)
            
            return str(file_path)
            
        except Exception as e:
            raise RuntimeError(f"Failed to save image file: {str(e)}")
    
    def save_audio_file(self, audio_data: bytes, filename: str = None, session_id: str = None) -> str:
        """
        Save audio data to file storage
        
        Args:
            audio_data: Audio bytes
            filename: Original filename (optional)
            session_id: Associated session ID (optional)
            
        Returns:
            Path to saved audio file
        """
        try:
            # Generate unique filename
            file_id = str(uuid.uuid4())
            
            # Determine file extension
            if filename:
                extension = Path(filename).suffix.lower()
                if not extension:
                    extension = '.mp3'  # Default extension
            else:
                extension = '.mp3'
            
            # Create filename with session prefix if provided
            if session_id:
                saved_filename = f"{session_id}_{file_id}{extension}"
            else:
                saved_filename = f"{file_id}{extension}"
            
            # Save file
            file_path = self.audio_dir / saved_filename
            with open(file_path, 'wb') as f:
                f.write(audio_data)
            
            return str(file_path)
            
        except Exception as e:
            raise RuntimeError(f"Failed to save audio file: {str(e)}")
    
    def load_file(self, file_path: str) -> bytes:
        """
        Load file data from storage
        
        Args:
            file_path: Path to file
            
        Returns:
            File data as bytes
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            with open(path, 'rb') as f:
                return f.read()
                
        except Exception as e:
            raise RuntimeError(f"Failed to load file {file_path}: {str(e)}")
    
    def delete_file(self, file_path: str) -> bool:
        """
        Delete a file from storage
        
        Args:
            file_path: Path to file to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                return False
            
            path.unlink()
            return True
            
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
            return False
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get information about a file
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with file information
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                return {"exists": False, "error": "File not found"}
            
            stat = path.stat()
            
            return {
                "exists": True,
                "filename": path.name,
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "extension": path.suffix.lower(),
                "file_type": self._get_file_type(path.suffix.lower())
            }
            
        except Exception as e:
            return {"exists": False, "error": str(e)}
    
    def _get_file_type(self, extension: str) -> str:
        """
        Determine file type from extension
        
        Args:
            extension: File extension
            
        Returns:
            File type string
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        audio_extensions = {'.mp3', '.wav', '.m4a', '.ogg', '.flac'}
        
        if extension in image_extensions:
            return "image"
        elif extension in audio_extensions:
            return "audio"
        else:
            return "unknown"
    
    def cleanup_session_files(self, session_id: str) -> int:
        """
        Clean up all files associated with a session
        
        Args:
            session_id: Session ID
            
        Returns:
            Number of files deleted
        """
        try:
            deleted_count = 0
            
            # Clean up image files
            for image_file in self.images_dir.glob(f"{session_id}_*"):
                if self.delete_file(str(image_file)):
                    deleted_count += 1
            
            # Clean up audio files
            for audio_file in self.audio_dir.glob(f"{session_id}_*"):
                if self.delete_file(str(audio_file)):
                    deleted_count += 1
            
            return deleted_count
            
        except Exception as e:
            print(f"Error cleaning up session files for {session_id}: {e}")
            return 0
    
    def get_storage_usage(self) -> Dict[str, Any]:
        """
        Get storage usage statistics
        
        Returns:
            Dictionary with storage usage information
        """
        try:
            image_count = 0
            image_size = 0
            audio_count = 0
            audio_size = 0
            
            # Count image files
            for image_file in self.images_dir.rglob("*"):
                if image_file.is_file():
                    image_count += 1
                    image_size += image_file.stat().st_size
            
            # Count audio files
            for audio_file in self.audio_dir.rglob("*"):
                if audio_file.is_file():
                    audio_count += 1
                    audio_size += audio_file.stat().st_size
            
            total_size = image_size + audio_size
            
            return {
                "image_files": image_count,
                "image_size_bytes": image_size,
                "image_size_mb": round(image_size / (1024 * 1024), 2),
                "audio_files": audio_count,
                "audio_size_bytes": audio_size,
                "audio_size_mb": round(audio_size / (1024 * 1024), 2),
                "total_files": image_count + audio_count,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2)
            }
            
        except Exception as e:
            return {"error": f"Failed to get storage usage: {str(e)}"}
    
    def cleanup_orphaned_files(self) -> int:
        """
        Clean up files that are not associated with any session
        
        Returns:
            Number of orphaned files deleted
        """
        try:
            # Get all session IDs
            sessions = self.storage_service.list_sessions()
            session_ids = {session["session_id"] for session in sessions}
            
            deleted_count = 0
            
            # Check image files
            for image_file in self.images_dir.rglob("*"):
                if image_file.is_file():
                    # Extract session ID from filename
                    filename = image_file.name
                    if '_' in filename:
                        session_id = filename.split('_')[0]
                        if session_id not in session_ids:
                            if self.delete_file(str(image_file)):
                                deleted_count += 1
            
            # Check audio files
            for audio_file in self.audio_dir.rglob("*"):
                if audio_file.is_file():
                    # Extract session ID from filename
                    filename = audio_file.name
                    if '_' in filename:
                        session_id = filename.split('_')[0]
                        if session_id not in session_ids:
                            if self.delete_file(str(audio_file)):
                                deleted_count += 1
            
            return deleted_count
            
        except Exception as e:
            print(f"Error cleaning up orphaned files: {e}")
            return 0