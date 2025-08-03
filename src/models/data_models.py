"""
Core data models for the multimodal Streamlit app
"""
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any
import json
import uuid


@dataclass
class PromptData:
    """Data model for individual prompts with multimodal content"""
    id: str
    text: Optional[str] = None
    image: Optional[bytes] = None
    audio: Optional[bytes] = None
    image_filename: Optional[str] = None
    audio_filename: Optional[str] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.id is None:
            self.id = str(uuid.uuid4())
    
    def validate(self) -> bool:
        """Validate that the prompt has at least one input type"""
        return bool(self.text or self.image or self.audio)
    
    def get_content_types(self) -> List[str]:
        """Get list of content types present in this prompt"""
        types = []
        if self.text:
            types.append("text")
        if self.image:
            types.append("image")
        if self.audio:
            types.append("audio")
        return types
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        # Convert bytes to base64 for JSON serialization if needed
        if self.image:
            import base64
            data['image'] = base64.b64encode(self.image).decode('utf-8')
        if self.audio:
            import base64
            data['audio'] = base64.b64encode(self.audio).decode('utf-8')
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptData':
        """Create instance from dictionary"""
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        # Convert base64 back to bytes if needed
        if 'image' in data and isinstance(data['image'], str):
            import base64
            data['image'] = base64.b64decode(data['image'])
        if 'audio' in data and isinstance(data['audio'], str):
            import base64
            data['audio'] = base64.b64decode(data['audio'])
        
        return cls(**data)


@dataclass
class SessionResult:
    """Data model for storing session results"""
    session_id: str
    prompts: List[PromptData]
    responses: List[str]
    created_at: datetime
    model_config: Dict[str, Any]
    
    def __post_init__(self):
        if self.session_id is None:
            self.session_id = str(uuid.uuid4())
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def validate(self) -> bool:
        """Validate session result data"""
        if not self.prompts or not self.responses:
            return False
        if len(self.prompts) != len(self.responses):
            return False
        return all(prompt.validate() for prompt in self.prompts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'session_id': self.session_id,
            'prompts': [prompt.to_dict() for prompt in self.prompts],
            'responses': self.responses,
            'created_at': self.created_at.isoformat(),
            'model_config': self.model_config
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionResult':
        """Create instance from dictionary"""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['prompts'] = [PromptData.from_dict(p) for p in data['prompts']]
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'SessionResult':
        """Create instance from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)