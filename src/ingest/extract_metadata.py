"""
Metadata extraction module for digital forensics evidence.
Handles EML, JSON, E01, and other forensic file formats.
"""

import os
import json
import email
from typing import Dict, Any, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Extract metadata from various digital forensics file formats."""
    
    def __init__(self):
        self.supported_formats = ['.eml', '.json', '.e01', '.msg', '.pst']
    
    def extract_from_eml(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from EML email files."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                msg = email.message_from_file(f)
            
            metadata = {
                'file_type': 'email',
                'subject': msg.get('Subject', ''),
                'from': msg.get('From', ''),
                'to': msg.get('To', ''),
                'date': msg.get('Date', ''),
                'message_id': msg.get('Message-ID', ''),
                'file_path': file_path,
                'file_size': os.path.getsize(file_path)
            }
            
            # Extract body content
            if msg.is_multipart():
                parts = []
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        parts.append(part.get_payload(decode=True).decode('utf-8', errors='ignore'))
                metadata['body'] = '\n'.join(parts)
            else:
                metadata['body'] = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting EML metadata from {file_path}: {e}")
            return {}
    
    def extract_from_json(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from JSON log files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            metadata = {
                'file_type': 'json_log',
                'file_path': file_path,
                'file_size': os.path.getsize(file_path),
                'record_count': len(data) if isinstance(data, list) else 1,
                'data': data
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting JSON metadata from {file_path}: {e}")
            return {}
    
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata based on file extension."""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.eml':
            return self.extract_from_eml(file_path)
        elif file_ext == '.json':
            return self.extract_from_json(file_path)
        else:
            logger.warning(f"Unsupported file format: {file_ext}")
            return {
                'file_type': 'unknown',
                'file_path': file_path,
                'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0
            }
    
    def batch_extract(self, directory: str) -> List[Dict[str, Any]]:
        """Extract metadata from all supported files in a directory."""
        metadata_list = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                if Path(file_path).suffix.lower() in self.supported_formats:
                    metadata = self.extract_metadata(file_path)
                    if metadata:
                        metadata_list.append(metadata)
        
        return metadata_list


if __name__ == "__main__":
    # Example usage
    extractor = MetadataExtractor()
    
    # Extract from single file
    # metadata = extractor.extract_metadata("path/to/evidence.eml")
    
    # Batch extract from directory
    # metadata_list = extractor.batch_extract("path/to/evidence/directory")
    
    print("Metadata extractor ready for use.")
