from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class DataIngestionArtifact:
    trained_file_path: str
    test_file_path: str