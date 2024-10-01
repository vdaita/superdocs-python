from dataclasses import dataclass
from typing import List, Optional

@dataclass
class File:
    filepath: str
    original_content: str
    modified_content: Optional[str]