import hashlib
from typing import List
from typing import Any


def calculate_knowledge_hash(urls: List[str], files: List[Any]) -> str:
    """Calculate a hash for the current knowledge sources to detect changes."""
    content = "|".join(sorted(urls))
    for file in files:
        content += file.getvalue().decode(errors="ignore")
    return hashlib.md5(content.encode()).hexdigest()
