"""
99 Flags Roll-up Aggregator - Collect all flags into global FLAGS line.
"""

import re
from typing import Set

from loguru import logger

from neuralshield.preprocessing.http_preprocessor import HttpPreprocessor


class FlagsRollup(HttpPreprocessor):
    """
    Aggregate all flags emitted by previous steps into a global FLAGS:[...] line.
    
    This step:
    - Collects all flag lines from the request
    - Deduplicates flags
    - Sorts flags alphabetically
    - Emits a single FLAGS:[...] line at the end
    - Removes individual flag lines to avoid duplication
    """

    def _is_flag_line(self, line: str) -> bool:
        """
        Determine if a line is a flag line (not a structured line with brackets).
        
        Args:
            line: Line to check
            
        Returns:
            True if this appears to be a flag line
        """
        line = line.strip()
        if not line:
            return False
            
        # Skip lines that start with brackets (structured content)
        if line.startswith("["):
            return False
            
        # Check if line contains flag-like patterns
        # Flags are typically uppercase words, possibly with colons and parameters
        flag_pattern = re.compile(r'^[A-Z_][A-Z0-9_]*(?::[A-Za-z0-9_<>.-]+)?(?:\s+[A-Z_][A-Z0-9_]*(?::[A-Za-z0-9_<>.-]+)?)*$')
        return bool(flag_pattern.match(line))

    def _extract_flags_from_line(self, line: str) -> list[str]:
        """
        Extract individual flags from a flag line.
        
        Args:
            line: Flag line (e.g., "FULLWIDTH CONTROL" or "QREPEAT:param")
            
        Returns:
            List of individual flags
        """
        return line.strip().split()

    def process(self, request: str) -> str:
        """
        Process request to aggregate all flags into a single FLAGS line.
        
        Args:
            request: Processed HTTP request with individual flag lines
            
        Returns:
            Request with individual flags replaced by single FLAGS:[...] line
        """
        lines = request.strip().split("\n")
        processed_lines = []
        all_flags: Set[str] = set()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if self._is_flag_line(line):
                # Extract flags from this line and add to collection
                flags = self._extract_flags_from_line(line)
                all_flags.update(flags)
                logger.debug(f"Collected flags from line: {flags}")
            else:
                # Keep non-flag lines
                processed_lines.append(line)
        
        # Add FLAGS line at the end if we have any flags
        if all_flags:
            sorted_flags = sorted(all_flags)
            flags_line = f"FLAGS:[{' '.join(sorted_flags)}]"
            processed_lines.append(flags_line)
            logger.debug(f"Generated flags rollup: {len(sorted_flags)} unique flags")
        
        return "\n".join(processed_lines)
