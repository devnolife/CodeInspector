"""
Code Preprocessor Module
Handles code cleaning, standardization, and normalization.
"""

import re
from typing import List, Dict, Optional


class CodePreprocessor:
    """Preprocesses code for evaluation."""

    def __init__(self):
        """Initialize Code Preprocessor."""
        pass

    def remove_comments(self, code: str, language: str = 'python') -> str:
        """
        Remove comments from code.

        Args:
            code: Source code string
            language: Programming language (python, java, javascript, cpp)

        Returns:
            Code without comments
        """
        if language in ['python']:
            # Remove single-line comments
            code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
            # Remove multi-line comments (docstrings)
            code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
            code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)

        elif language in ['java', 'javascript', 'cpp', 'c', 'typescript']:
            # Remove single-line comments
            code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
            # Remove multi-line comments
            code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)

        return code

    def remove_blank_lines(self, code: str) -> str:
        """
        Remove blank lines and excessive whitespace.

        Args:
            code: Source code string

        Returns:
            Code without blank lines
        """
        # Remove lines that are only whitespace
        lines = [line for line in code.split('\n') if line.strip()]
        return '\n'.join(lines)

    def normalize_whitespace(self, code: str) -> str:
        """
        Normalize whitespace in code.

        Args:
            code: Source code string

        Returns:
            Code with normalized whitespace
        """
        # Replace multiple spaces with single space
        code = re.sub(r' +', ' ', code)
        # Replace multiple newlines with single newline
        code = re.sub(r'\n+', '\n', code)
        # Remove trailing whitespace
        code = '\n'.join(line.rstrip() for line in code.split('\n'))
        return code.strip()

    def remove_imports(self, code: str, language: str = 'python') -> str:
        """
        Remove import statements from code.

        Args:
            code: Source code string
            language: Programming language

        Returns:
            Code without import statements
        """
        if language == 'python':
            # Remove import and from...import statements
            code = re.sub(r'^import .*$', '', code, flags=re.MULTILINE)
            code = re.sub(r'^from .* import .*$', '', code, flags=re.MULTILINE)

        elif language in ['java']:
            code = re.sub(r'^import .*$', '', code, flags=re.MULTILINE)

        elif language in ['javascript', 'typescript']:
            code = re.sub(r'^import .*$', '', code, flags=re.MULTILINE)
            code = re.sub(r"^import .*from .*$", '', code, flags=re.MULTILINE)

        return code

    def extract_functions(self, code: str, language: str = 'python') -> List[str]:
        """
        Extract function definitions from code.

        Args:
            code: Source code string
            language: Programming language

        Returns:
            List of function code blocks
        """
        functions = []

        if language == 'python':
            # Match function definitions with their bodies (basic implementation)
            pattern = r'def\s+\w+\s*\([^)]*\):\s*\n(?:(?:    |\t).*\n)*'
            functions = re.findall(pattern, code)

        elif language == 'java':
            # Match method definitions
            pattern = r'(?:public|private|protected)?\s*(?:static)?\s*\w+\s+\w+\s*\([^)]*\)\s*\{[^}]*\}'
            functions = re.findall(pattern, code, re.DOTALL)

        elif language in ['javascript', 'typescript']:
            # Match function declarations
            pattern = r'function\s+\w+\s*\([^)]*\)\s*\{[^}]*\}'
            functions.extend(re.findall(pattern, code, re.DOTALL))
            # Match arrow functions
            pattern = r'const\s+\w+\s*=\s*\([^)]*\)\s*=>\s*\{[^}]*\}'
            functions.extend(re.findall(pattern, code, re.DOTALL))

        return functions

    def standardize_code(
        self,
        code: str,
        language: str = 'python',
        remove_comments: bool = True,
        remove_imports: bool = False,
        normalize_ws: bool = True,
        remove_blank: bool = True
    ) -> str:
        """
        Standardize code by applying multiple preprocessing steps.

        Args:
            code: Source code string
            language: Programming language
            remove_comments: Whether to remove comments
            remove_imports: Whether to remove import statements
            normalize_ws: Whether to normalize whitespace
            remove_blank: Whether to remove blank lines

        Returns:
            Standardized code
        """
        processed_code = code

        if remove_comments:
            processed_code = self.remove_comments(processed_code, language)

        if remove_imports:
            processed_code = self.remove_imports(processed_code, language)

        if remove_blank:
            processed_code = self.remove_blank_lines(processed_code)

        if normalize_ws:
            processed_code = self.normalize_whitespace(processed_code)

        return processed_code

    def preprocess_for_codebert(self, code: str, language: str = 'python') -> str:
        """
        Preprocess code specifically for CodeBERT input.

        Args:
            code: Source code string
            language: Programming language

        Returns:
            Preprocessed code ready for CodeBERT
        """
        # For CodeBERT, we typically keep the structure but normalize formatting
        processed = self.standardize_code(
            code,
            language=language,
            remove_comments=False,  # Keep comments as they may contain semantic info
            remove_imports=False,   # Keep imports
            normalize_ws=True,
            remove_blank=True
        )

        # Truncate if too long (CodeBERT has token limits)
        max_length = 10000  # characters
        if len(processed) > max_length:
            processed = processed[:max_length]

        return processed

    def preprocess_for_token_similarity(self, code: str, language: str = 'python') -> str:
        """
        Preprocess code for token-based similarity comparison.

        Args:
            code: Source code string
            language: Programming language

        Returns:
            Preprocessed code ready for tokenization
        """
        # For token similarity, we remove noise but keep the core structure
        processed = self.standardize_code(
            code,
            language=language,
            remove_comments=True,   # Remove comments (noise for token matching)
            remove_imports=True,    # Remove imports (usually similar)
            normalize_ws=True,
            remove_blank=True
        )

        return processed

    def get_language_from_extension(self, extension: str) -> str:
        """
        Get language name from file extension.

        Args:
            extension: File extension (e.g., '.py', '.java')

        Returns:
            Language name
        """
        language_map = {
            '.py': 'python',
            '.java': 'java',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php'
        }

        return language_map.get(extension.lower(), 'unknown')

    def merge_code_files(self, code_files: List[Dict[str, str]]) -> str:
        """
        Merge multiple code files into a single string.

        Args:
            code_files: List of dictionaries with 'path' and 'content' keys

        Returns:
            Merged code string
        """
        merged = []

        for file_info in code_files:
            path = file_info.get('path', 'unknown')
            content = file_info.get('content', '')

            merged.append(f"# File: {path}")
            merged.append(content)
            merged.append("")  # Add blank line between files

        return '\n'.join(merged)


# Example usage
if __name__ == "__main__":
    # Example code
    sample_code = """
import os
import sys

# This is a comment
def calculate_sum(a, b):
    '''This function calculates sum'''
    # Add the numbers
    return a + b


def main():
    result = calculate_sum(5, 3)
    print(result)


if __name__ == "__main__":
    main()
    """

    preprocessor = CodePreprocessor()

    print("Original code:")
    print(sample_code)
    print("\n" + "="*50 + "\n")

    print("Standardized code:")
    standardized = preprocessor.standardize_code(sample_code, language='python')
    print(standardized)
    print("\n" + "="*50 + "\n")

    print("Preprocessed for CodeBERT:")
    codebert_ready = preprocessor.preprocess_for_codebert(sample_code)
    print(codebert_ready)
    print("\n" + "="*50 + "\n")

    print("Preprocessed for Token Similarity:")
    token_ready = preprocessor.preprocess_for_token_similarity(sample_code)
    print(token_ready)
