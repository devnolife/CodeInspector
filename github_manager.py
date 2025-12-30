"""
GitHub Manager Module
Handles GitHub repository cloning, URL parsing, and file extraction.
"""

import os
import re
import shutil
from typing import List, Dict, Optional
from pathlib import Path
import tempfile

try:
    from git import Repo
    from github import Github
except ImportError:
    print("Warning: GitPython or PyGithub not installed. GitHub features may not work.")


class GitHubManager:
    """Manages GitHub repository operations."""

    def __init__(self, github_token: Optional[str] = None):
        """
        Initialize GitHub Manager.

        Args:
            github_token: Optional GitHub personal access token for API access
        """
        self.github_token = github_token
        self.gh_client = Github(github_token) if github_token else None
        self.temp_dir = None

    def parse_github_url(self, url: str) -> Dict[str, str]:
        """
        Parse GitHub URL to extract owner and repository name.

        Args:
            url: GitHub repository URL

        Returns:
            Dictionary with 'owner' and 'repo' keys

        Example:
            >>> manager = GitHubManager()
            >>> info = manager.parse_github_url("https://github.com/microsoft/codebert")
            >>> print(info)
            {'owner': 'microsoft', 'repo': 'codebert'}
        """
        # Handle different GitHub URL formats
        patterns = [
            r'github\.com[:/]([^/]+)/([^/\.]+)',  # https://github.com/owner/repo or git@github.com:owner/repo
            r'github\.com/([^/]+)/([^/]+)\.git',   # https://github.com/owner/repo.git
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return {
                    'owner': match.group(1),
                    'repo': match.group(2).replace('.git', '')
                }

        raise ValueError(f"Invalid GitHub URL format: {url}")

    def clone_repository(self, url: str, target_dir: Optional[str] = None) -> str:
        """
        Clone a GitHub repository to local directory.

        Args:
            url: GitHub repository URL
            target_dir: Target directory for cloning (if None, uses temp directory)

        Returns:
            Path to cloned repository
        """
        if target_dir is None:
            # Create temporary directory
            self.temp_dir = tempfile.mkdtemp(prefix="codeinspector_")
            target_dir = self.temp_dir

        repo_info = self.parse_github_url(url)
        clone_path = os.path.join(target_dir, repo_info['repo'])

        # Remove directory if it already exists
        if os.path.exists(clone_path):
            shutil.rmtree(clone_path)

        print(f"Cloning repository from {url}...")
        try:
            Repo.clone_from(url, clone_path)
            print(f"Repository cloned to: {clone_path}")
            return clone_path
        except Exception as e:
            raise Exception(f"Failed to clone repository: {str(e)}")

    def extract_code_files(
        self,
        repo_path: str,
        extensions: List[str] = None,
        exclude_dirs: List[str] = None
    ) -> List[Dict[str, str]]:
        """
        Extract code files from repository.

        Args:
            repo_path: Path to cloned repository
            extensions: List of file extensions to include (e.g., ['.py', '.java', '.js'])
            exclude_dirs: List of directory names to exclude (e.g., ['node_modules', '.git'])

        Returns:
            List of dictionaries with 'path' and 'content' keys
        """
        if extensions is None:
            extensions = ['.py', '.java', '.js', '.cpp', '.c', '.ts', '.jsx', '.tsx', '.go']

        if exclude_dirs is None:
            exclude_dirs = ['.git', 'node_modules', '__pycache__', 'venv', 'env', '.venv', 'dist', 'build']

        code_files = []

        for root, dirs, files in os.walk(repo_path):
            # Remove excluded directories from traversal
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                file_path = os.path.join(root, file)
                _, ext = os.path.splitext(file)

                if ext in extensions:
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()

                        # Get relative path from repo root
                        rel_path = os.path.relpath(file_path, repo_path)

                        code_files.append({
                            'path': rel_path,
                            'full_path': file_path,
                            'content': content,
                            'extension': ext
                        })
                    except Exception as e:
                        print(f"Warning: Could not read file {file_path}: {str(e)}")

        print(f"Extracted {len(code_files)} code files from repository")
        return code_files

    def get_repository_info(self, url: str) -> Dict[str, any]:
        """
        Get repository information using GitHub API.

        Args:
            url: GitHub repository URL

        Returns:
            Dictionary with repository metadata
        """
        if not self.gh_client:
            return {"error": "GitHub token not provided. Cannot fetch repository info."}

        try:
            repo_info = self.parse_github_url(url)
            repo = self.gh_client.get_repo(f"{repo_info['owner']}/{repo_info['repo']}")

            return {
                'name': repo.name,
                'full_name': repo.full_name,
                'description': repo.description,
                'stars': repo.stargazers_count,
                'language': repo.language,
                'created_at': repo.created_at,
                'updated_at': repo.updated_at,
                'url': repo.html_url
            }
        except Exception as e:
            return {"error": f"Failed to fetch repository info: {str(e)}"}

    def cleanup(self):
        """Clean up temporary directories."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"Cleaned up temporary directory: {self.temp_dir}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()


# Example usage
if __name__ == "__main__":
    # Example: Clone a repository and extract code files
    manager = GitHubManager()

    # Parse URL
    url = "https://github.com/microsoft/codebert"
    info = manager.parse_github_url(url)
    print(f"Repository info: {info}")

    # Clone repository (commented out to avoid actual cloning during testing)
    # repo_path = manager.clone_repository(url)
    # code_files = manager.extract_code_files(repo_path, extensions=['.py'])
    # print(f"Found {len(code_files)} Python files")
    # manager.cleanup()
