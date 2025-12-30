"""
CodeBERT Evaluator Module
Handles CodeBERT model loading, embedding generation, and similarity calculation.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    from sklearn.metrics.pairwise import cosine_similarity
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    warnings.warn("Required dependencies not available. Install transformers, torch, and scikit-learn.")


class CodeBERTEvaluator:
    """Evaluates code similarity using CodeBERT embeddings."""

    def __init__(self, model_name: str = "microsoft/codebert-base"):
        """
        Initialize CodeBERT Evaluator.

        Args:
            model_name: HuggingFace model name (default: microsoft/codebert-base)
        """
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError("Required dependencies not installed. Run: pip install transformers torch scikit-learn")

        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Initializing CodeBERT Evaluator with model: {model_name}")
        print(f"Using device: {self.device}")

    def load_model(self):
        """Load CodeBERT model and tokenizer."""
        if self.model is not None:
            print("Model already loaded.")
            return

        print(f"Loading tokenizer and model from {self.model_name}...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            print("Model loaded successfully!")
        except Exception as e:
            raise Exception(f"Failed to load CodeBERT model: {str(e)}")

    def generate_embedding(self, code: str, max_length: int = 512) -> np.ndarray:
        """
        Generate embedding for a code snippet using CodeBERT.

        Args:
            code: Source code string
            max_length: Maximum token length (default: 512)

        Returns:
            Numpy array containing the code embedding
        """
        if self.model is None:
            self.load_model()

        # Tokenize the code
        inputs = self.tokenizer(
            code,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding="max_length"
        )

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Use [CLS] token embedding as code representation
        # Shape: (batch_size, hidden_size)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return embedding[0]  # Return first (and only) embedding

    def calculate_similarity(
        self,
        code1: str,
        code2: str,
        max_length: int = 512
    ) -> float:
        """
        Calculate cosine similarity between two code snippets.

        Args:
            code1: First code snippet
            code2: Second code snippet
            max_length: Maximum token length

        Returns:
            Similarity score between 0 and 1
        """
        # Generate embeddings for both codes
        embedding1 = self.generate_embedding(code1, max_length)
        embedding2 = self.generate_embedding(code2, max_length)

        # Calculate cosine similarity
        similarity = cosine_similarity(
            embedding1.reshape(1, -1),
            embedding2.reshape(1, -1)
        )[0][0]

        return float(similarity)

    def calculate_similarity_to_reference(
        self,
        student_code: str,
        reference_code: str,
        requirements: Optional[str] = None,
        max_length: int = 512
    ) -> Dict[str, float]:
        """
        Calculate similarity between student code and reference code.
        Optionally also compare against requirements.

        Args:
            student_code: Student's submitted code
            reference_code: Reference/expected code
            requirements: Optional requirements text
            max_length: Maximum token length

        Returns:
            Dictionary with similarity scores
        """
        results = {}

        # Calculate student vs reference similarity
        code_similarity = self.calculate_similarity(
            student_code,
            reference_code,
            max_length
        )
        results['code_similarity'] = code_similarity
        results['code_similarity_percent'] = code_similarity * 100

        # If requirements provided, calculate semantic alignment
        if requirements:
            # Generate embeddings
            student_emb = self.generate_embedding(student_code, max_length)
            reference_emb = self.generate_embedding(reference_code, max_length)
            req_emb = self.generate_embedding(requirements, max_length)

            # Calculate similarities to requirements
            student_req_sim = cosine_similarity(
                student_emb.reshape(1, -1),
                req_emb.reshape(1, -1)
            )[0][0]

            reference_req_sim = cosine_similarity(
                reference_emb.reshape(1, -1),
                req_emb.reshape(1, -1)
            )[0][0]

            results['student_requirements_similarity'] = float(student_req_sim)
            results['reference_requirements_similarity'] = float(reference_req_sim)
            results['student_requirements_similarity_percent'] = float(student_req_sim * 100)
            results['reference_requirements_similarity_percent'] = float(reference_req_sim * 100)

        return results

    def batch_evaluate(
        self,
        student_codes: List[str],
        reference_code: str,
        max_length: int = 512
    ) -> List[Dict[str, float]]:
        """
        Evaluate multiple student codes against a reference.

        Args:
            student_codes: List of student code snippets
            reference_code: Reference code
            max_length: Maximum token length

        Returns:
            List of evaluation results
        """
        results = []

        # Generate reference embedding once
        reference_emb = self.generate_embedding(reference_code, max_length)

        print(f"Evaluating {len(student_codes)} student submissions...")

        for i, student_code in enumerate(student_codes, 1):
            try:
                # Generate student embedding
                student_emb = self.generate_embedding(student_code, max_length)

                # Calculate similarity
                similarity = cosine_similarity(
                    student_emb.reshape(1, -1),
                    reference_emb.reshape(1, -1)
                )[0][0]

                results.append({
                    'index': i - 1,
                    'code_similarity': float(similarity),
                    'code_similarity_percent': float(similarity * 100),
                    'status': 'success'
                })

                if i % 10 == 0:
                    print(f"Processed {i}/{len(student_codes)} submissions")

            except Exception as e:
                results.append({
                    'index': i - 1,
                    'error': str(e),
                    'status': 'failed'
                })

        print(f"Batch evaluation complete!")
        return results

    def get_confidence_metrics(self, similarity_score: float) -> Dict[str, any]:
        """
        Generate confidence metrics based on similarity score.

        Args:
            similarity_score: Similarity score (0-1)

        Returns:
            Dictionary with confidence metrics
        """
        # Define thresholds
        if similarity_score >= 0.9:
            confidence = "Very High"
            evaluation = "Excellent match"
        elif similarity_score >= 0.75:
            confidence = "High"
            evaluation = "Good match"
        elif similarity_score >= 0.6:
            confidence = "Medium"
            evaluation = "Moderate match"
        elif similarity_score >= 0.4:
            confidence = "Low"
            evaluation = "Poor match"
        else:
            confidence = "Very Low"
            evaluation = "Very poor match"

        return {
            'confidence_level': confidence,
            'evaluation': evaluation,
            'similarity_score': similarity_score,
            'similarity_percent': similarity_score * 100
        }


# Example usage
if __name__ == "__main__":
    # Sample codes for testing
    code1 = """
    def calculate_sum(a, b):
        return a + b

    def main():
        result = calculate_sum(5, 3)
        print(result)
    """

    code2 = """
    def add_numbers(x, y):
        total = x + y
        return total

    def run():
        answer = add_numbers(5, 3)
        print(answer)
    """

    code3 = """
    def multiply(a, b):
        return a * b
    """

    print("Initializing CodeBERT Evaluator...")
    evaluator = CodeBERTEvaluator()

    print("\nCalculating similarity between similar codes...")
    sim1 = evaluator.calculate_similarity(code1, code2)
    print(f"Similarity (code1 vs code2): {sim1:.4f} ({sim1*100:.2f}%)")

    print("\nCalculating similarity between different codes...")
    sim2 = evaluator.calculate_similarity(code1, code3)
    print(f"Similarity (code1 vs code3): {sim2:.4f} ({sim2*100:.2f}%)")

    print("\nGenerating confidence metrics...")
    metrics = evaluator.get_confidence_metrics(sim1)
    print(f"Confidence: {metrics['confidence_level']}")
    print(f"Evaluation: {metrics['evaluation']}")
