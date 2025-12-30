"""
Token Similarity Evaluator Module
Implements token-based similarity calculation using Jaccard similarity and token overlap.
"""

import re
from typing import List, Dict, Set, Tuple
from collections import Counter
import string


class TokenSimilarityEvaluator:
    """Evaluates code similarity based on token overlap."""

    def __init__(self):
        """Initialize Token Similarity Evaluator."""
        self.stop_tokens = set(['if', 'else', 'for', 'while', 'return', 'def', 'class',
                                'import', 'from', 'and', 'or', 'not', 'in', 'is'])

    def tokenize_code(self, code: str, language: str = 'python') -> List[str]:
        """
        Tokenize code into meaningful tokens.

        Args:
            code: Source code string
            language: Programming language

        Returns:
            List of tokens
        """
        # Remove punctuation and split by whitespace and special characters
        # Keep underscores as they're part of identifiers
        tokens = re.findall(r'\b\w+\b', code.lower())

        # Filter out single characters and numbers only tokens
        tokens = [t for t in tokens if len(t) > 1 and not t.isdigit()]

        return tokens

    def extract_identifiers(self, code: str, language: str = 'python') -> Set[str]:
        """
        Extract identifiers (function names, variable names) from code.

        Args:
            code: Source code string
            language: Programming language

        Returns:
            Set of identifier names
        """
        identifiers = set()

        if language == 'python':
            # Extract function definitions
            func_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)'
            identifiers.update(re.findall(func_pattern, code))

            # Extract class definitions
            class_pattern = r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)'
            identifiers.update(re.findall(class_pattern, code))

            # Extract variable assignments (simple case)
            var_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*='
            identifiers.update(re.findall(var_pattern, code))

        elif language in ['java', 'javascript', 'typescript', 'cpp', 'c']:
            # Extract identifiers (simplified approach)
            identifier_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
            identifiers.update(re.findall(identifier_pattern, code))

        # Remove common keywords
        identifiers = identifiers - self.stop_tokens

        return identifiers

    def extract_keywords(self, code: str, language: str = 'python') -> List[str]:
        """
        Extract programming keywords from code.

        Args:
            code: Source code string
            language: Programming language

        Returns:
            List of keywords found in code
        """
        if language == 'python':
            keywords = [
                'def', 'class', 'if', 'else', 'elif', 'for', 'while', 'return',
                'import', 'from', 'as', 'try', 'except', 'finally', 'with',
                'lambda', 'yield', 'pass', 'break', 'continue', 'and', 'or', 'not',
                'in', 'is', 'True', 'False', 'None'
            ]
        elif language == 'java':
            keywords = [
                'public', 'private', 'protected', 'static', 'void', 'class',
                'interface', 'extends', 'implements', 'if', 'else', 'for',
                'while', 'return', 'new', 'this', 'super', 'try', 'catch',
                'finally', 'throw', 'throws', 'boolean', 'int', 'String'
            ]
        else:
            keywords = []

        found_keywords = []
        for keyword in keywords:
            if re.search(r'\b' + keyword + r'\b', code):
                found_keywords.append(keyword)

        return found_keywords

    def calculate_jaccard_similarity(self, set1: Set, set2: Set) -> float:
        """
        Calculate Jaccard similarity between two sets.

        Args:
            set1: First set of items
            set2: Second set of items

        Returns:
            Jaccard similarity score (0-1)
        """
        if len(set1) == 0 and len(set2) == 0:
            return 1.0

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        if union == 0:
            return 0.0

        return intersection / union

    def calculate_token_overlap(
        self,
        tokens1: List[str],
        tokens2: List[str]
    ) -> Dict[str, any]:
        """
        Calculate detailed token overlap metrics.

        Args:
            tokens1: First list of tokens
            tokens2: Second list of tokens

        Returns:
            Dictionary with overlap metrics
        """
        set1 = set(tokens1)
        set2 = set(tokens2)

        intersection = set1.intersection(set2)
        union = set1.union(set2)
        only_in_1 = set1 - set2
        only_in_2 = set2 - set1

        # Calculate various metrics
        jaccard = self.calculate_jaccard_similarity(set1, set2)

        # Overlap coefficient (Szymkiewicz–Simpson coefficient)
        overlap_coef = len(intersection) / min(len(set1), len(set2)) if min(len(set1), len(set2)) > 0 else 0

        # Dice coefficient
        dice = (2 * len(intersection)) / (len(set1) + len(set2)) if (len(set1) + len(set2)) > 0 else 0

        return {
            'jaccard_similarity': jaccard,
            'overlap_coefficient': overlap_coef,
            'dice_coefficient': dice,
            'intersection_size': len(intersection),
            'union_size': len(union),
            'tokens_in_common': sorted(list(intersection)),
            'tokens_only_in_first': sorted(list(only_in_1)),
            'tokens_only_in_second': sorted(list(only_in_2)),
            'total_tokens_first': len(set1),
            'total_tokens_second': len(set2)
        }

    def calculate_similarity(
        self,
        code1: str,
        code2: str,
        language: str = 'python',
        use_identifiers: bool = True
    ) -> float:
        """
        Calculate token-based similarity between two code snippets.

        Args:
            code1: First code snippet
            code2: Second code snippet
            language: Programming language
            use_identifiers: Whether to focus on identifiers (True) or all tokens (False)

        Returns:
            Similarity score between 0 and 1
        """
        if use_identifiers:
            # Extract and compare identifiers
            identifiers1 = self.extract_identifiers(code1, language)
            identifiers2 = self.extract_identifiers(code2, language)
            similarity = self.calculate_jaccard_similarity(identifiers1, identifiers2)
        else:
            # Compare all tokens
            tokens1 = self.tokenize_code(code1, language)
            tokens2 = self.tokenize_code(code2, language)
            set1 = set(tokens1)
            set2 = set(tokens2)
            similarity = self.calculate_jaccard_similarity(set1, set2)

        return similarity

    def detailed_similarity_analysis(
        self,
        student_code: str,
        reference_code: str,
        language: str = 'python'
    ) -> Dict[str, any]:
        """
        Perform detailed similarity analysis between student and reference code.

        Args:
            student_code: Student's submitted code
            reference_code: Reference/expected code
            language: Programming language

        Returns:
            Dictionary with comprehensive similarity metrics
        """
        # Tokenize both codes
        student_tokens = self.tokenize_code(student_code, language)
        reference_tokens = self.tokenize_code(reference_code, language)

        # Extract identifiers
        student_identifiers = self.extract_identifiers(student_code, language)
        reference_identifiers = self.extract_identifiers(reference_code, language)

        # Extract keywords
        student_keywords = self.extract_keywords(student_code, language)
        reference_keywords = self.extract_keywords(reference_code, language)

        # Calculate token overlap
        token_overlap = self.calculate_token_overlap(student_tokens, reference_tokens)

        # Calculate identifier similarity
        identifier_similarity = self.calculate_jaccard_similarity(
            student_identifiers,
            reference_identifiers
        )

        # Calculate keyword similarity
        keyword_similarity = self.calculate_jaccard_similarity(
            set(student_keywords),
            set(reference_keywords)
        )

        # Combined similarity (weighted average)
        # Weight: identifiers (50%), tokens (30%), keywords (20%)
        combined_similarity = (
            identifier_similarity * 0.5 +
            token_overlap['jaccard_similarity'] * 0.3 +
            keyword_similarity * 0.2
        )

        return {
            'combined_similarity': combined_similarity,
            'combined_similarity_percent': combined_similarity * 100,
            'identifier_similarity': identifier_similarity,
            'identifier_similarity_percent': identifier_similarity * 100,
            'token_jaccard_similarity': token_overlap['jaccard_similarity'],
            'token_jaccard_similarity_percent': token_overlap['jaccard_similarity'] * 100,
            'keyword_similarity': keyword_similarity,
            'keyword_similarity_percent': keyword_similarity * 100,
            'token_overlap': token_overlap,
            'student_identifiers': sorted(list(student_identifiers)),
            'reference_identifiers': sorted(list(reference_identifiers)),
            'common_identifiers': sorted(list(student_identifiers.intersection(reference_identifiers))),
            'missing_identifiers': sorted(list(reference_identifiers - student_identifiers)),
            'extra_identifiers': sorted(list(student_identifiers - reference_identifiers))
        }

    def batch_evaluate(
        self,
        student_codes: List[str],
        reference_code: str,
        language: str = 'python'
    ) -> List[Dict[str, any]]:
        """
        Evaluate multiple student codes against a reference.

        Args:
            student_codes: List of student code snippets
            reference_code: Reference code
            language: Programming language

        Returns:
            List of evaluation results
        """
        results = []

        print(f"Evaluating {len(student_codes)} student submissions using token similarity...")

        for i, student_code in enumerate(student_codes, 1):
            try:
                analysis = self.detailed_similarity_analysis(
                    student_code,
                    reference_code,
                    language
                )
                analysis['index'] = i - 1
                analysis['status'] = 'success'
                results.append(analysis)

                if i % 10 == 0:
                    print(f"Processed {i}/{len(student_codes)} submissions")

            except Exception as e:
                results.append({
                    'index': i - 1,
                    'error': str(e),
                    'status': 'failed'
                })

        print("Batch evaluation complete!")
        return results


# Example usage
if __name__ == "__main__":
    # Sample codes for testing
    code1 = """
    def calculate_sum(a, b):
        result = a + b
        return result

    def main():
        answer = calculate_sum(5, 3)
        print(answer)
    """

    code2 = """
    def add_numbers(x, y):
        total = x + y
        return total

    def run():
        result = add_numbers(5, 3)
        print(result)
    """

    code3 = """
    def multiply(a, b):
        product = a * b
        return product
    """

    print("Initializing Token Similarity Evaluator...")
    evaluator = TokenSimilarityEvaluator()

    print("\n" + "="*60)
    print("Analyzing similarity between similar codes (code1 vs code2):")
    print("="*60)
    analysis1 = evaluator.detailed_similarity_analysis(code1, code2)
    print(f"Combined Similarity: {analysis1['combined_similarity_percent']:.2f}%")
    print(f"Identifier Similarity: {analysis1['identifier_similarity_percent']:.2f}%")
    print(f"Token Similarity: {analysis1['token_jaccard_similarity_percent']:.2f}%")
    print(f"Common Identifiers: {analysis1['common_identifiers']}")
    print(f"Missing Identifiers: {analysis1['missing_identifiers']}")

    print("\n" + "="*60)
    print("Analyzing similarity between different codes (code1 vs code3):")
    print("="*60)
    analysis2 = evaluator.detailed_similarity_analysis(code1, code3)
    print(f"Combined Similarity: {analysis2['combined_similarity_percent']:.2f}%")
    print(f"Identifier Similarity: {analysis2['identifier_similarity_percent']:.2f}%")
    print(f"Token Similarity: {analysis2['token_jaccard_similarity_percent']:.2f}%")
    print(f"Common Identifiers: {analysis2['common_identifiers']}")
    print(f"Missing Identifiers: {analysis2['missing_identifiers']}")
