"""
Score Combiner Module
Combines scores from CodeBERT and Token Similarity methods.
"""

from typing import Dict, List, Optional, Tuple
import statistics


class ScoreCombiner:
    """Combines evaluation scores from multiple methods."""

    def __init__(
        self,
        codebert_weight: float = 0.6,
        token_weight: float = 0.4,
        pass_threshold: float = 0.7
    ):
        """
        Initialize Score Combiner.

        Args:
            codebert_weight: Weight for CodeBERT score (default: 0.6)
            token_weight: Weight for Token similarity score (default: 0.4)
            pass_threshold: Threshold for pass/fail decision (default: 0.7, i.e., 70%)
        """
        # Validate weights sum to 1.0
        if abs(codebert_weight + token_weight - 1.0) > 0.001:
            raise ValueError("Weights must sum to 1.0")

        self.codebert_weight = codebert_weight
        self.token_weight = token_weight
        self.pass_threshold = pass_threshold

    def combine_scores(
        self,
        codebert_score: float,
        token_score: float,
        method: str = 'weighted'
    ) -> Dict[str, float]:
        """
        Combine scores from both evaluation methods.

        Args:
            codebert_score: Similarity score from CodeBERT (0-1)
            token_score: Similarity score from Token method (0-1)
            method: Combination method ('weighted', 'average', 'max', 'min', 'harmonic')

        Returns:
            Dictionary with combined scores and metadata
        """
        if method == 'weighted':
            combined = (
                codebert_score * self.codebert_weight +
                token_score * self.token_weight
            )
        elif method == 'average':
            combined = (codebert_score + token_score) / 2
        elif method == 'max':
            combined = max(codebert_score, token_score)
        elif method == 'min':
            combined = min(codebert_score, token_score)
        elif method == 'harmonic':
            # Harmonic mean (penalizes if one score is very low)
            if codebert_score == 0 or token_score == 0:
                combined = 0
            else:
                combined = 2 / ((1 / codebert_score) + (1 / token_score))
        else:
            raise ValueError(f"Unknown combination method: {method}")

        return {
            'combined_score': combined,
            'combined_score_percent': combined * 100,
            'codebert_score': codebert_score,
            'codebert_score_percent': codebert_score * 100,
            'token_score': token_score,
            'token_score_percent': token_score * 100,
            'method': method,
            'codebert_weight': self.codebert_weight,
            'token_weight': self.token_weight
        }

    def evaluate_pass_fail(
        self,
        combined_score: float,
        custom_threshold: Optional[float] = None
    ) -> Dict[str, any]:
        """
        Determine pass/fail based on combined score.

        Args:
            combined_score: Combined similarity score (0-1)
            custom_threshold: Optional custom threshold (overrides default)

        Returns:
            Dictionary with pass/fail result and details
        """
        threshold = custom_threshold if custom_threshold is not None else self.pass_threshold

        passed = combined_score >= threshold
        margin = combined_score - threshold

        # Determine grade category
        if combined_score >= 0.9:
            grade = 'A'
            category = 'Excellent'
        elif combined_score >= 0.8:
            grade = 'B'
            category = 'Good'
        elif combined_score >= 0.7:
            grade = 'C'
            category = 'Satisfactory'
        elif combined_score >= 0.6:
            grade = 'D'
            category = 'Needs Improvement'
        else:
            grade = 'F'
            category = 'Poor'

        return {
            'passed': passed,
            'grade': grade,
            'category': category,
            'score': combined_score,
            'score_percent': combined_score * 100,
            'threshold': threshold,
            'threshold_percent': threshold * 100,
            'margin': margin,
            'margin_percent': margin * 100
        }

    def generate_recommendations(
        self,
        codebert_score: float,
        token_score: float,
        combined_score: float,
        token_details: Optional[Dict] = None
    ) -> List[str]:
        """
        Generate recommendations based on evaluation results.

        Args:
            codebert_score: CodeBERT similarity score
            token_score: Token similarity score
            combined_score: Combined score
            token_details: Optional detailed token analysis

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Overall score recommendations
        if combined_score < 0.5:
            recommendations.append("The code differs significantly from the reference. Review core logic and implementation approach.")
        elif combined_score < 0.7:
            recommendations.append("The code shows moderate similarity. Consider reviewing key functions and algorithms.")

        # CodeBERT vs Token discrepancy
        score_diff = abs(codebert_score - token_score)
        if score_diff > 0.3:
            if codebert_score > token_score:
                recommendations.append("High semantic similarity but low token overlap. Code may use different naming conventions or structure but achieves similar functionality.")
            else:
                recommendations.append("High token overlap but low semantic similarity. Code may share similar naming but different logic or functionality.")

        # Specific recommendations based on token details
        if token_details:
            if 'missing_identifiers' in token_details and len(token_details['missing_identifiers']) > 0:
                missing = ', '.join(token_details['missing_identifiers'][:5])
                if len(token_details['missing_identifiers']) > 5:
                    missing += f", and {len(token_details['missing_identifiers']) - 5} more"
                recommendations.append(f"Missing key identifiers from reference: {missing}")

            if 'extra_identifiers' in token_details and len(token_details['extra_identifiers']) > 3:
                recommendations.append("Code contains many additional identifiers not in reference. Verify if extra functionality is required.")

        # Score-specific recommendations
        if codebert_score < 0.5:
            recommendations.append("Low semantic similarity detected. The code logic may differ significantly from expected implementation.")

        if token_score < 0.5:
            recommendations.append("Low token similarity. Consider using similar naming conventions and structure as the reference.")

        if not recommendations:
            recommendations.append("Code shows good similarity to reference implementation. Good work!")

        return recommendations

    def comprehensive_evaluation(
        self,
        codebert_results: Dict[str, float],
        token_results: Dict[str, any],
        method: str = 'weighted'
    ) -> Dict[str, any]:
        """
        Perform comprehensive evaluation combining all metrics.

        Args:
            codebert_results: Results from CodeBERT evaluator
            token_results: Results from Token similarity evaluator
            method: Score combination method

        Returns:
            Comprehensive evaluation results
        """
        # Extract scores
        codebert_score = codebert_results.get('code_similarity', 0)
        token_score = token_results.get('combined_similarity', 0)

        # Combine scores
        combined = self.combine_scores(codebert_score, token_score, method)

        # Evaluate pass/fail
        evaluation = self.evaluate_pass_fail(combined['combined_score'])

        # Generate recommendations
        recommendations = self.generate_recommendations(
            codebert_score,
            token_score,
            combined['combined_score'],
            token_results
        )

        # Compile comprehensive results
        return {
            'final_score': combined['combined_score'],
            'final_score_percent': combined['combined_score_percent'],
            'passed': evaluation['passed'],
            'grade': evaluation['grade'],
            'category': evaluation['category'],
            'codebert_score': codebert_score,
            'codebert_score_percent': codebert_score * 100,
            'token_score': token_score,
            'token_score_percent': token_score * 100,
            'combination_method': method,
            'weights': {
                'codebert': self.codebert_weight,
                'token': self.token_weight
            },
            'threshold': self.pass_threshold,
            'threshold_percent': self.pass_threshold * 100,
            'recommendations': recommendations,
            'detailed_token_analysis': token_results,
            'detailed_codebert_analysis': codebert_results
        }

    def batch_combine_scores(
        self,
        codebert_results_list: List[Dict],
        token_results_list: List[Dict],
        method: str = 'weighted'
    ) -> List[Dict[str, any]]:
        """
        Combine scores for multiple evaluations.

        Args:
            codebert_results_list: List of CodeBERT results
            token_results_list: List of Token similarity results
            method: Score combination method

        Returns:
            List of comprehensive evaluation results
        """
        if len(codebert_results_list) != len(token_results_list):
            raise ValueError("CodeBERT and Token results lists must have same length")

        combined_results = []

        for i, (codebert_res, token_res) in enumerate(zip(codebert_results_list, token_results_list)):
            try:
                result = self.comprehensive_evaluation(codebert_res, token_res, method)
                result['index'] = i
                combined_results.append(result)
            except Exception as e:
                combined_results.append({
                    'index': i,
                    'error': str(e),
                    'status': 'failed'
                })

        return combined_results


# Example usage
if __name__ == "__main__":
    # Sample evaluation results
    codebert_result = {
        'code_similarity': 0.85,
        'code_similarity_percent': 85.0
    }

    token_result = {
        'combined_similarity': 0.72,
        'combined_similarity_percent': 72.0,
        'identifier_similarity': 0.75,
        'missing_identifiers': ['calculate', 'process'],
        'extra_identifiers': ['compute', 'handle']
    }

    print("Initializing Score Combiner...")
    combiner = ScoreCombiner(codebert_weight=0.6, token_weight=0.4, pass_threshold=0.7)

    print("\nCombining scores with weighted method:")
    result = combiner.comprehensive_evaluation(codebert_result, token_result, method='weighted')

    print(f"Final Score: {result['final_score_percent']:.2f}%")
    print(f"Grade: {result['grade']} ({result['category']})")
    print(f"Passed: {result['passed']}")
    print(f"\nCodeBERT Score: {result['codebert_score_percent']:.2f}%")
    print(f"Token Score: {result['token_score_percent']:.2f}%")
    print(f"\nRecommendations:")
    for i, rec in enumerate(result['recommendations'], 1):
        print(f"{i}. {rec}")

    print("\n" + "="*60)
    print("\nComparing different combination methods:")
    methods = ['weighted', 'average', 'max', 'min', 'harmonic']
    for method in methods:
        scores = combiner.combine_scores(0.85, 0.72, method=method)
        print(f"{method.capitalize():12s}: {scores['combined_score_percent']:6.2f}%")
