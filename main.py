"""
Main Orchestrator Module
Coordinates all components of the Code Inspector system.
"""

import os
import sys
from typing import Dict, List, Optional
import argparse

from github_manager import GitHubManager
from code_preprocessor import CodePreprocessor
from codebert_evaluator import CodeBERTEvaluator
from token_similarity_evaluator import TokenSimilarityEvaluator
from score_combiner import ScoreCombiner
from report_generator import ReportGenerator


class CodeInspector:
    """Main orchestrator for code evaluation system."""

    def __init__(
        self,
        github_token: Optional[str] = None,
        codebert_weight: float = 0.6,
        token_weight: float = 0.4,
        pass_threshold: float = 0.7
    ):
        """
        Initialize Code Inspector.

        Args:
            github_token: Optional GitHub personal access token
            codebert_weight: Weight for CodeBERT score (default: 0.6)
            token_weight: Weight for Token similarity score (default: 0.4)
            pass_threshold: Threshold for pass/fail decision (default: 0.7)
        """
        print("Initializing Code Inspector...")

        # Initialize all components
        self.github_manager = GitHubManager(github_token)
        self.preprocessor = CodePreprocessor()
        self.codebert_evaluator = CodeBERTEvaluator()
        self.token_evaluator = TokenSimilarityEvaluator()
        self.score_combiner = ScoreCombiner(codebert_weight, token_weight, pass_threshold)
        self.report_generator = ReportGenerator()

        print("Code Inspector initialized successfully!")

    def evaluate_code(
        self,
        student_code: str,
        reference_code: str,
        requirements: Optional[str] = None,
        language: str = 'python',
        combination_method: str = 'weighted'
    ) -> Dict[str, any]:
        """
        Evaluate student code against reference code.

        Args:
            student_code: Student's code
            reference_code: Reference/expected code
            requirements: Optional project requirements
            language: Programming language (default: 'python')
            combination_method: Score combination method (default: 'weighted')

        Returns:
            Comprehensive evaluation results
        """
        print("\n" + "="*60)
        print("Starting Code Evaluation")
        print("="*60)

        # Step 1: Preprocess codes for CodeBERT
        print("\n[1/4] Preprocessing code for CodeBERT...")
        student_code_codebert = self.preprocessor.preprocess_for_codebert(student_code, language)
        reference_code_codebert = self.preprocessor.preprocess_for_codebert(reference_code, language)

        # Step 2: Preprocess codes for Token Similarity
        print("[2/4] Preprocessing code for Token Similarity...")
        student_code_token = self.preprocessor.preprocess_for_token_similarity(student_code, language)
        reference_code_token = self.preprocessor.preprocess_for_token_similarity(reference_code, language)

        # Step 3: Run CodeBERT evaluation
        print("[3/4] Running CodeBERT evaluation...")
        codebert_results = self.codebert_evaluator.calculate_similarity_to_reference(
            student_code_codebert,
            reference_code_codebert,
            requirements
        )
        print(f"    CodeBERT similarity: {codebert_results['code_similarity_percent']:.2f}%")

        # Step 4: Run Token Similarity evaluation
        print("[4/4] Running Token Similarity evaluation...")
        token_results = self.token_evaluator.detailed_similarity_analysis(
            student_code_token,
            reference_code_token,
            language
        )
        print(f"    Token similarity: {token_results['combined_similarity_percent']:.2f}%")

        # Combine scores
        print("\nCombining scores...")
        final_results = self.score_combiner.comprehensive_evaluation(
            codebert_results,
            token_results,
            combination_method
        )

        print(f"\nFinal Score: {final_results['final_score_percent']:.2f}%")
        print(f"Grade: {final_results['grade']} ({final_results['category']})")
        print(f"Status: {'PASSED' if final_results['passed'] else 'FAILED'}")
        print("="*60)

        return final_results

    def evaluate_github_project(
        self,
        student_url: str,
        reference_code: str,
        requirements: Optional[str] = None,
        language: str = 'python',
        file_extensions: Optional[List[str]] = None,
        combination_method: str = 'weighted'
    ) -> Dict[str, any]:
        """
        Evaluate a GitHub project against reference code.

        Args:
            student_url: GitHub repository URL
            reference_code: Reference/expected code
            requirements: Optional project requirements
            language: Programming language
            file_extensions: File extensions to include (e.g., ['.py', '.java'])
            combination_method: Score combination method

        Returns:
            Comprehensive evaluation results
        """
        print("\n" + "="*60)
        print("Evaluating GitHub Project")
        print("="*60)
        print(f"Repository URL: {student_url}")

        try:
            # Clone repository
            print("\nCloning repository...")
            repo_path = self.github_manager.clone_repository(student_url)

            # Extract code files
            print("Extracting code files...")
            code_files = self.github_manager.extract_code_files(
                repo_path,
                extensions=file_extensions
            )

            if not code_files:
                raise Exception("No code files found in repository")

            # Merge code files into single string
            print("Merging code files...")
            student_code = self.preprocessor.merge_code_files(code_files)

            # Evaluate
            results = self.evaluate_code(
                student_code,
                reference_code,
                requirements,
                language,
                combination_method
            )

            # Add repository info
            results['repository_url'] = student_url
            results['files_analyzed'] = len(code_files)

            return results

        except Exception as e:
            print(f"\nError evaluating GitHub project: {str(e)}")
            return {
                'error': str(e),
                'status': 'failed'
            }
        finally:
            # Cleanup
            self.github_manager.cleanup()

    def batch_evaluate(
        self,
        student_codes: List[str],
        reference_code: str,
        requirements: Optional[str] = None,
        language: str = 'python',
        combination_method: str = 'weighted'
    ) -> List[Dict[str, any]]:
        """
        Evaluate multiple student codes.

        Args:
            student_codes: List of student code strings
            reference_code: Reference code
            requirements: Optional requirements
            language: Programming language
            combination_method: Score combination method

        Returns:
            List of evaluation results
        """
        print("\n" + "="*60)
        print(f"Batch Evaluation: {len(student_codes)} submissions")
        print("="*60)

        results = []

        for i, student_code in enumerate(student_codes, 1):
            print(f"\n--- Evaluating submission {i}/{len(student_codes)} ---")
            try:
                result = self.evaluate_code(
                    student_code,
                    reference_code,
                    requirements,
                    language,
                    combination_method
                )
                result['submission_index'] = i - 1
                results.append(result)
            except Exception as e:
                print(f"Error evaluating submission {i}: {str(e)}")
                results.append({
                    'submission_index': i - 1,
                    'error': str(e),
                    'status': 'failed'
                })

        # Summary
        print("\n" + "="*60)
        print("Batch Evaluation Summary")
        print("="*60)
        successful = sum(1 for r in results if r.get('status') != 'failed')
        print(f"Successful evaluations: {successful}/{len(student_codes)}")

        if successful > 0:
            avg_score = sum(r['final_score_percent'] for r in results if 'final_score_percent' in r) / successful
            print(f"Average score: {avg_score:.2f}%")

        return results

    def generate_report(
        self,
        evaluation_results: Dict[str, any],
        student_info: Optional[Dict[str, str]] = None,
        requirements: Optional[str] = None,
        output_format: str = 'all',
        output_dir: str = 'reports'
    ) -> Dict[str, str]:
        """
        Generate evaluation report.

        Args:
            evaluation_results: Evaluation results
            student_info: Optional student information
            requirements: Optional requirements
            output_format: Format ('text', 'html', 'json', or 'all')
            output_dir: Output directory

        Returns:
            Dictionary with paths to generated reports
        """
        print("\nGenerating reports...")

        if output_format == 'all':
            paths = self.report_generator.generate_all_reports(
                evaluation_results,
                student_info,
                requirements,
                output_dir=output_dir
            )
        else:
            # Generate specific format
            if output_format == 'text':
                content = self.report_generator.generate_text_report(
                    evaluation_results, student_info, requirements
                )
            elif output_format == 'html':
                content = self.report_generator.generate_html_report(
                    evaluation_results, student_info, requirements
                )
            elif output_format == 'json':
                content = self.report_generator.generate_json_report(
                    evaluation_results, student_info, requirements
                )
            else:
                raise ValueError(f"Unknown output format: {output_format}")

            path = self.report_generator.save_report(
                content,
                f"evaluation_report.{output_format}",
                output_dir
            )
            paths = {output_format: path}

        print("Reports generated:")
        for format_type, path in paths.items():
            print(f"  {format_type.upper()}: {path}")

        return paths


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Code Inspector - Evaluate code similarity")
    parser.add_argument('--student-code', type=str, help='Path to student code file')
    parser.add_argument('--reference-code', type=str, help='Path to reference code file')
    parser.add_argument('--github-url', type=str, help='GitHub repository URL')
    parser.add_argument('--requirements', type=str, help='Path to requirements file')
    parser.add_argument('--language', type=str, default='python', help='Programming language')
    parser.add_argument('--output-dir', type=str, default='reports', help='Output directory')
    parser.add_argument('--format', type=str, default='all', choices=['text', 'html', 'json', 'all'],
                       help='Output format')

    args = parser.parse_args()

    # Initialize Code Inspector
    inspector = CodeInspector()

    # Load reference code
    if not args.reference_code:
        print("Error: --reference-code is required")
        return

    with open(args.reference_code, 'r', encoding='utf-8') as f:
        reference_code = f.read()

    # Load requirements if provided
    requirements = None
    if args.requirements and os.path.exists(args.requirements):
        with open(args.requirements, 'r', encoding='utf-8') as f:
            requirements = f.read()

    # Evaluate
    if args.github_url:
        # Evaluate GitHub project
        results = inspector.evaluate_github_project(
            args.github_url,
            reference_code,
            requirements,
            args.language
        )
    elif args.student_code:
        # Evaluate local file
        with open(args.student_code, 'r', encoding='utf-8') as f:
            student_code = f.read()

        results = inspector.evaluate_code(
            student_code,
            reference_code,
            requirements,
            args.language
        )
    else:
        print("Error: Either --student-code or --github-url must be provided")
        return

    # Generate report
    if results.get('status') != 'failed':
        inspector.generate_report(
            results,
            output_format=args.format,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    # Check if running with arguments
    if len(sys.argv) > 1:
        main()
    else:
        # Demo mode
        print("Code Inspector - Demo Mode")
        print("Use --help to see command-line options")
        print("\nExample usage:")
        print("  python main.py --student-code student.py --reference-code reference.py")
        print("  python main.py --github-url https://github.com/user/repo --reference-code reference.py")
