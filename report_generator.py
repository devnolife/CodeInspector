"""
Report Generator Module
Generates evaluation reports in various formats (text, HTML, JSON).
"""

import json
from typing import Dict, List, Optional
from datetime import datetime
import os


class ReportGenerator:
    """Generates evaluation reports."""

    def __init__(self):
        """Initialize Report Generator."""
        self.report_version = "1.0"

    def generate_text_report(
        self,
        evaluation_results: Dict[str, any],
        student_info: Optional[Dict[str, str]] = None,
        requirements: Optional[str] = None
    ) -> str:
        """
        Generate text-based evaluation report.

        Args:
            evaluation_results: Comprehensive evaluation results
            student_info: Optional student information
            requirements: Optional requirements text

        Returns:
            Text report string
        """
        lines = []
        lines.append("="*70)
        lines.append("CODE EVALUATION REPORT")
        lines.append("="*70)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Report Version: {self.report_version}")
        lines.append("")

        # Student information
        if student_info:
            lines.append("-"*70)
            lines.append("STUDENT INFORMATION")
            lines.append("-"*70)
            for key, value in student_info.items():
                lines.append(f"{key.replace('_', ' ').title()}: {value}")
            lines.append("")

        # Requirements
        if requirements:
            lines.append("-"*70)
            lines.append("PROJECT REQUIREMENTS")
            lines.append("-"*70)
            lines.append(requirements[:500] + "..." if len(requirements) > 500 else requirements)
            lines.append("")

        # Evaluation Summary
        lines.append("-"*70)
        lines.append("EVALUATION SUMMARY")
        lines.append("-"*70)
        lines.append(f"Final Score: {evaluation_results['final_score_percent']:.2f}%")
        lines.append(f"Grade: {evaluation_results['grade']}")
        lines.append(f"Category: {evaluation_results['category']}")
        lines.append(f"Status: {'PASSED' if evaluation_results['passed'] else 'FAILED'}")
        lines.append(f"Threshold: {evaluation_results['threshold_percent']:.2f}%")
        lines.append("")

        # Individual Method Scores
        lines.append("-"*70)
        lines.append("DETAILED SCORES")
        lines.append("-"*70)
        lines.append(f"CodeBERT Similarity:  {evaluation_results['codebert_score_percent']:.2f}%")
        lines.append(f"Token Similarity:     {evaluation_results['token_score_percent']:.2f}%")
        lines.append(f"Combined Score:       {evaluation_results['final_score_percent']:.2f}%")
        lines.append("")
        lines.append(f"Combination Method: {evaluation_results['combination_method']}")
        lines.append(f"Weights: CodeBERT={evaluation_results['weights']['codebert']}, "
                    f"Token={evaluation_results['weights']['token']}")
        lines.append("")

        # Token Analysis Details
        if 'detailed_token_analysis' in evaluation_results:
            token_details = evaluation_results['detailed_token_analysis']
            lines.append("-"*70)
            lines.append("TOKEN ANALYSIS")
            lines.append("-"*70)
            lines.append(f"Identifier Similarity: {token_details.get('identifier_similarity_percent', 0):.2f}%")

            if 'common_identifiers' in token_details and token_details['common_identifiers']:
                lines.append(f"Common Identifiers: {', '.join(token_details['common_identifiers'][:10])}")
                if len(token_details['common_identifiers']) > 10:
                    lines.append(f"  ... and {len(token_details['common_identifiers']) - 10} more")

            if 'missing_identifiers' in token_details and token_details['missing_identifiers']:
                lines.append(f"Missing Identifiers: {', '.join(token_details['missing_identifiers'][:10])}")
                if len(token_details['missing_identifiers']) > 10:
                    lines.append(f"  ... and {len(token_details['missing_identifiers']) - 10} more")

            if 'extra_identifiers' in token_details and token_details['extra_identifiers']:
                lines.append(f"Extra Identifiers: {', '.join(token_details['extra_identifiers'][:10])}")
                if len(token_details['extra_identifiers']) > 10:
                    lines.append(f"  ... and {len(token_details['extra_identifiers']) - 10} more")
            lines.append("")

        # Recommendations
        if 'recommendations' in evaluation_results and evaluation_results['recommendations']:
            lines.append("-"*70)
            lines.append("RECOMMENDATIONS")
            lines.append("-"*70)
            for i, rec in enumerate(evaluation_results['recommendations'], 1):
                lines.append(f"{i}. {rec}")
            lines.append("")

        lines.append("="*70)
        lines.append("END OF REPORT")
        lines.append("="*70)

        return '\n'.join(lines)

    def generate_html_report(
        self,
        evaluation_results: Dict[str, any],
        student_info: Optional[Dict[str, str]] = None,
        requirements: Optional[str] = None
    ) -> str:
        """
        Generate HTML evaluation report.

        Args:
            evaluation_results: Comprehensive evaluation results
            student_info: Optional student information
            requirements: Optional requirements text

        Returns:
            HTML report string
        """
        # Determine color based on grade
        grade_colors = {
            'A': '#28a745',
            'B': '#5cb85c',
            'C': '#ffc107',
            'D': '#fd7e14',
            'F': '#dc3545'
        }
        grade = evaluation_results.get('grade', 'F')
        grade_color = grade_colors.get(grade, '#6c757d')

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Code Evaluation Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #007bff;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 8px;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        .grade-badge {{
            display: inline-block;
            background-color: {grade_color};
            color: white;
            padding: 15px 30px;
            border-radius: 50px;
            font-size: 32px;
            font-weight: bold;
            margin: 20px 0;
        }}
        .score-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .score-card {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #007bff;
        }}
        .score-label {{
            color: #666;
            font-size: 14px;
            margin-bottom: 5px;
        }}
        .score-value {{
            font-size: 28px;
            font-weight: bold;
            color: #333;
        }}
        .status {{
            padding: 8px 16px;
            border-radius: 4px;
            display: inline-block;
            font-weight: bold;
        }}
        .status.passed {{
            background-color: #d4edda;
            color: #155724;
        }}
        .status.failed {{
            background-color: #f8d7da;
            color: #721c24;
        }}
        .info-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        .info-table td {{
            padding: 10px;
            border-bottom: 1px solid #e0e0e0;
        }}
        .info-table td:first-child {{
            font-weight: bold;
            color: #555;
            width: 30%;
        }}
        .recommendations {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 15px 0;
        }}
        .recommendations li {{
            margin: 10px 0;
        }}
        .token-list {{
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
            font-family: 'Courier New', monospace;
            font-size: 14px;
        }}
        .footer {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 2px solid #e0e0e0;
            text-align: center;
            color: #666;
            font-size: 14px;
        }}
        .progress-bar {{
            width: 100%;
            height: 30px;
            background-color: #e0e0e0;
            border-radius: 15px;
            overflow: hidden;
            margin: 10px 0;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #007bff, #0056b3);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Code Evaluation Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
"""

        # Student Information
        if student_info:
            html += """
        <h2>Student Information</h2>
        <table class="info-table">
"""
            for key, value in student_info.items():
                html += f"""
            <tr>
                <td>{key.replace('_', ' ').title()}</td>
                <td>{value}</td>
            </tr>
"""
            html += """
        </table>
"""

        # Evaluation Summary
        status_class = 'passed' if evaluation_results['passed'] else 'failed'
        status_text = 'PASSED' if evaluation_results['passed'] else 'FAILED'

        html += f"""
        <h2>Evaluation Summary</h2>
        <div class="header">
            <div class="grade-badge">{grade}</div>
            <p style="font-size: 20px; color: #666;">{evaluation_results['category']}</p>
            <span class="status {status_class}">{status_text}</span>
        </div>

        <div class="progress-bar">
            <div class="progress-fill" style="width: {evaluation_results['final_score_percent']}%">
                {evaluation_results['final_score_percent']:.1f}%
            </div>
        </div>

        <h2>Detailed Scores</h2>
        <div class="score-container">
            <div class="score-card">
                <div class="score-label">CodeBERT Similarity</div>
                <div class="score-value">{evaluation_results['codebert_score_percent']:.1f}%</div>
            </div>
            <div class="score-card">
                <div class="score-label">Token Similarity</div>
                <div class="score-value">{evaluation_results['token_score_percent']:.1f}%</div>
            </div>
            <div class="score-card" style="border-left-color: {grade_color};">
                <div class="score-label">Combined Score</div>
                <div class="score-value">{evaluation_results['final_score_percent']:.1f}%</div>
            </div>
        </div>

        <table class="info-table">
            <tr>
                <td>Combination Method</td>
                <td>{evaluation_results['combination_method']}</td>
            </tr>
            <tr>
                <td>Weights</td>
                <td>CodeBERT: {evaluation_results['weights']['codebert']}, Token: {evaluation_results['weights']['token']}</td>
            </tr>
            <tr>
                <td>Pass Threshold</td>
                <td>{evaluation_results['threshold_percent']:.1f}%</td>
            </tr>
        </table>
"""

        # Token Analysis
        if 'detailed_token_analysis' in evaluation_results:
            token_details = evaluation_results['detailed_token_analysis']
            html += f"""
        <h2>Token Analysis</h2>
        <table class="info-table">
            <tr>
                <td>Identifier Similarity</td>
                <td>{token_details.get('identifier_similarity_percent', 0):.1f}%</td>
            </tr>
        </table>
"""
            if token_details.get('common_identifiers'):
                html += f"""
        <p><strong>Common Identifiers:</strong></p>
        <div class="token-list">{', '.join(token_details['common_identifiers'][:20])}</div>
"""
            if token_details.get('missing_identifiers'):
                html += f"""
        <p><strong>Missing Identifiers:</strong></p>
        <div class="token-list">{', '.join(token_details['missing_identifiers'][:20])}</div>
"""

        # Recommendations
        if evaluation_results.get('recommendations'):
            html += """
        <h2>Recommendations</h2>
        <div class="recommendations">
            <ul>
"""
            for rec in evaluation_results['recommendations']:
                html += f"                <li>{rec}</li>\n"
            html += """
            </ul>
        </div>
"""

        html += f"""
        <div class="footer">
            <p>Code Inspector v{self.report_version} | Powered by CodeBERT and Token Similarity Analysis</p>
        </div>
    </div>
</body>
</html>
"""
        return html

    def generate_json_report(
        self,
        evaluation_results: Dict[str, any],
        student_info: Optional[Dict[str, str]] = None,
        requirements: Optional[str] = None
    ) -> str:
        """
        Generate JSON evaluation report.

        Args:
            evaluation_results: Comprehensive evaluation results
            student_info: Optional student information
            requirements: Optional requirements text

        Returns:
            JSON report string
        """
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_version': self.report_version
            },
            'student_info': student_info or {},
            'requirements': requirements,
            'evaluation': evaluation_results
        }

        return json.dumps(report, indent=2)

    def save_report(
        self,
        report_content: str,
        filename: str,
        output_dir: str = "reports"
    ) -> str:
        """
        Save report to file.

        Args:
            report_content: Report content string
            filename: Output filename
            output_dir: Output directory (default: "reports")

        Returns:
            Path to saved file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Full output path
        output_path = os.path.join(output_dir, filename)

        # Save file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        return output_path

    def generate_all_reports(
        self,
        evaluation_results: Dict[str, any],
        student_info: Optional[Dict[str, str]] = None,
        requirements: Optional[str] = None,
        base_filename: str = "evaluation_report",
        output_dir: str = "reports"
    ) -> Dict[str, str]:
        """
        Generate and save all report formats.

        Args:
            evaluation_results: Comprehensive evaluation results
            student_info: Optional student information
            requirements: Optional requirements text
            base_filename: Base filename (without extension)
            output_dir: Output directory

        Returns:
            Dictionary with paths to saved reports
        """
        # Add timestamp to filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"{base_filename}_{timestamp}"

        # Generate reports
        text_report = self.generate_text_report(evaluation_results, student_info, requirements)
        html_report = self.generate_html_report(evaluation_results, student_info, requirements)
        json_report = self.generate_json_report(evaluation_results, student_info, requirements)

        # Save reports
        paths = {
            'text': self.save_report(text_report, f"{base_name}.txt", output_dir),
            'html': self.save_report(html_report, f"{base_name}.html", output_dir),
            'json': self.save_report(json_report, f"{base_name}.json", output_dir)
        }

        return paths


# Example usage
if __name__ == "__main__":
    # Sample evaluation results
    sample_results = {
        'final_score': 0.82,
        'final_score_percent': 82.0,
        'passed': True,
        'grade': 'B',
        'category': 'Good',
        'codebert_score': 0.85,
        'codebert_score_percent': 85.0,
        'token_score': 0.78,
        'token_score_percent': 78.0,
        'combination_method': 'weighted',
        'weights': {'codebert': 0.6, 'token': 0.4},
        'threshold': 0.7,
        'threshold_percent': 70.0,
        'recommendations': [
            'Code shows good similarity to reference implementation.',
            'Consider using more consistent naming conventions.'
        ],
        'detailed_token_analysis': {
            'identifier_similarity_percent': 75.0,
            'common_identifiers': ['calculate', 'sum', 'result', 'main'],
            'missing_identifiers': ['process', 'validate'],
            'extra_identifiers': ['compute', 'check']
        }
    }

    sample_student_info = {
        'name': 'John Doe',
        'student_id': '12345',
        'project_url': 'https://github.com/johndoe/calculator'
    }

    print("Generating reports...")
    generator = ReportGenerator()

    # Generate text report
    text_report = generator.generate_text_report(sample_results, sample_student_info)
    print("\nText Report:")
    print(text_report)

    # Save all reports
    # paths = generator.generate_all_reports(sample_results, sample_student_info)
    # print(f"\nReports saved:")
    # for format_type, path in paths.items():
    #     print(f"{format_type.upper()}: {path}")
