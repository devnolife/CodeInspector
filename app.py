"""
Flask Web Application for Code Inspector
Provides a web interface for code evaluation.
"""

import os
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from werkzeug.utils import secure_filename
import json
from datetime import datetime

from main import CodeInspector


# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'codeinspector-secret-key-change-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary directories
os.makedirs('uploads', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# Initialize Code Inspector
inspector = CodeInspector()

# Store results temporarily (in production, use database)
evaluation_cache = {}


@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')


@app.route('/evaluate', methods=['POST'])
def evaluate():
    """Evaluate code submission."""
    try:
        # Get form data
        evaluation_type = request.form.get('evaluation_type', 'code')
        language = request.form.get('language', 'python')
        requirements = request.form.get('requirements', '')

        # Get reference code
        if 'reference_file' in request.files and request.files['reference_file'].filename:
            reference_file = request.files['reference_file']
            reference_code = reference_file.read().decode('utf-8')
        else:
            reference_code = request.form.get('reference_code', '')

        if not reference_code:
            return jsonify({
                'error': 'Reference code is required',
                'status': 'failed'
            }), 400

        # Get student code based on evaluation type
        if evaluation_type == 'github':
            github_url = request.form.get('github_url', '')
            if not github_url:
                return jsonify({
                    'error': 'GitHub URL is required',
                    'status': 'failed'
                }), 400

            # Evaluate GitHub project
            results = inspector.evaluate_github_project(
                github_url,
                reference_code,
                requirements if requirements else None,
                language
            )

            student_info = {
                'github_url': github_url,
                'evaluation_type': 'GitHub Repository'
            }

        else:  # code evaluation
            if 'student_file' in request.files and request.files['student_file'].filename:
                student_file = request.files['student_file']
                student_code = student_file.read().decode('utf-8')
                student_info = {
                    'filename': student_file.filename,
                    'evaluation_type': 'File Upload'
                }
            else:
                student_code = request.form.get('student_code', '')
                student_info = {
                    'evaluation_type': 'Direct Input'
                }

            if not student_code:
                return jsonify({
                    'error': 'Student code is required',
                    'status': 'failed'
                }), 400

            # Evaluate code
            results = inspector.evaluate_code(
                student_code,
                reference_code,
                requirements if requirements else None,
                language
            )

        # Check for errors
        if results.get('status') == 'failed':
            return jsonify(results), 400

        # Generate evaluation ID
        eval_id = datetime.now().strftime('%Y%m%d%H%M%S')
        evaluation_cache[eval_id] = {
            'results': results,
            'student_info': student_info,
            'requirements': requirements,
            'timestamp': datetime.now().isoformat()
        }

        # Return results
        return jsonify({
            'status': 'success',
            'eval_id': eval_id,
            'results': results
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'failed'
        }), 500


@app.route('/report/<eval_id>')
def view_report(eval_id):
    """View evaluation report."""
    if eval_id not in evaluation_cache:
        return "Report not found", 404

    data = evaluation_cache[eval_id]
    return render_template(
        'report.html',
        results=data['results'],
        student_info=data['student_info'],
        requirements=data.get('requirements')
    )


@app.route('/download/<eval_id>/<format>')
def download_report(eval_id, format):
    """Download evaluation report."""
    if eval_id not in evaluation_cache:
        return "Report not found", 404

    if format not in ['text', 'html', 'json']:
        return "Invalid format", 400

    data = evaluation_cache[eval_id]

    # Generate report
    if format == 'text':
        content = inspector.report_generator.generate_text_report(
            data['results'],
            data['student_info'],
            data.get('requirements')
        )
        mimetype = 'text/plain'
        extension = 'txt'
    elif format == 'html':
        content = inspector.report_generator.generate_html_report(
            data['results'],
            data['student_info'],
            data.get('requirements')
        )
        mimetype = 'text/html'
        extension = 'html'
    else:  # json
        content = inspector.report_generator.generate_json_report(
            data['results'],
            data['student_info'],
            data.get('requirements')
        )
        mimetype = 'application/json'
        extension = 'json'

    # Save temporarily
    filename = f"evaluation_report_{eval_id}.{extension}"
    filepath = os.path.join('reports', filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    return send_file(
        filepath,
        mimetype=mimetype,
        as_attachment=True,
        download_name=filename
    )


@app.route('/batch-evaluate', methods=['POST'])
def batch_evaluate():
    """Batch evaluation endpoint."""
    try:
        # Get reference code
        reference_code = request.form.get('reference_code', '')
        requirements = request.form.get('requirements', '')
        language = request.form.get('language', 'python')

        if not reference_code:
            return jsonify({
                'error': 'Reference code is required',
                'status': 'failed'
            }), 400

        # Get student codes from uploaded files
        student_codes = []
        files = request.files.getlist('student_files')

        for file in files:
            if file and file.filename:
                code = file.read().decode('utf-8')
                student_codes.append(code)

        if not student_codes:
            return jsonify({
                'error': 'No student files provided',
                'status': 'failed'
            }), 400

        # Batch evaluate
        results = inspector.batch_evaluate(
            student_codes,
            reference_code,
            requirements if requirements else None,
            language
        )

        # Store results
        eval_id = datetime.now().strftime('%Y%m%d%H%M%S')
        evaluation_cache[eval_id] = {
            'results': results,
            'batch': True,
            'timestamp': datetime.now().isoformat()
        }

        return jsonify({
            'status': 'success',
            'eval_id': eval_id,
            'results': results,
            'total': len(results)
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'failed'
        }), 500


@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    print("="*60)
    print("Code Inspector Web Interface")
    print("="*60)
    print("Starting Flask server...")
    print("Open your browser and navigate to: http://localhost:5000")
    print("="*60)

    app.run(debug=True, host='0.0.0.0', port=5000)
