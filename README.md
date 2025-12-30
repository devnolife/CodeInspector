# 🔍 Code Inspector

**Automatic Code Evaluation System Using CodeBERT and Token-based Similarity**

A comprehensive research project for evaluating code similarity using state-of-the-art deep learning (CodeBERT) combined with traditional token-based analysis methods.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Research Methodology](#research-methodology)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

Code Inspector is an automated code evaluation system designed to assess the similarity between student submissions and reference implementations. It combines:

1. **CodeBERT Analysis**: Deep learning-based semantic understanding of code
2. **Token Similarity**: Traditional token-based comparison for structural analysis
3. **Combined Scoring**: Weighted combination of both methods for accurate evaluation

### Research Objective

This system aims to measure the accuracy of combining CodeBERT and token-based similarity methods in evaluating functional equivalence of code without executing it.

## ✨ Features

- **Dual Evaluation Methods**
  - CodeBERT semantic embeddings
  - Token-based similarity (Jaccard, Dice coefficients)

- **Multiple Input Sources**
  - Direct code input
  - File upload
  - GitHub repository analysis

- **Comprehensive Reporting**
  - HTML reports with visualizations
  - JSON exports for further analysis
  - Text reports for documentation

- **Web Interface**
  - User-friendly Flask-based UI
  - Real-time evaluation
  - Batch processing support

- **Accuracy Measurement**
  - MAE, RMSE, R² metrics
  - Classification accuracy
  - Error analysis and visualization

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Code Inspector                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Input Layer                                             │
│  ├─ GitHub Repository Manager                           │
│  ├─ File Upload Handler                                 │
│  └─ Code Preprocessor                                    │
│                                                          │
│  Evaluation Layer                                        │
│  ├─ CodeBERT Evaluator                                  │
│  │   ├─ Model: microsoft/codebert-base                  │
│  │   ├─ Embedding Generation                            │
│  │   └─ Cosine Similarity                               │
│  │                                                       │
│  └─ Token Similarity Evaluator                          │
│      ├─ Tokenization                                     │
│      ├─ Identifier Extraction                           │
│      └─ Jaccard/Dice Similarity                         │
│                                                          │
│  Combination Layer                                       │
│  └─ Score Combiner                                       │
│      ├─ Weighted Average                                 │
│      ├─ Pass/Fail Decision                              │
│      └─ Recommendations                                  │
│                                                          │
│  Output Layer                                            │
│  ├─ Report Generator (HTML/JSON/Text)                   │
│  ├─ Accuracy Calculator                                  │
│  └─ Visualization                                        │
│                                                          │
│  Interface Layer                                         │
│  ├─ Flask Web Application                               │
│  └─ Command-line Interface                              │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for GitHub integration)
- (Optional) CUDA-capable GPU for faster CodeBERT inference

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/CodeInspector.git
cd CodeInspector
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: First installation may take several minutes as it downloads the CodeBERT model (~500MB).

### Step 4: Verify Installation

```bash
# Test with sample data
python main.py --student-code samples/student_code_high_similarity.py --reference-code samples/reference_code.py
```

## 📖 Usage

### Web Interface (Recommended)

1. Start the web server:
   ```bash
   python app.py
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

3. Upload or paste code and click "Evaluate Code"

### Command-line Interface

#### Evaluate Local Files

```bash
python main.py \
  --student-code path/to/student.py \
  --reference-code path/to/reference.py \
  --requirements path/to/requirements.txt \
  --language python \
  --output-dir reports \
  --format html
```

#### Evaluate GitHub Repository

```bash
python main.py \
  --github-url https://github.com/student/project \
  --reference-code path/to/reference.py \
  --language python
```

### Python API

```python
from main import CodeInspector

# Initialize
inspector = CodeInspector(
    codebert_weight=0.6,
    token_weight=0.4,
    pass_threshold=0.7
)

# Evaluate code
results = inspector.evaluate_code(
    student_code="def add(a, b): return a + b",
    reference_code="def add(x, y): return x + y",
    language='python'
)

# Generate report
inspector.generate_report(
    results,
    student_info={'name': 'John Doe'},
    output_format='html'
)
```

## 🔬 Research Methodology

### Evaluation Process

1. **Preprocessing**
   - Code normalization
   - Comment removal (configurable)
   - Whitespace standardization

2. **CodeBERT Analysis**
   - Convert code to embeddings using pre-trained CodeBERT
   - Calculate cosine similarity between embeddings
   - Generate semantic similarity score (0-1)

3. **Token Analysis**
   - Extract identifiers, keywords, and tokens
   - Calculate Jaccard similarity
   - Generate structural similarity score (0-1)

4. **Score Combination**
   - Default: Weighted average (CodeBERT: 60%, Token: 40%)
   - Alternative methods: Average, Max, Min, Harmonic mean

5. **Grading**
   - A: ≥90% similarity
   - B: 80-89%
   - C: 70-79%
   - D: 60-69%
   - F: <60%

### Accuracy Metrics

The system measures its own accuracy using:

- **Mean Absolute Error (MAE)**: Average prediction error
- **Root Mean Squared Error (RMSE)**: Error with penalty for large deviations
- **R² Score**: Correlation between predictions and ground truth
- **Classification Accuracy**: Pass/fail decision accuracy

## 📁 Project Structure

```
CodeInspector/
│
├── main.py                          # Main orchestrator
├── app.py                           # Flask web application
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
│
├── Core Modules/
│   ├── github_manager.py           # GitHub integration
│   ├── code_preprocessor.py        # Code preprocessing
│   ├── codebert_evaluator.py       # CodeBERT evaluation
│   ├── token_similarity_evaluator.py # Token-based evaluation
│   ├── score_combiner.py           # Score combination
│   ├── accuracy_calculator.py      # Accuracy metrics
│   └── report_generator.py         # Report generation
│
├── templates/                       # HTML templates
│   ├── index.html                  # Main page
│   └── report.html                 # Report page
│
├── samples/                         # Sample data
│   ├── reference_code.py
│   ├── student_code_high_similarity.py
│   ├── student_code_medium_similarity.py
│   ├── student_code_low_similarity.py
│   └── requirements.txt
│
├── reports/                         # Generated reports (created at runtime)
├── uploads/                         # Uploaded files (created at runtime)
└── data/                           # Dataset storage (optional)
```

## 🔧 Configuration

### Adjusting Weights

Edit the initialization in `main.py` or `app.py`:

```python
inspector = CodeInspector(
    codebert_weight=0.6,    # CodeBERT importance (0-1)
    token_weight=0.4,       # Token similarity importance (0-1)
    pass_threshold=0.7      # Minimum score to pass (0-1)
)
```

### Combination Methods

Available methods in `score_combiner.py`:
- `weighted`: Custom weights (default)
- `average`: Simple average
- `max`: Take maximum score
- `min`: Take minimum score (conservative)
- `harmonic`: Harmonic mean (penalizes low scores)

### Supported Languages

Currently supported:
- Python (.py)
- Java (.java)
- JavaScript (.js)
- C++ (.cpp)
- C (.c)

To add more languages, extend the preprocessor and tokenizer.

## 📊 API Reference

### CodeInspector Class

```python
class CodeInspector:
    def __init__(self, github_token=None, codebert_weight=0.6,
                 token_weight=0.4, pass_threshold=0.7)

    def evaluate_code(self, student_code, reference_code,
                     requirements=None, language='python',
                     combination_method='weighted') -> Dict

    def evaluate_github_project(self, student_url, reference_code,
                               requirements=None, language='python') -> Dict

    def batch_evaluate(self, student_codes, reference_code,
                      requirements=None, language='python') -> List[Dict]

    def generate_report(self, evaluation_results, student_info=None,
                       requirements=None, output_format='all') -> Dict[str, str]
```

### Web API Endpoints

- `GET /`: Home page
- `POST /evaluate`: Evaluate code submission
- `GET /report/<eval_id>`: View evaluation report
- `GET /download/<eval_id>/<format>`: Download report (html/json/text)
- `POST /batch-evaluate`: Batch evaluation
- `GET /api/health`: Health check

## 🧪 Testing

### Run Sample Evaluations

```bash
# Test high similarity
python main.py --student-code samples/student_code_high_similarity.py --reference-code samples/reference_code.py

# Test medium similarity
python main.py --student-code samples/student_code_medium_similarity.py --reference-code samples/reference_code.py

# Test low similarity
python main.py --student-code samples/student_code_low_similarity.py --reference-code samples/reference_code.py
```

### Expected Results

- **High Similarity**: 85-95% combined score, Grade A/B
- **Medium Similarity**: 60-75% combined score, Grade B/C
- **Low Similarity**: 40-55% combined score, Grade C/D

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@software{codeinspector2024,
  title={Code Inspector: Automated Code Evaluation Using CodeBERT and Token Similarity},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/CodeInspector}
}
```

## 🙏 Acknowledgments

- Microsoft Research for the [CodeBERT](https://github.com/microsoft/CodeBERT) model
- HuggingFace for the Transformers library
- Flask framework for web interface

## 📞 Support

For questions, issues, or feature requests:
- Open an issue on GitHub
- Email: your.email@example.com

## 🗺️ Roadmap

Future enhancements:
- [ ] Support for more programming languages
- [ ] Custom model fine-tuning
- [ ] Plagiarism detection
- [ ] Code quality metrics
- [ ] Integration with LMS platforms
- [ ] Real-time collaboration features
- [ ] Advanced visualization dashboards

---

**Built with ❤️ for Computer Science Education**
