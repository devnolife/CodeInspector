# 🚀 Quick Start Guide

Get Code Inspector up and running in 5 minutes!

## Prerequisites

- Python 3.8+ installed
- Internet connection (for downloading CodeBERT model)

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

⏳ **Note**: First installation takes 5-10 minutes to download CodeBERT model (~500MB)

### 2. Verify Installation

```bash
# Quick test with sample data
python main.py --student-code samples/student_code_high_similarity.py --reference-code samples/reference_code.py
```

✅ If you see evaluation results, installation is successful!

## Usage

### Option 1: Web Interface (Easiest)

1. **Start the server**:
   ```bash
   python app.py
   ```

2. **Open browser**:
   ```
   http://localhost:5000
   ```

3. **Upload code and evaluate**!

### Option 2: Command Line

```bash
python main.py \
  --student-code path/to/student.py \
  --reference-code path/to/reference.py
```

### Option 3: Python Script

```python
from main import CodeInspector

inspector = CodeInspector()

results = inspector.evaluate_code(
    student_code="""
def add(a, b):
    return a + b
    """,
    reference_code="""
def add(x, y):
    return x + y
    """,
    language='python'
)

print(f"Similarity: {results['final_score_percent']:.2f}%")
print(f"Grade: {results['grade']}")
```

## Try the Samples

We've included sample code files for testing:

```bash
# High similarity (expect ~90%)
python main.py --student-code samples/student_code_high_similarity.py --reference-code samples/reference_code.py

# Medium similarity (expect ~65%)
python main.py --student-code samples/student_code_medium_similarity.py --reference-code samples/reference_code.py

# Low similarity (expect ~45%)
python main.py --student-code samples/student_code_low_similarity.py --reference-code samples/reference_code.py
```

## Understanding Results

The system provides:

- **Final Score**: Combined similarity (0-100%)
- **Grade**: A, B, C, D, or F
- **CodeBERT Score**: Semantic similarity
- **Token Score**: Structural similarity
- **Recommendations**: What to improve

### Example Output

```
Starting Code Evaluation
==============================
[1/4] Preprocessing code for CodeBERT...
[2/4] Preprocessing code for Token Similarity...
[3/4] Running CodeBERT evaluation...
    CodeBERT similarity: 87.50%
[4/4] Running Token Similarity evaluation...
    Token similarity: 82.30%

Combining scores...
Final Score: 85.50%
Grade: B (Good)
Status: PASSED
==============================
```

## Next Steps

- Read the [full README](README.md) for detailed documentation
- Explore the [samples](samples/) directory
- Check out the [methodology](plan.md) for research details
- Customize weights and thresholds in code

## Common Issues

### Issue: "torch not found"
**Solution**: Install PyTorch
```bash
pip install torch
```

### Issue: "Model download slow"
**Solution**: First run downloads ~500MB model. Be patient or use GPU-enabled environment.

### Issue: "Out of memory"
**Solution**: CodeBERT uses ~2GB RAM. Close other applications or use smaller code samples.

### Issue: "GitHub rate limit"
**Solution**: Set GITHUB_TOKEN environment variable with your personal access token.

## Configuration Tips

### Adjust Pass Threshold

```python
inspector = CodeInspector(pass_threshold=0.75)  # 75% required to pass
```

### Change Weights

```python
inspector = CodeInspector(
    codebert_weight=0.7,  # 70% weight to CodeBERT
    token_weight=0.3      # 30% weight to Token similarity
)
```

### Use Different Combination Method

```python
results = inspector.evaluate_code(
    student_code=code1,
    reference_code=code2,
    combination_method='harmonic'  # or 'average', 'max', 'min'
)
```

## Support

Need help?
- Check the [full documentation](README.md)
- Review [sample code](samples/)
- Open an issue on GitHub

---

Happy Code Evaluating! 🎉
