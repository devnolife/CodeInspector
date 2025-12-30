# Sample Code Files

This directory contains sample code files for testing the Code Inspector system.

## Files

### Reference Code
- **reference_code.py**: The expected/ideal implementation of a simple calculator

### Student Code Examples

1. **student_code_high_similarity.py**
   - Very similar to reference code
   - Uses different variable names but same logic
   - Expected similarity: 80-90%
   - Grade: A or B

2. **student_code_medium_similarity.py**
   - Uses class-based approach instead of functions
   - Different naming conventions
   - Achieves same functionality with different structure
   - Expected similarity: 60-75%
   - Grade: C or B

3. **student_code_low_similarity.py**
   - Completely different implementation approach
   - Uses expression parsing instead of separate functions
   - Different architecture but achieves similar results
   - Expected similarity: 40-55%
   - Grade: D or C

### Requirements
- **requirements.txt**: Project requirements document describing what the calculator should do

## Testing the Samples

### Using Command Line

```bash
# Test high similarity
python main.py --student-code samples/student_code_high_similarity.py --reference-code samples/reference_code.py --requirements samples/requirements.txt

# Test medium similarity
python main.py --student-code samples/student_code_medium_similarity.py --reference-code samples/reference_code.py

# Test low similarity
python main.py --student-code samples/student_code_low_similarity.py --reference-code samples/reference_code.py
```

### Using Web Interface

1. Start the web server:
   ```bash
   python app.py
   ```

2. Open browser to http://localhost:5000

3. Upload the files:
   - Student Code: Choose one of the student_code_*.py files
   - Reference Code: reference_code.py
   - Requirements: requirements.txt (optional)

4. Click "Evaluate Code" to see results

## Expected Results

The Code Inspector should be able to:
- Identify that high similarity code is very close to reference (85-95%)
- Recognize medium similarity despite different structure (60-75%)
- Detect low similarity for completely different approaches (40-55%)
- Provide recommendations for each case
- Generate detailed reports with token analysis
