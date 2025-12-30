# Project Planning: Sistem Evaluasi Code Berbasis CodeBERT dan Token Similarity

## 1. Gambaran Umum Penelitian

### Judul Penelitian
**"Studi Akurasi Kombinasi CodeBERT dan Token-based Similarity untuk Mengevaluasi Functional Equivalence Code"**

### Tujuan
Mengukur akurasi dua metode (CodeBERT + Token Similarity) dalam mengevaluasi kesesuaian code mahasiswa dengan expected outcomes, tanpa harus menjalankan code.

### Target Output
- Sistem evaluasi otomatis
- Similarity score (0-100%)
- Accuracy report dari kedua metode
- Analisis perbandingan akurasi

---

## 2. Komponen Sistem

### 2.1 Input Management
**Yang perlu disiapkan:**
- GitHub project URL dari mahasiswa
- Text description of requirements (ekspektasi proyek)
- Reference solution code (kode ideal/standar)

**Deliverables:**
- GitHub repository URL parser
- Requirements documentation format standardizer
- Reference code storage system

---

### 2.2 CodeBERT Module

**Fungsi:**
- Convert code menjadi semantic embeddings
- Understand code intent/functionality secara semantic
- Generate embedding untuk student code, reference code, requirements

**Components yang perlu dibuat:**
- CodeBERT model loader (from HuggingFace)
- Code preprocessing pipeline (standardisasi format input)
- Embedding generator (generate vector representations)
- Semantic similarity calculator (cosine similarity antara embeddings)

**Input:**
- Student code (text)
- Reference code (text)
- Requirements text (optional)

**Output:**
- CodeBERT similarity score (0-1, converted to 0-100%)
- Confidence metrics

**Workflow:**
```
Raw Code → Preprocessing → CodeBERT Model → Embeddings → Cosine Similarity → Score
```

---

### 2.3 Token-based Similarity Module

**Fungsi:**
- Simple baseline method
- Extract tokens dari code
- Compare token overlap antara student vs reference

**Components yang perlu dibuat:**
- Code tokenizer (split code into meaningful tokens)
- Token extractor (extract function names, variables, keywords)
- Token comparison engine (Jaccard similarity, overlap percentage)
- Scoring mechanism

**Input:**
- Student code (text)
- Reference code (text)

**Output:**
- Token similarity score (0-1, converted to 0-100%)
- Token overlap details (matched tokens, missing tokens, extra tokens)

**Workflow:**
```
Code → Tokenization → Token Extraction → Token Comparison → Jaccard/Overlap Score
```

---

### 2.4 Score Combination & Aggregation

**Fungsi:**
- Combine scores dari kedua metode
- Generate final evaluation score

**Mechanism:**
- Average: (CodeBERT_score + Token_score) / 2
- Weighted: (CodeBERT_score × w1) + (Token_score × w2), dimana w1+w2=1
- Custom logic: bisa disesuaikan

**Output:**
- Final similarity score (0-100%)
- Individual scores dari masing-masing metode
- Breakdown analysis

---

### 2.5 Evaluation & Accuracy Measurement

**Fungsi:**
- Measure seberapa akurat sistem evaluate code
- Compare dengan ground truth (manual evaluation)

**Components yang perlu dibuat:**
- Ground truth dataset (manual grading dari dosen)
- Accuracy metrics calculator (Precision, Recall, F1, MAE, RMSE, dll)
- Comparison visualizer
- Confusion matrix generator

**Metrics yang diukur:**
- Mean Absolute Error (MAE) - rata-rata error dari predictions
- Root Mean Squared Error (RMSE) - error dengan penalti untuk deviasi besar
- Correlation coefficient - seberapa korelasi predicted score vs actual score
- Classification accuracy (jika pakai threshold, e.g., pass/fail)

**Output:**
- Accuracy report per metode
- Accuracy report combined
- Error analysis (kapan sistem salah evaluasi)
- Visualization (scatter plot, confusion matrix, dll)

---

### 2.6 User Interface / Output Report

**Components yang perlu dibuat:**
- Web dashboard (input form untuk URL, generate report)
- Report generator (HTML/PDF output)
- Visualization (chart, graphs, heatmap)
- Logging system (track semua evaluations)

**Report Content:**
- Student info
- Requirements summary
- CodeBERT score + details
- Token similarity score + details
- Final combined score
- Evaluation result (Pass/Fail dengan threshold)
- Recommendations (apa yang perlu diperbaiki)

---

## 3. Data Pipeline

### 3.1 Data Collection Phase
**Aktivitas:**
- Kumpulkan ~20-30 student projects (untuk dataset awal)
- Dokumentasikan requirements setiap project
- Pilih/buat reference solutions
- Manual evaluation oleh dosen (ground truth)

**Output:**
- Dataset dengan 20-30 projects
- Ground truth labels (manual scores/pass-fail)

### 3.2 Data Preprocessing Phase
**Aktivitas:**
- Clone repositories dari GitHub
- Extract code files
- Clean/standardize code format
- Handle different programming languages (jika ada)

**Output:**
- Preprocessed code dataset ready for analysis

### 3.3 Evaluation Phase
**Aktivitas:**
- Run CodeBERT module pada semua projects
- Run Token similarity module pada semua projects
- Generate scores untuk semua projects
- Aggregate results

**Output:**
- Predicted scores dari kedua metode
- Combined final scores

### 3.4 Accuracy Measurement Phase
**Aktivitas:**
- Compare predicted scores vs ground truth
- Calculate accuracy metrics
- Analyze errors dan patterns
- Generate report

**Output:**
- Accuracy report
- Error analysis
- Recommendations untuk improvement

---

## 4. Technical Stack & Tools

### 4.1 Programming & Libraries
- **Language:** Python (recommended)
- **CodeBERT:** transformers library (HuggingFace)
- **Token processing:** NLTK, spaCy, atau custom tokenizer
- **Similarity calculation:** scikit-learn (cosine_similarity)
- **Data processing:** pandas, numpy
- **Web framework:** Flask/FastAPI (untuk dashboard)
- **Visualization:** matplotlib, plotly, seaborn
- **GitHub API:** PyGithub atau requests

### 4.2 Models
- **CodeBERT:** microsoft/codebert-base (dari HuggingFace)
- **Alternative:** CodeT5 (jika butuh lebih powerful)

### 4.3 Infrastructure
- Local machine (untuk development)
- GPU (optional, untuk faster CodeBERT inference)
- Git untuk version control

---

## 5. Implementation Stages

### Stage 1: Setup & Preparation (Week 1-2)
**Tasks:**
- Setup development environment
- Install libraries & models
- Create project structure
- Create sample datasets (3-5 projects)

**Deliverables:**
- Dev environment ready
- Sample data prepared
- Project repository initialized

### Stage 2: CodeBERT Module (Week 2-3)
**Tasks:**
- Implement code preprocessing
- Load CodeBERT model
- Create embedding generator
- Implement similarity calculation
- Test pada sample data

**Deliverables:**
- Working CodeBERT module
- Test results
- Performance metrics (inference time)

### Stage 3: Token Similarity Module (Week 3-4)
**Tasks:**
- Design tokenization strategy
- Implement tokenizer
- Implement token comparison engine
- Test pada sample data
- Compare dengan CodeBERT

**Deliverables:**
- Working Token similarity module
- Comparison analysis CodeBERT vs Token
- Performance metrics

### Stage 4: Integration & Scoring (Week 4-5)
**Tasks:**
- Integrate kedua modules
- Implement score combination logic
- Create aggregation pipeline
- Test end-to-end

**Deliverables:**
- Integrated system
- Combined scoring working
- Test reports

### Stage 5: Evaluation & Accuracy (Week 5-6)
**Tasks:**
- Prepare ground truth dataset (20-30 projects dengan manual evaluation)
- Run full system pada dataset
- Calculate accuracy metrics
- Analyze errors
- Generate accuracy report

**Deliverables:**
- Accuracy metrics
- Error analysis report
- Findings & insights

### Stage 6: Dashboard & Reporting (Week 6-7)
**Tasks:**
- Build simple web dashboard
- Create report generator
- Add visualizations
- Test user interface

**Deliverables:**
- Working dashboard
- Report templates
- User documentation

### Stage 7: Analysis & Writeup (Week 7-8)
**Tasks:**
- Deep analysis pada results
- Identify when system works/fails
- Write research findings
- Create final report/thesis

**Deliverables:**
- Research findings document
- Final thesis/paper
- Code documentation

---

## 6. Key Artifacts yang Harus Dibuat

### Code Modules
1. **github_manager.py** - Handle GitHub cloning, URL parsing
2. **code_preprocessor.py** - Clean & standardize code
3. **codebert_evaluator.py** - CodeBERT embedding & similarity
4. **token_similarity_evaluator.py** - Token-based evaluation
5. **score_combiner.py** - Combine scores dari kedua metode
6. **accuracy_calculator.py** - Calculate accuracy metrics
7. **report_generator.py** - Generate evaluation reports
8. **main.py** - Orchestrate semua module

### Web Components
1. **app.py / main.py** - Flask/FastAPI application
2. **templates/index.html** - Input form
3. **templates/result.html** - Result display
4. **static/style.css** - Styling
5. **static/script.js** - Frontend logic

### Documentation
1. **README.md** - Project overview & setup
2. **METHODOLOGY.md** - Detailed methodology
3. **RESULTS.md** - Findings & analysis
4. **API.md** - API documentation (jika ada)

### Data Files
1. **dataset.csv** - Ground truth dataset
2. **results.csv** - Evaluation results
3. **accuracy_report.json** - Accuracy metrics

---

## 7. Expected Outputs & Metrics

### System Outputs
- Similarity score (0-100%) per project
- Individual scores dari CodeBERT & Token similarity
- Evaluation status (Pass/Fail based on threshold)
- Detailed report dengan analysis

### Research Outputs
- Accuracy of CodeBERT method
- Accuracy of Token similarity method
- Comparison antara kedua metode
- Optimal combination strategy
- Error patterns & insights
- Recommendations untuk improvement

### Metrics to Report
- **CodeBERT Accuracy:** MAE, RMSE, correlation
- **Token Similarity Accuracy:** MAE, RMSE, correlation
- **Combined Method Accuracy:** MAE, RMSE, correlation
- **Inference Time:** CodeBERT speed, Token method speed
- **Resource Usage:** Memory, CPU requirements
- **Robustness:** Performance pada different code styles, lengths, languages

---

## 8. Success Criteria

### System must:
1. ✅ Successfully load & parse GitHub repositories
2. ✅ Run CodeBERT embedding generation
3. ✅ Run Token similarity calculation
4. ✅ Produce combined similarity scores
5. ✅ Generate evaluation reports
6. ✅ Calculate accuracy metrics

### Research must:
1. ✅ Identify which method (CodeBERT or Token) more accurate
2. ✅ Show optimal combination strategy
3. ✅ Provide insights tentang kapan system works/fails
4. ✅ Contribute meaningful findings untuk academic community

---

## 9. Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| CodeBERT slow inference | Delayed processing | Use GPU, optimize batch processing, cache embeddings |
| GitHub API rate limiting | Cannot clone repos | Implement rate limit handler, use pagination |
| Inconsistent code format | Preprocessing issues | Build robust preprocessor, handle edge cases |
| Small ground truth dataset | Accuracy not reliable | Expand dataset size, cross-validation |
| Different programming languages | Tokenizer incompatibility | Handle language-specific tokenization |

---

## 10. Timeline Summary

```
Week 1-2:  Setup & Preparation
Week 2-3:  CodeBERT Module Implementation
Week 3-4:  Token Similarity Module Implementation
Week 4-5:  Integration & Combined Scoring
Week 5-6:  Evaluation & Accuracy Measurement
Week 6-7:  Dashboard & Reporting
Week 7-8:  Analysis & Final Writeup
```

**Total Duration:** ~8 weeks (adjust sesuai availability)

---

## 11. Deliverables Checklist

- [ ] Source code (modules & main application)
- [ ] Dataset (ground truth + results)
- [ ] Accuracy report
- [ ] Web dashboard/interface
- [ ] Documentation (README, methodology, API)
- [ ] Research paper/thesis
- [ ] Presentation slides
- [ ] Video demo (optional)

---

**End of Planning Document**
