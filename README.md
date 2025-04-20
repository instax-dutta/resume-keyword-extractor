# Resume Keyword Extractor

## Overview
This tool extracts and categorizes keywords from resumes to help with job matching and applicant tracking. It uses natural language processing to identify:

- **Skills**: Technical and professional competencies
- **Technologies**: Programming languages, frameworks, tools
- **Education**: Degrees, institutions, graduation years
- **Experience**: Job titles, companies, durations
- **Certifications**: Professional certifications and licenses
- **Languages**: Languages spoken and proficiency levels
- **Locations**: Geographic locations mentioned

The extracted data helps recruiters and hiring managers quickly assess candidate qualifications and match them to job requirements.

## Installation

### Prerequisites
- Python 3.6+ (recommended 3.8+ for best compatibility)
- pip package manager (latest version)
- 200MB disk space for NLP models and dependencies

### Step-by-Step Setup
1. **Clone the repository** (if applicable):
   ```
   git clone https://github.com/instax-dutta/resume-keyword-extractor.git
   cd resume-keyword-extractor
   ```

2. **Install required packages**:
   ```
   pip install -r requirements.txt
   ```
   Or install individually:
   ```
   pip install flair==0.11.3 nltk==3.6.7 PyPDF2==1.27.0 python-docx==0.8.11 pdfplumber==0.7.5
   ```

3. **Download NLTK data** (run once):
   ```
   python -m nltk.downloader punkt stopwords
   ```

4. **Verify installation**:
   ```
   python -c "import flair; import nltk; import PyPDF2; import pdfplumber; print('All dependencies installed successfully')"
   ```

## Usage

### Basic Command
```
python resume_keyword_extractor.py [RESUME_FILE]
```

### Command Line Options
| Option | Description | Example |
|--------|-------------|---------|
| `--file` | Path to resume file | `--file resume.pdf` |
| `--text` | Process raw text input | `--text "John Doe, Python developer..."` |
| `--format` | Output format (text/json) | `--format json` |
| `--save` | Save results to file | `--save output.txt` |
| `--verbose` | Show detailed processing info | `--verbose` |

### Examples
1. **Basic PDF processing**:
   ```
   python resume_keyword_extractor.py resume.pdf
   ```
   Outputs categorized keywords to console

2. **Save as JSON file**:
   ```
   python resume_keyword_extractor.py resume.pdf --format json --save results.json
   ```
   Creates a JSON file with structured output

3. **Process multiple files**:
   ```
   for file in *.pdf; do python resume_keyword_extractor.py "$file" --save "${file%.pdf}_keywords.txt"; done
   ```
   Processes all PDFs in current directory

## Module Explanations

### Core Modules
- **flair**: State-of-the-art NLP library for named entity recognition (NER)
  - Uses pre-trained models to identify entities in text
  - Handles context-aware entity classification

- **nltk**: Natural Language Toolkit for text processing
  - Tokenization and sentence splitting
  - Stopword removal and basic text cleaning
  - Provides linguistic resources

- **PyPDF2/pdfplumber**: PDF text extraction
  - PyPDF2 for basic PDF text extraction
  - pdfplumber as fallback for complex PDF layouts
  - Handles encrypted and scanned PDFs (OCR not included)

- **python-docx**: Microsoft Word document processing
  - Extracts text from .docx files
  - Preserves formatting and structure

### Key Functions
1. **`extract_keywords()`**
   - Main processing pipeline
   - Coordinates text extraction and analysis
   - Returns categorized keyword dictionary

2. **`_extract_text_from_pdf()`**
   - Attempts text extraction with PyPDF2 first
   - Falls back to pdfplumber if PyPDF2 fails
   - Handles common PDF extraction issues

3. **`_extract_text_from_docx()`**
   - Processes Word documents paragraph by paragraph
   - Preserves bullet points and numbered lists
   - Handles embedded tables and text boxes

## Deploying as an API on Vercel

This project can be deployed as a serverless API on [Vercel](https://vercel.com/) using the `/api/extract.py` endpoint.

### Usage
- **Endpoint:** `/api/extract`
- **Method:** `POST`
- **Content-Type:** `multipart/form-data`
- **Form field:** `file` (PDF or DOCX)
- **Response:** JSON with extracted keywords (customize logic as needed)

### Example (using curl)
```sh
curl -X POST https://<your-vercel-deployment-url>/api/extract \
  -F "file=@/path/to/resume.pdf"
```

### Vercel Deployment Steps
1. **Push this repo to GitHub/GitLab.**
2. **Connect your repo to Vercel.**
3. **Set Python as the runtime for `/api/extract.py`.**
4. **Ensure `requirements.txt` is present in the root.**
5. **Deploy!**

---

**Note:** The `/api/extract.py` API uses a minimal keyword extraction for demonstration. Replace the logic with your full `ResumeKeywordExtractor` class for production use.

## Troubleshooting

### Common Issues and Solutions

**1. PDF Extraction Problems**
- Symptom: Empty or garbled text output
- Solutions:
  - Install pdfplumber manually: `pip install pdfplumber`
  - Try a different PDF library: `pip install pdfminer.six`
  - For scanned PDFs: Use OCR software first

**2. NLTK Data Download Issues**
- Symptom: "Resource punkt not found" error
- Solutions:
  - Run download manually: `python -m nltk.downloader punkt stopwords`
  - Set NLTK_DATA environment variable if behind proxy
  - Download data manually from nltk.org

**3. File Access Problems**
- Symptom: "File not found" errors
- Solutions:
  - Use absolute paths instead of relative paths
  - Check file permissions
  - Verify file isn't open in another program

**4. Memory Errors**
- Symptom: "MemoryError" or crashes
- Solutions:
  - Process smaller files
  - Increase system memory
  - Use 64-bit Python

**5. Dependency Conflicts**
- Symptom: Import errors
- Solutions:
  - Create virtual environment
  - Check installed versions with `pip freeze`
  - Reinstall exact versions from requirements.txt

## Output Format

The tool categorizes keywords into structured output with the following fields:

### Text Output Example
```
SKILL: Python, Machine Learning, Data Analysis
TECHNOLOGY: TensorFlow, PyTorch, Scikit-learn
EDUCATION: MS Computer Science (Stanford University 2020)
EXPERIENCE: Senior Data Scientist (Acme Corp 2018-2022)
CERTIFICATION: AWS Certified Solutions Architect
LANGUAGE: English (Fluent), Spanish (Intermediate)
LOCATION: San Francisco, CA
```

### JSON Output Structure
```json
{
  "SKILL": ["Python", "Machine Learning", "Data Analysis"],
  "TECHNOLOGY": ["TensorFlow", "PyTorch", "Scikit-learn"],
  "EDUCATION": ["MS Computer Science (Stanford University 2020)"],
  "EXPERIENCE": ["Senior Data Scientist (Acme Corp 2018-2022)"],
  "CERTIFICATION": ["AWS Certified Solutions Architect"],
  "LANGUAGE": ["English (Fluent)", "Spanish (Intermediate)"],
  "LOCATION": ["San Francisco, CA"]
}
```

### Field Descriptions
- **SKILL**: Demonstrated abilities and competencies
- **TECHNOLOGY**: Specific tools, languages, frameworks
- **EDUCATION**: Degrees, institutions, graduation years
- **EXPERIENCE**: Job titles, companies, employment periods
- **CERTIFICATION**: Professional certifications/licenses
- **LANGUAGE**: Languages and proficiency levels
- **LOCATION**: Cities, states, countries mentioned