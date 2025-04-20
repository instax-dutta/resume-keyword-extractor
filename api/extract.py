import os
import sys
import tempfile
import json
from typing import Any

from flair.data import Sentence
from flair.models import SequenceTagger
import nltk
import PyPDF2
import docx

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

# Import ResumeKeywordExtractor from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from resume_keyword_extractor import ResumeKeywordExtractor

# --- Helper functions (adapted from ResumeKeywordExtractor) ---
def extract_text_from_pdf(file_path):
    try:
        if pdfplumber:
            with pdfplumber.open(file_path) as pdf:
                return "\n".join(page.extract_text() or '' for page in pdf.pages)
        else:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                return "\n".join(page.extract_text() or '' for page in reader.pages)
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")

def extract_text_from_docx(file_path):
    try:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        raise Exception(f"Error extracting text from DOCX: {str(e)}")

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_type = self.headers.get('Content-Type')
            if not content_type or 'multipart/form-data' not in content_type:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Content-Type must be multipart/form-data"}).encode('utf-8'))
                return
            # Parse multipart form data
            import cgi
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={'REQUEST_METHOD': 'POST', 'CONTENT_TYPE': content_type}
            )
            fileitem = form['file'] if 'file' in form else None
            if not fileitem or not fileitem.filename:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "No file uploaded."}).encode('utf-8'))
                return
            # Save uploaded file to temp
            ext = os.path.splitext(fileitem.filename)[1].lower()
            if ext not in ['.pdf', '.docx']:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Unsupported file type. Only PDF and DOCX are supported."}).encode('utf-8'))
                return
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(fileitem.file.read())
                tmp_path = tmp.name
            try:
                if ext == '.pdf':
                    text = extract_text_from_pdf(tmp_path)
                elif ext == '.docx':
                    text = extract_text_from_docx(tmp_path)
                # Use full ResumeKeywordExtractor
                extractor = ResumeKeywordExtractor()
                results = extractor.extract_keywords(text)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(results).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Processing error: {str(e)}"}).encode('utf-8'))
            finally:
                os.remove(tmp_path)
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"Server error: {str(e)}"}).encode('utf-8'))

def handler(request, context):
    try:
        if request.method == 'POST':
            from io import BytesIO
            import cgi
            content_type = request.headers.get('content-type')
            if not content_type or 'multipart/form-data' not in content_type:
                return (400, {'Content-Type': 'application/json'}, json.dumps({"error": "Content-Type must be multipart/form-data"}).encode('utf-8'))
            form = cgi.FieldStorage(
                fp=BytesIO(request.body),
                headers=request.headers,
                environ={'REQUEST_METHOD': 'POST', 'CONTENT_TYPE': content_type}
            )
            fileitem = form['file'] if 'file' in form else None
            if not fileitem or not fileitem.filename:
                return (400, {'Content-Type': 'application/json'}, json.dumps({"error": "No file uploaded."}).encode('utf-8'))
            ext = os.path.splitext(fileitem.filename)[1].lower()
            if ext not in ['.pdf', '.docx']:
                return (400, {'Content-Type': 'application/json'}, json.dumps({"error": "Unsupported file type. Only PDF and DOCX are supported."}).encode('utf-8'))
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(fileitem.file.read())
                tmp_path = tmp.name
            try:
                if ext == '.pdf':
                    text = extract_text_from_pdf(tmp_path)
                elif ext == '.docx':
                    text = extract_text_from_docx(tmp_path)
                extractor = ResumeKeywordExtractor()
                results = extractor.extract_keywords(text)
                return (200, {'Content-Type': 'application/json'}, json.dumps(results).encode('utf-8'))
            except Exception as e:
                return (500, {'Content-Type': 'application/json'}, json.dumps({"error": f"Processing error: {str(e)}"}).encode('utf-8'))
            finally:
                os.remove(tmp_path)
        else:
            return (405, {'Content-Type': 'application/json'}, json.dumps({"error": "Method Not Allowed"}).encode('utf-8'))
    except Exception as e:
        return (500, {'Content-Type': 'application/json'}, json.dumps({"error": f"Server error: {str(e)}"}).encode('utf-8'))
