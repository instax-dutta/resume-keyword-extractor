import flair
from flair.data import Sentence
from flair.models import SequenceTagger
import re
from collections import Counter
import argparse
import os
import sys
import platform
import PyPDF2
import docx
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Cross-platform package installation
try:
    import pdfplumber  # Better PDF text extraction
except ImportError:
    print("Installing pdfplumber for improved PDF extraction...")
    try:
        # Use sys.executable to ensure we use the correct Python interpreter
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pdfplumber"])
        import pdfplumber
    except Exception as e:
        print(f"Warning: Could not install pdfplumber automatically: {e}")
        print("Please install manually with: pip install pdfplumber")
        print("Continuing with PyPDF2 only...")

class ResumeKeywordExtractor:
    def __init__(self):
        # Load the pre-trained NER model
        print("Loading NER model (this may take a moment)...")
        self.tagger = SequenceTagger.load('flair/ner-english-ontonotes')
        
        # Download NLTK resources if needed
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("Downloading required NLTK resources...")
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
            except Exception as e:
                print(f"Warning: Could not download NLTK resources automatically: {e}")
                print("Please download manually with: python -m nltk.downloader punkt stopwords")
        
        # Define categories of interest for job-related skills
        self.relevant_entity_types = {
            'SKILL': ['WORK_OF_ART', 'PRODUCT'],
            'TECHNOLOGY': ['PRODUCT'],
            'EDUCATION': ['ORG'],
            'EXPERIENCE': ['DATE', 'TIME'],
            'CERTIFICATION': ['WORK_OF_ART', 'LAW'],
            'LANGUAGE': ['LANGUAGE'],
            'LOCATION': ['GPE', 'LOC']
        }
        
        # Common technical skills to look for
        self.tech_skills = [
            'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin',
            'react', 'angular', 'vue', 'node', 'express', 'django', 'flask', 'spring',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'jenkins',
            'sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'nosql', 'redis',
            'machine learning', 'deep learning', 'ai', 'nlp', 'computer vision',
            'html', 'css', 'sass', 'less', 'bootstrap', 'tailwind',
            'git', 'github', 'gitlab', 'bitbucket', 'svn',
            'agile', 'scrum', 'kanban', 'jira', 'confluence',
            'rest', 'graphql', 'soap', 'api', 'microservices'
        ]
        
        # Define education-related keywords
        self.education_keywords = [
            'university', 'college', 'school', 'institute', 'academy', 
            'bachelor', 'master', 'phd', 'degree', 'diploma', 'certification',
            'b.tech', 'm.tech', 'b.sc', 'm.sc', 'mba'
        ]
        
        # Define certification-related keywords
        self.certification_keywords = [
            'certified', 'certificate', 'certification', 'license', 'credential',
            'aws certified', 'microsoft certified', 'google certified', 'oracle certified'
        ]
        
        # Define experience-related keywords
        self.experience_keywords = [
            'experience', 'work', 'job', 'position', 'role', 'career',
            'intern', 'internship', 'employment', 'project'
        ]
        
        # Add non-technical professional skills
        self.professional_skills = [
            # Management skills
            'leadership', 'management', 'team management', 'project management', 'strategic planning',
            'budgeting', 'forecasting', 'resource allocation', 'decision making', 'conflict resolution',
            
            # Communication skills
            'communication', 'presentation', 'public speaking', 'writing', 'reporting',
            'negotiation', 'facilitation', 'interpersonal', 'customer service', 'client relations',
            
            # Business skills
            'sales', 'marketing', 'business development', 'account management', 'customer acquisition',
            'market research', 'competitive analysis', 'strategic planning', 'operations', 'logistics',
            
            # Finance skills
            'accounting', 'financial analysis', 'bookkeeping', 'auditing', 'tax preparation',
            'financial reporting', 'budgeting', 'forecasting', 'risk assessment', 'investment',
            
            # Healthcare skills
            'patient care', 'medical coding', 'clinical', 'diagnosis', 'treatment planning',
            'medical records', 'healthcare management', 'patient assessment', 'vital signs', 'medical terminology',
            
            # Legal skills
            'legal research', 'contract drafting', 'litigation', 'compliance', 'regulatory',
            'legal writing', 'case management', 'legal analysis', 'negotiation', 'mediation',
            
            # Creative skills
            'design', 'creative direction', 'content creation', 'copywriting', 'editing',
            'photography', 'videography', 'illustration', 'animation', 'graphic design',
            
            # Research skills
            'research', 'data analysis', 'statistical analysis', 'qualitative research', 'quantitative research',
            'survey design', 'literature review', 'experimental design', 'hypothesis testing', 'data collection'
        ]
        
    def _split_text_into_chunks(self, text, max_length=5000):
        """Split text into chunks to avoid memory issues with long resumes."""
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += sentence + " "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
        
    def extract_keywords(self, resume_text):
        # Create a Flair sentence
        # For long resumes, we need to process in chunks to avoid memory issues
        chunks = self._split_text_into_chunks(resume_text)
        
        all_entities = []
        for chunk in chunks:
            sentence = Sentence(chunk)
            self.tagger.predict(sentence)
            all_entities.extend(sentence.get_spans('ner'))
        
        # Initialize categorized entities
        categorized_entities = {
            'SKILL': [],
            'TECHNOLOGY': [],
            'EDUCATION': [],
            'EXPERIENCE': [],
            'CERTIFICATION': [],
            'LANGUAGE': [],
            'LOCATION': [],
            'PROFESSIONAL_SKILL': []
        }
        
        # First pass: Extract entities based on NER tags
        for entity in all_entities:  # Fixed: changed 'entities' to 'all_entities'
            entity_text = entity.text
            entity_type = entity.tag
            
            # Basic categorization based on entity type
            for category, types in self.relevant_entity_types.items():
                if entity_type in types:
                    # Don't add to category yet, we'll refine later
                    if category == 'EDUCATION' and any(keyword.lower() in entity_text.lower() for keyword in self.education_keywords):
                        categorized_entities['EDUCATION'].append(entity_text)
                    elif category == 'CERTIFICATION' and any(keyword.lower() in entity_text.lower() for keyword in self.certification_keywords):
                        categorized_entities['CERTIFICATION'].append(entity_text)
                    elif category == 'EXPERIENCE' and any(keyword.lower() in entity_text.lower() for keyword in self.experience_keywords):
                        categorized_entities['EXPERIENCE'].append(entity_text)
                    elif category == 'LOCATION':
                        categorized_entities['LOCATION'].append(entity_text)
                    elif category == 'LANGUAGE':
                        categorized_entities['LANGUAGE'].append(entity_text)
                    else:
                        # For skills and technologies, we'll handle them separately
                        if entity_type in ['WORK_OF_ART', 'PRODUCT']:
                            # Check if it's a technology
                            if any(tech.lower() in entity_text.lower() for tech in self.tech_skills):
                                categorized_entities['TECHNOLOGY'].append(entity_text)
                            else:
                                categorized_entities['SKILL'].append(entity_text)
        
        # Second pass: Extract dates for experience
        date_pattern = r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}\s+(?:to|-)\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}\b'
        dates = re.findall(date_pattern, resume_text)
        categorized_entities['EXPERIENCE'].extend(dates)
        
        # Extract years of experience
        year_pattern = r'\b\d+\+?\s+years?\s+(?:of\s+)?experience\b'
        years_exp = re.findall(year_pattern, resume_text, re.IGNORECASE)
        categorized_entities['EXPERIENCE'].extend(years_exp)
        
        # Extract technical skills using regex pattern matching
        tech_skills_found = []
        for skill in self.tech_skills:
            if re.search(r'\b' + re.escape(skill) + r'\b', resume_text.lower()):
                tech_skills_found.append(skill)
        
        categorized_entities['TECHNOLOGY'].extend(tech_skills_found)
        
        # Extract professional skills
        prof_skills_found = []
        for skill in self.professional_skills:
            if re.search(r'\b' + re.escape(skill) + r'\b', resume_text.lower()):
                prof_skills_found.append(skill)
        
        categorized_entities['PROFESSIONAL_SKILL'].extend(prof_skills_found)
        
        # Extract education institutions
        for edu_keyword in self.education_keywords:
            pattern = r'\b\w+\s+' + re.escape(edu_keyword) + r'\b'
            matches = re.findall(pattern, resume_text, re.IGNORECASE)
            categorized_entities['EDUCATION'].extend(matches)
        
        # Extract certifications
        for cert_keyword in self.certification_keywords:
            pattern = r'\b\w+\s+' + re.escape(cert_keyword) + r'\b'
            matches = re.findall(pattern, resume_text, re.IGNORECASE)
            categorized_entities['CERTIFICATION'].extend(matches)
        
        # Remove duplicates and count occurrences
        for category in categorized_entities:
            # Remove items that appear in multiple categories (prioritize)
            if category == 'SKILL':
                # Remove technologies from skills
                categorized_entities[category] = [item for item in categorized_entities[category] 
                                                if item not in categorized_entities['TECHNOLOGY']]
            
            # Convert to dictionary with counts
            categorized_entities[category] = dict(Counter(categorized_entities[category]))
        
        return categorized_entities
    
    def extract_from_file(self, file_path):
        try:
            # Normalize path for cross-platform compatibility
            file_path = os.path.normpath(file_path)
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.pdf':
                resume_text = self._extract_text_from_pdf(file_path)
            elif file_extension == '.docx':
                resume_text = self._extract_text_from_docx(file_path)
            else:  # Assume it's a text file
                # Handle different encodings for cross-platform compatibility
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        resume_text = file.read()
                except UnicodeDecodeError:
                    # Try with a different encoding if utf-8 fails
                    with open(file_path, 'r', encoding='latin-1') as file:
                        resume_text = file.read()
                    
            return self.extract_keywords(resume_text)
        except Exception as e:
            return {"error": f"Error processing file: {str(e)}"}
    
    def _extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file using pdfplumber for better results."""
        text = ""
        pdf_extraction_error = None
        
        # Try pdfplumber first if available
        if 'pdfplumber' in sys.modules:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text() or ""
                        text += page_text + "\n"
                        
                        # Extract text from tables as well
                        tables = page.extract_tables()
                        for table in tables:
                            for row in table:
                                text += " ".join([cell or "" for cell in row]) + "\n"
                
                if text.strip():  # If we got text, return it
                    return text
            except Exception as e:
                pdf_extraction_error = str(e)
                print(f"pdfplumber extraction failed: {e}")
        
        # Fallback to PyPDF2
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
            
            if not text.strip() and pdf_extraction_error:
                raise Exception(f"Failed to extract text from PDF. Original error: {pdf_extraction_error}")
                
            return text
        except Exception as e:
            raise Exception(f"All PDF extraction methods failed: {str(e)}")

    def _extract_text_from_docx(self, docx_path):
        """Extract text from a DOCX file."""
        doc = docx.Document(docx_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)

def main():
    parser = argparse.ArgumentParser(description='Extract keywords from a resume')
    
    # Create a mutually exclusive group for input methods
    input_group = parser.add_mutually_exclusive_group(required=True)
    
    # Add positional argument for file (optional)
    input_group.add_argument('resume_file', nargs='?', type=str, 
                        help='Path to resume file (txt, pdf, or docx)')
    
    # Keep the existing arguments as alternatives
    input_group.add_argument('--file', type=str, help='Path to resume file (txt, pdf, or docx)')
    input_group.add_argument('--text', type=str, help='Resume text content')
    
    # Add output format option
    parser.add_argument('--format', choices=['text', 'json'], default='text',
                        help='Output format (text or json)')
    
    # Add option to save results to file
    parser.add_argument('--save', action='store_true', help='Save results to a text file')
    
    args = parser.parse_args()
    
    # Print system information for debugging
    print(f"Operating System: {platform.system()} {platform.release()}")
    
    extractor = ResumeKeywordExtractor()
    
    # Determine which input method to use
    input_file_path = None
    if args.resume_file:
        results = extractor.extract_from_file(args.resume_file)
        input_file_path = args.resume_file
    elif args.file:
        results = extractor.extract_from_file(args.file)
        input_file_path = args.file
    elif args.text:
        results = extractor.extract_keywords(args.text)
    
    # Print results in the specified format
    if args.format == 'json':
        print_json_results(results)
    else:
        print_results(results)
    
    # Save results to file if requested
    if args.save and input_file_path and "error" not in results:  # Added error check
        # Create output filename based on input filename
        base_name = os.path.splitext(input_file_path)[0]
        output_file_path = f"{base_name}_keywords.txt"
        
        # Save the results
        save_results_to_file(results, output_file_path, format=args.format)
        print(f"\nResults saved to: {output_file_path}")

def save_results_to_file(results, file_path, format='text'):
    """Save extraction results to a file."""
    # Normalize path for cross-platform compatibility
    file_path = os.path.normpath(file_path)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            if format == 'json':
                import json
                f.write(json.dumps(results, indent=2))
            else:
                f.write("===== RESUME KEYWORD EXTRACTION RESULTS =====\n\n")
                
                for category, items in results.items():
                    if items:
                        f.write(f"\n{category}:\n")
                        for item, count in items.items():
                            f.write(f"  - {item} ({count})\n")
                
                # Write summary
                total_keywords = sum(len(items) for items in results.values())
                f.write(f"\nTotal unique keywords extracted: {total_keywords}\n")
    except Exception as e:
        print(f"Error saving results to file: {str(e)}")
        # Try with a different encoding if utf-8 fails
        try:
            with open(file_path, 'w', encoding='latin-1') as f:
                f.write("Error saving with UTF-8. Using Latin-1 encoding.\n\n")
                # ... same writing logic as above ...
        except Exception as e2:
            print(f"Failed to save results with alternative encoding: {str(e2)}")

def main():
    parser = argparse.ArgumentParser(description='Extract keywords from a resume')
    
    # Create a mutually exclusive group for input methods
    input_group = parser.add_mutually_exclusive_group(required=True)
    
    # Add positional argument for file (optional)
    input_group.add_argument('resume_file', nargs='?', type=str, 
                        help='Path to resume file (txt, pdf, or docx)')
    
    # Keep the existing arguments as alternatives
    input_group.add_argument('--file', type=str, help='Path to resume file (txt, pdf, or docx)')
    input_group.add_argument('--text', type=str, help='Resume text content')
    
    # Add output format option
    parser.add_argument('--format', choices=['text', 'json'], default='text',
                        help='Output format (text or json)')
    
    # Add option to save results to file
    parser.add_argument('--save', action='store_true', help='Save results to a text file')
    
    args = parser.parse_args()
    
    # Print system information for debugging
    print(f"Operating System: {platform.system()} {platform.release()}")
    
    extractor = ResumeKeywordExtractor()
    
    # Determine which input method to use
    input_file_path = None
    if args.resume_file:
        results = extractor.extract_from_file(args.resume_file)
        input_file_path = args.resume_file
    elif args.file:
        results = extractor.extract_from_file(args.file)
        input_file_path = args.file
    elif args.text:
        results = extractor.extract_keywords(args.text)
    
    # Print results in the specified format
    if args.format == 'json':
        print_json_results(results)
    else:
        print_results(results)
    
    # Save results to file if requested
    if args.save and input_file_path and "error" not in results:
        # Create output filename based on input filename
        base_name = os.path.splitext(input_file_path)[0]
        output_file_path = f"{base_name}_keywords.txt"
        
        # Save the results
        save_results_to_file(results, output_file_path, format=args.format)
        print(f"\nResults saved to: {output_file_path}")

# Add a new function to print JSON results
def print_json_results(results):
    import json
    print(json.dumps(results, indent=2))

def print_results(results):
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    print("\n===== RESUME KEYWORD EXTRACTION RESULTS =====\n")
    
    for category, items in results.items():
        if items:
            print(f"\n{category}:")
            for item, count in items.items():
                print(f"  - {item} ({count})")
    
    # Print summary
    total_keywords = sum(len(items) for items in results.values())
    print(f"\nTotal unique keywords extracted: {total_keywords}")

if __name__ == "__main__":
    main()