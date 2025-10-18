import io
import os
import logging
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor
import PyPDF2
import fitz  # PyMuPDF
import pytesseract
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from font_management import get_language_mapper

logger = logging.getLogger(__name__)

@dataclass
class PdfTextElement:
    text: str
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    font_name: str
    font_size: float
    color: Tuple[float, float, float] # RGB (0-1)
    is_bold: bool = False
    is_italic: bool = False

@dataclass
class PdfLayoutData:
    page_number: int
    width: float
    height: float
    text_elements: List[PdfTextElement] = field(default_factory=list)

# Model/tokenizer/ip/DEVICE globals
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = None
model = None
ip = None

def initialize_translation_core():
    global tokenizer, model, ip, DEVICE
    model_name = "ai4bharat/indictrans2-en-indic-dist-200M"
    cache_dir = os.path.expanduser("~/.cache/huggingface")
    os.makedirs(cache_dir, exist_ok=True)
    logger.info(f"Using cache directory: {cache_dir}")
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=cache_dir,
        local_files_only=False
    )
    logger.info("✅ Tokenizer loaded")
    logger.info("Loading model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        cache_dir=cache_dir,
        local_files_only=False,
        low_cpu_mem_usage=True
    ).to(DEVICE)
    model.eval()
    logger.info("✅ Model loaded")
    logger.info("Loading IndicProcessor...")
    ip = IndicProcessor(inference=True)
    logger.info("✅ IndicProcessor loaded")

def _process_pymupdf_layout(page_dict: Dict[str, Any], page_number: int) -> PdfLayoutData:
    page_layout = PdfLayoutData(page_number=page_number, width=page_dict['width'], height=page_dict['height'])
    for block in page_dict['blocks']:
        if block['type'] == 0:  # Text block
            for line in block['lines']:
                for span in line['spans']:
                    font_name = span['font']
                    font_size = span['size']
                    color_rgb = fitz.utils.sRGB_to_rgb(span['color'])
                    is_bold = "bold" in font_name.lower()
                    is_italic = "italic" in font_name.lower()
                    page_layout.text_elements.append(
                        PdfTextElement(
                            text=span['text'],
                            bbox=tuple(span['bbox']),
                            font_name=font_name,
                            font_size=font_size,
                            color=color_rgb,
                            is_bold=is_bold,
                            is_italic=is_italic
                        )
                    )
    return page_layout

def extract_text_from_pdf(pdf_content: bytes, max_pages: int = 2) -> tuple[str, str, List[PdfLayoutData]]:
    extraction_method = "Unknown"
    extracted_text = ""
    all_layout_data: List[PdfLayoutData] = []
    try:
        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        max_pages_to_process = min(len(pdf_reader.pages), max_pages)
        for page_num in range(max_pages_to_process):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            if page_text.strip():
                extracted_text += page_text + "\n"
        if extracted_text.strip():
            extraction_method = "PyPDF2 (text-based)"
            logger.info(f"Extracted text using {extraction_method}")
            return extracted_text.strip(), extraction_method, []
    except Exception as e:
        logger.warning(f"PyPDF2 extraction failed: {e}")
    try:
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        max_pages_to_process = min(len(pdf_document), max_pages)
        current_page_text = ""
        for page_num in range(max_pages_to_process):
            page = pdf_document[page_num]
            page_dict = page.get_text("dict")
            page_layout = _process_pymupdf_layout(page_dict, page_num)
            all_layout_data.append(page_layout)
            page_text = page.get_text()
            if page_text.strip():
                current_page_text += page_text + "\n"
        pdf_document.close()
        if current_page_text.strip():
            extracted_text = current_page_text.strip()
            extraction_method = "PyMuPDF (layout-rich text)"
            logger.info(f"Extracted text using {extraction_method}")
            return extracted_text, extraction_method, all_layout_data
    except Exception as e:
        logger.warning(f"PyMuPDF layout extraction failed: {e}")
        all_layout_data = []
        try:
            pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
            current_page_text = ""
            max_pages_to_process = min(len(pdf_document), max_pages)
            for page_num in range(max_pages_to_process):
                page = pdf_document[page_num]
                page_text = page.get_text()
                if page_text.strip():
                    current_page_text += page_text + "\n"
            pdf_document.close()
            if current_page_text.strip():
                extracted_text = current_page_text.strip()
                extraction_method = "PyMuPDF (simple text)"
                logger.info(f"Extracted text using {extraction_method} after layout extraction failed.")
                return extracted_text, extraction_method, []
        except Exception as e_fallback:
            logger.warning(f"PyMuPDF simple text extraction fallback also failed: {e_fallback}")
    try:
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        max_pages_to_process = min(len(pdf_document), max_pages)
        current_page_text = ""
        for page_num in range(max_pages_to_process):
            page = pdf_document[page_num]
            mat = fitz.Matrix(1.5, 1.5)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            from PIL import Image
            image = Image.open(io.BytesIO(img_data))
            max_dimension = 2000
            if max(image.size) > max_dimension:
                ratio = max_dimension / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            page_text = pytesseract.image_to_string(image, lang='eng')
            if page_text.strip():
                current_page_text += page_text + "\n"
            del image
            del pix
        pdf_document.close()
        if current_page_text.strip():
            extracted_text = current_page_text.strip()
            extraction_method = "OCR (scanned PDF)"
            logger.info(f"Extracted text using {extraction_method}")
            return extracted_text, extraction_method, []
    except Exception as e:
        logger.warning(f"OCR extraction failed: {e}")
    raise ValueError("Could not extract text from PDF. The file might be corrupted, password-protected, or contain only images without readable text.")

def create_pdf_from_text(original_text_chunks: List[str], translated_text_chunks: List[str], filename: str, target_language: str, language_code: str = None, layout_data: List[PdfLayoutData] = None) -> io.BytesIO:
    """
    Create a PDF with original and translated text using appropriate fonts for the target language and preserving layout.
    This function is implemented in main.py to avoid circular imports.
    """
    raise NotImplementedError("This function should be imported from main.py to avoid circular imports")

def chunk_text(text: str, max_chunk_size: int = 200) -> List[str]:
    """Smaller chunk text chunking for faster translation performance"""
    # Use smaller sentence-based chunking for better speed
    sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if s.strip()]
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # Use smaller chunks for faster processing
        test_chunk = current_chunk + ". " + sentence if current_chunk else sentence
        if len(test_chunk) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk = test_chunk
    
    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # If chunks are still too large, split further by phrases
    if any(len(chunk) > max_chunk_size for chunk in chunks):
        smaller_chunks = []
        for chunk in chunks:
            if len(chunk) <= max_chunk_size:
                smaller_chunks.append(chunk)
            else:
                # Split by commas and semicolons for smaller phrases
                phrases = [p.strip() for p in chunk.replace(';', ',').split(',') if p.strip()]
                current_phrase_chunk = ""
                for phrase in phrases:
                    test_phrase = current_phrase_chunk + ", " + phrase if current_phrase_chunk else phrase
                    if len(test_phrase) > max_chunk_size and current_phrase_chunk:
                        smaller_chunks.append(current_phrase_chunk.strip())
                        current_phrase_chunk = phrase
                    else:
                        current_phrase_chunk = test_phrase
                if current_phrase_chunk:
                    smaller_chunks.append(current_phrase_chunk.strip())
        chunks = smaller_chunks
    
    # Final fallback to word-based chunking if needed
    if not chunks and text:
        words = text.split()
        current_chunk = ""
        for word in words:
            if len(current_chunk) + len(word) + 1 > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = word
            else:
                current_chunk += " " + word if current_chunk else word
        if current_chunk:
            chunks.append(current_chunk.strip())
    
    return chunks if chunks else [text]

def remove_duplicates_from_text(text: str) -> str:
    lines = text.split('\n')
    unique_lines = []
    seen_lines = set()
    for line in lines:
        clean_line = line.strip().lower()
        if clean_line and clean_line not in seen_lines and len(clean_line) > 3:
            unique_lines.append(line.strip())
            seen_lines.add(clean_line)
    sentences = '. '.join(unique_lines).split('.')
    unique_sentences = []
    seen_sentences = set()
    for sentence in sentences:
        clean_sentence = sentence.strip().lower()
        if clean_sentence and clean_sentence not in seen_sentences and len(clean_sentence) > 5:
            unique_sentences.append(sentence.strip())
            seen_sentences.add(clean_sentence)
    return '. '.join(unique_sentences)
