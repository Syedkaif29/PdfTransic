from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import torch
import logging
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor
import PyPDF2
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import os
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from fastapi.responses import StreamingResponse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="IndicTrans2 Translation API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for Hugging Face Spaces
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global holders
tokenizer = None
model = None
ip = None
DEVICE = None


class TranslationRequest(BaseModel):
    sentences: List[str]
    src_lang: str
    tgt_lang: str

class SimpleTranslationRequest(BaseModel):
    text: str
    target_language: str
    source_language: Optional[str] = "eng_Latn"


@app.get("/")
async def root():
    return {
        "message": "IndicTrans2 Translation API",
        "status": "running",
        "endpoints": ["/health", "/translate", "/docs"]
    }


@app.get("/health")
async def health_check():
    global tokenizer, model, ip, DEVICE
    models_loaded = all([tokenizer, model, ip])
    return {
        "status": "healthy" if models_loaded else "loading",
        "device": str(DEVICE) if DEVICE else "unknown",
        "model": "indictrans2-en-indic-dist-200M",
        "components_loaded": {
            "tokenizer": tokenizer is not None,
            "model": model is not None,
            "processor": ip is not None
        }
    }


@app.on_event("startup")
async def startup_event():
    global tokenizer, model, ip, DEVICE

    try:
        logger.info("🚀 Starting model loading...")
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {DEVICE}")

        # Set memory management for CUDA
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            # Set memory fraction to avoid OOM
            torch.cuda.set_per_process_memory_fraction(0.8)

        model_name = "ai4bharat/indictrans2-en-indic-dist-200M"

        # Set cache directory explicitly
        import os
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
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,  # Use float16 for GPU to save memory
            cache_dir=cache_dir,
            local_files_only=False,
            low_cpu_mem_usage=True  # Enable low memory usage
        ).to(DEVICE)
        
        # Set model to eval mode for inference
        model.eval()
        logger.info("✅ Model loaded")

        logger.info("Loading IndicProcessor...")
        ip = IndicProcessor(inference=True)
        logger.info("✅ IndicProcessor loaded")

        # Clear any remaining cache
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        logger.info("🎉 All components loaded successfully!")

    except Exception as e:
        logger.error(f"❌ Failed to load components: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        # App stays alive, health endpoint will show "loading"


@app.post("/translate")
def translate(request: TranslationRequest):
    try:
        # Step 1: Preprocess
        batch = ip.preprocess_batch(
            request.sentences,
            src_lang=request.src_lang,
            tgt_lang=request.tgt_lang,
        )

        # Step 2: Tokenize
        inputs = tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)

        # Step 3: Generate (⚡ FIXED with memory management)
        with torch.no_grad():
            generated_tokens = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,  # avoids "0 layers" cache bug
                min_length=0,
                max_length=256,
                num_beams=3,  # Reduced from 5 to save memory
                num_return_sequences=1,
                do_sample=False,
            )

        # Step 4: Decode
        decoded = tokenizer.batch_decode(
            generated_tokens.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # Step 5: Postprocess
        translations = ip.postprocess_batch(decoded, lang=request.tgt_lang)

        return {
            "translations": translations,
            "source_language": request.src_lang,
            "target_language": request.tgt_lang,
            "input_sentences": request.sentences
        }

    except Exception as e:
        logger.error(f"❌ Translation error: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


@app.post("/translate-simple")
def translate_simple(request: SimpleTranslationRequest):
    """Simple translation endpoint for single text input"""
    global tokenizer, model, ip, DEVICE
    
    if not all([tokenizer, model, ip]):
        raise HTTPException(status_code=503, detail="Models are still loading. Please try again in a moment.")
    
    try:
        # Convert single text to list format
        sentences = [request.text]
        
        # Step 1: Preprocess
        batch = ip.preprocess_batch(
            sentences,
            src_lang=request.source_language,
            tgt_lang=request.target_language,
        )

        # Step 2: Tokenize
        inputs = tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)

        # Step 3: Generate (with memory management)
        with torch.no_grad():
            generated_tokens = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
                min_length=0,
                max_length=256,
                num_beams=3,  # Reduced from 5 to save memory
                num_return_sequences=1,
                do_sample=False,
            )

        # Step 4: Decode
        decoded = tokenizer.batch_decode(
            generated_tokens.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # Step 5: Postprocess
        translations = ip.postprocess_batch(decoded, lang=request.target_language)

        return {
            "translated_text": translations[0] if translations else "",
            "original_text": request.text,
            "source_language": request.source_language,
            "target_language": request.target_language,
            "success": True
        }

    except Exception as e:
        logger.error(f"❌ Simple translation error: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


def extract_text_from_pdf(pdf_content: bytes, max_pages: int = 2) -> tuple[str, str]:
    """
    Enhanced PDF text extraction with OCR fallback and memory management
    Returns: (extracted_text, extraction_method)
    """
    extraction_method = "Unknown"
    extracted_text = ""
    
    try:
        # Method 1: Try PyPDF2 first (fastest for text-based PDFs)
        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        # Limit pages
        max_pages = min(len(pdf_reader.pages), max_pages)
        
        for page_num in range(max_pages):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            if page_text.strip():
                extracted_text += page_text + "\n"
        
        if extracted_text.strip():
            extraction_method = "PyPDF2 (text-based)"
            return extracted_text.strip(), extraction_method
            
    except Exception as e:
        logger.warning(f"PyPDF2 extraction failed: {e}")
    
    try:
        # Method 2: Try PyMuPDF (better for complex PDFs)
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        
        max_pages = min(len(pdf_document), max_pages)
        
        for page_num in range(max_pages):
            page = pdf_document[page_num]
            page_text = page.get_text()
            if page_text.strip():
                extracted_text += page_text + "\n"
        
        pdf_document.close()
        
        if extracted_text.strip():
            extraction_method = "PyMuPDF (advanced text)"
            return extracted_text.strip(), extraction_method
            
    except Exception as e:
        logger.warning(f"PyMuPDF extraction failed: {e}")
    
    try:
        # Method 3: OCR fallback for scanned PDFs (with memory management)
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        
        max_pages = min(len(pdf_document), max_pages)
        
        for page_num in range(max_pages):
            page = pdf_document[page_num]
            
            # Convert page to image with lower resolution to save memory
            mat = fitz.Matrix(1.5, 1.5)  # Reduced from 2.0 to save memory
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(img_data))
            
            # Resize image if too large (memory management)
            max_dimension = 2000
            if max(image.size) > max_dimension:
                ratio = max_dimension / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Perform OCR
            page_text = pytesseract.image_to_string(image, lang='eng')
            if page_text.strip():
                extracted_text += page_text + "\n"
            
            # Clean up memory
            del image
            del pix
        
        pdf_document.close()
        
        if extracted_text.strip():
            extraction_method = "OCR (scanned PDF)"
            return extracted_text.strip(), extraction_method
            
    except Exception as e:
        logger.warning(f"OCR extraction failed: {e}")
    
    # If all methods fail
    raise ValueError("Could not extract text from PDF. The file might be corrupted, password-protected, or contain only images without readable text.")


def chunk_text(text: str, max_chunk_size: int = 500) -> List[str]:
    """
    Split text into smaller chunks for memory-efficient processing
    """
    # Split by sentences first
    sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if s.strip()]
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would exceed the limit, start a new chunk
        if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += ". " + sentence if current_chunk else sentence
    
    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # If no sentences found, split by words
    if not chunks and text:
        words = text.split()
        current_chunk = ""
        for word in words:
            if len(current_chunk) + len(word) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = word
            else:
                current_chunk += " " + word if current_chunk else word
        if current_chunk:
            chunks.append(current_chunk.strip())
    
    return chunks if chunks else [text]


@app.post("/translate-pdf")
async def translate_pdf(
    file: UploadFile = File(...),
    target_language: str = Form(...)
):
    """Extract text from PDF and translate it with enhanced extraction methods and memory management"""
    global tokenizer, model, ip, DEVICE
    
    if not all([tokenizer, model, ip]):
        raise HTTPException(status_code=503, detail="Models are still loading. Please try again in a moment.")
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Read PDF content
        pdf_content = await file.read()
        
        # Enhanced text extraction with multiple methods
        extracted_text, extraction_method = extract_text_from_pdf(pdf_content, max_pages=2)
        
        logger.info(f"Text extracted using: {extraction_method}")
        
        # Clean up the extracted text
        extracted_text = extracted_text.strip()
        
        # Check text length and apply chunking if needed
        if len(extracted_text) > 1000:  # If text is too long, use chunking
            logger.info(f"Text length: {len(extracted_text)} characters. Using chunking for memory efficiency.")
            text_chunks = chunk_text(extracted_text, max_chunk_size=500)
            logger.info(f"Split into {len(text_chunks)} chunks")
        else:
            # For shorter texts, split by sentences as before
            text_chunks = [sent.strip() for sent in extracted_text.split('.') if sent.strip()]
            if not text_chunks:
                text_chunks = [extracted_text]
        
        # Translate chunks in batches to manage memory
        all_translations = []
        batch_size = 3  # Process 3 chunks at a time to avoid memory issues
        
        for i in range(0, len(text_chunks), batch_size):
            batch_chunks = text_chunks[i:i + batch_size]
            
            try:
                # Preprocess batch
                batch = ip.preprocess_batch(
                    batch_chunks,
                    src_lang="eng_Latn",
                    tgt_lang=target_language,
                )

                # Tokenize with memory management
                inputs = tokenizer(
                    batch,
                    truncation=True,
                    padding="longest",
                    return_tensors="pt",
                    return_attention_mask=True,
                    max_length=512,  # Limit input length
                ).to(DEVICE)

                # Generate translations
                with torch.no_grad():
                    generated_tokens = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        use_cache=False,
                        min_length=0,
                        max_length=256,
                        num_beams=3,  # Reduced from 5 to save memory
                        num_return_sequences=1,
                        do_sample=False,
                    )

                # Decode
                decoded = tokenizer.batch_decode(
                    generated_tokens.detach().cpu().tolist(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )

                # Postprocess
                batch_translations = ip.postprocess_batch(decoded, lang=target_language)
                all_translations.extend(batch_translations)
                
                # Clear GPU memory after each batch
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(text_chunks) + batch_size - 1)//batch_size}")
                
            except Exception as batch_error:
                logger.error(f"Error processing batch {i//batch_size + 1}: {batch_error}")
                # Add placeholder for failed batch
                all_translations.extend(["[Translation failed for this section]"] * len(batch_chunks))
        
        # Join translated chunks back together
        if len(extracted_text) > 1000:
            # For chunked text, join with spaces
            translated_text = ' '.join(all_translations)
        else:
            # For sentence-split text, join with periods
            translated_text = '. '.join(all_translations) if len(all_translations) > 1 else all_translations[0]

        return {
            "success": True,
            "filename": file.filename,
            "pages_processed": 2,
            "extracted_text": extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text,  # Truncate for response
            "translated_text": translated_text,
            "target_language": target_language,
            "source_language": "eng_Latn",
            "extraction_method": extraction_method,
            "text_length": len(extracted_text),
            "chunks_processed": len(text_chunks),
            "memory_management": "chunking" if len(extracted_text) > 1000 else "standard"
        }

    except Exception as e:
        logger.error(f"❌ PDF translation error: {e}")
        raise HTTPException(status_code=500, detail=f"PDF translation failed: {str(e)}")


@app.post("/clear-memory")
def clear_memory():
    """Clear GPU memory cache"""
    global DEVICE
    try:
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            return {"status": "success", "message": "GPU memory cache cleared"}
        else:
            return {"status": "info", "message": "Running on CPU, no GPU memory to clear"}
    except Exception as e:
        logger.error(f"Error clearing memory: {e}")
        return {"status": "error", "message": f"Failed to clear memory: {str(e)}"}


@app.get("/memory-info")
def get_memory_info():
    """Get current memory usage information"""
    global DEVICE
    try:
        if DEVICE == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            cached = torch.cuda.memory_reserved() / 1024**3  # GB
            return {
                "device": DEVICE,
                "allocated_gb": round(allocated, 2),
                "cached_gb": round(cached, 2),
                "total_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
            }
        else:
            return {"device": DEVICE, "message": "Running on CPU"}
    except Exception as e:
        return {"error": f"Failed to get memory info: {str(e)}"}


def create_pdf_from_text(original_text: str, translated_text: str, filename: str, target_language: str) -> io.BytesIO:
    """Create a PDF with original and translated text"""
    buffer = io.BytesIO()
    
    # Create PDF document
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*inch, bottomMargin=1*inch)
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=20,
        textColor='#2563eb'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        textColor='#374151'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=12,
        leading=16
    )
    
    # Build content
    content = []
    
    # Title
    content.append(Paragraph("PDF Translation Result", title_style))
    content.append(Spacer(1, 12))
    
    # File info
    content.append(Paragraph(f"<b>Original File:</b> {filename}", body_style))
    content.append(Paragraph(f"<b>Target Language:</b> {target_language}", body_style))
    content.append(Spacer(1, 20))
    
    # Original text section
    content.append(Paragraph("Original Text", heading_style))
    # Split long text into paragraphs
    original_paragraphs = original_text.split('\n\n') if '\n\n' in original_text else [original_text]
    for para in original_paragraphs:
        if para.strip():
            content.append(Paragraph(para.strip(), body_style))
    
    content.append(Spacer(1, 20))
    
    # Translated text section
    content.append(Paragraph("Translated Text", heading_style))
    # Split long text into paragraphs
    translated_paragraphs = translated_text.split('\n\n') if '\n\n' in translated_text else [translated_text]
    for para in translated_paragraphs:
        if para.strip():
            content.append(Paragraph(para.strip(), body_style))
    
    # Build PDF
    doc.build(content)
    buffer.seek(0)
    return buffer


@app.post("/download-pdf")
async def download_translated_pdf(
    original_text: str = Form(...),
    translated_text: str = Form(...),
    filename: str = Form(...),
    target_language: str = Form(...)
):
    """Generate and download translated PDF"""
    try:
        # Create PDF with both original and translated text
        pdf_buffer = create_pdf_from_text(
            original_text=original_text,
            translated_text=translated_text,
            filename=filename,
            target_language=target_language
        )
        
        # Create filename for download
        base_name = os.path.splitext(filename)[0]
        download_filename = f"{base_name}_translated_{target_language}.pdf"
        
        # Return as streaming response
        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={download_filename}"}
        )
        
    except Exception as e:
        logger.error(f"❌ PDF generation error: {e}")
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")


def remove_duplicates_from_text(text: str) -> str:
    """Remove duplicate sentences and lines from text"""
    lines = text.split('\n')
    unique_lines = []
    seen_lines = set()
    
    for line in lines:
        clean_line = line.strip().lower()
        # Only add if not empty, not seen before, and has meaningful content
        if clean_line and clean_line not in seen_lines and len(clean_line) > 3:
            unique_lines.append(line.strip())
            seen_lines.add(clean_line)
    
    # Also remove duplicate sentences within the text
    sentences = '. '.join(unique_lines).split('.')
    unique_sentences = []
    seen_sentences = set()
    
    for sentence in sentences:
        clean_sentence = sentence.strip().lower()
        if clean_sentence and clean_sentence not in seen_sentences and len(clean_sentence) > 5:
            unique_sentences.append(sentence.strip())
            seen_sentences.add(clean_sentence)
    
    return '. '.join(unique_sentences)


@app.post("/translate-pdf-enhanced")
async def translate_pdf_enhanced(
    file: UploadFile = File(...),
    target_language: str = Form(...)
):
    """Enhanced PDF translation with duplicate removal and download option"""
    global tokenizer, model, ip, DEVICE
    
    if not all([tokenizer, model, ip]):
        raise HTTPException(status_code=503, detail="Models are still loading. Please try again in a moment.")
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Read PDF content
        pdf_content = await file.read()
        
        # Enhanced text extraction with multiple methods
        extracted_text, extraction_method = extract_text_from_pdf(pdf_content, max_pages=2)
        
        logger.info(f"Text extracted using: {extraction_method}")
        
        # Clean up and remove duplicates from extracted text
        extracted_text = remove_duplicates_from_text(extracted_text.strip())
        
        # Check text length and apply chunking if needed
        if len(extracted_text) > 1000:
            logger.info(f"Text length: {len(extracted_text)} characters. Using chunking for memory efficiency.")
            text_chunks = chunk_text(extracted_text, max_chunk_size=500)
            logger.info(f"Split into {len(text_chunks)} chunks")
        else:
            text_chunks = [sent.strip() for sent in extracted_text.split('.') if sent.strip()]
            if not text_chunks:
                text_chunks = [extracted_text]
        
        # Translate chunks in batches
        all_translations = []
        batch_size = 3
        
        for i in range(0, len(text_chunks), batch_size):
            batch_chunks = text_chunks[i:i + batch_size]
            
            try:
                batch = ip.preprocess_batch(
                    batch_chunks,
                    src_lang="eng_Latn",
                    tgt_lang=target_language,
                )

                inputs = tokenizer(
                    batch,
                    truncation=True,
                    padding="longest",
                    return_tensors="pt",
                    return_attention_mask=True,
                    max_length=512,
                ).to(DEVICE)

                with torch.no_grad():
                    generated_tokens = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        use_cache=False,
                        min_length=0,
                        max_length=256,
                        num_beams=3,
                        num_return_sequences=1,
                        do_sample=False,
                    )

                decoded = tokenizer.batch_decode(
                    generated_tokens.detach().cpu().tolist(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )

                batch_translations = ip.postprocess_batch(decoded, lang=target_language)
                all_translations.extend(batch_translations)
                
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(text_chunks) + batch_size - 1)//batch_size}")
                
            except Exception as batch_error:
                logger.error(f"Error processing batch {i//batch_size + 1}: {batch_error}")
                all_translations.extend(["[Translation failed for this section]"] * len(batch_chunks))
        
        # Join translated chunks
        if len(extracted_text) > 1000:
            translated_text = ' '.join(all_translations)
        else:
            translated_text = '. '.join(all_translations) if len(all_translations) > 1 else all_translations[0]

        # Remove duplicates from translated text as well
        translated_text = remove_duplicates_from_text(translated_text)

        return {
            "success": True,
            "filename": file.filename,
            "pages_processed": 2,
            "extracted_text": extracted_text,
            "translated_text": translated_text,
            "target_language": target_language,
            "source_language": "eng_Latn",
            "extraction_method": extraction_method,
            "text_length": len(extracted_text),
            "chunks_processed": len(text_chunks),
            "memory_management": "chunking" if len(extracted_text) > 1000 else "standard",
            "duplicates_removed": True,
            "download_available": True
        }

    except Exception as e:
        logger.error(f"❌ PDF translation error: {e}")
        raise HTTPException(status_code=500, detail=f"PDF translation failed: {str(e)}")


@app.post("/download-translated-pdf")
async def download_translated_pdf_endpoint(
    original_text: str = Form(...),
    translated_text: str = Form(...),
    filename: str = Form(...),
    target_language: str = Form(...)
):
    """Generate and download translated PDF"""
    try:
        # Create PDF with both original and translated text
        pdf_buffer = create_pdf_from_text(
            original_text=original_text,
            translated_text=translated_text,
            filename=filename,
            target_language=target_language
        )
        
        # Create filename for download
        base_name = os.path.splitext(filename)[0]
        download_filename = f"{base_name}_translated_{target_language}.pdf"
        
        # Return as streaming response
        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={download_filename}"}
        )
        
    except Exception as e:
        logger.error(f"❌ PDF generation error: {e}")
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")