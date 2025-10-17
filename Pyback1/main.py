from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict, Any
import torch
import logging
import asyncio
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor
import PyPDF2
import fitz  # PyMuPDF
import pytesseract
import io
import os
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from fastapi.responses import StreamingResponse
from font_management import initialize_font_system, get_font_registry, get_language_mapper, get_style_factory
from typing import AsyncGenerator

from dataclasses import dataclass, field
import json
from pdf_translation_api import router as pdf_translation_router
from pdf_simple_service import router as pdf_simple_router
from translation_core import (
    extract_text_from_pdf, create_pdf_from_text, PdfTextElement, PdfLayoutData,
    ip, tokenizer, model, DEVICE, chunk_text, remove_duplicates_from_text
)

# Create FastAPI app
app = FastAPI(
    title="IndicTrans2 Translation API",
    description="API for translating text and PDFs using IndicTrans2 model with font support",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logging.getLogger('font_management').setLevel(logging.WARNING)

# Font system globals
font_system_initialized = False
font_initialization_error = None


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
    global tokenizer, model, ip, DEVICE, font_system_initialized, font_initialization_error
    models_loaded = all([tokenizer, model, ip])
    
    # Get font system status
    font_status = "initialized" if font_system_initialized else "loading"
    if font_initialization_error:
        font_status = "error"
    
    # Get font registry summary if available
    font_info = {}
    if font_system_initialized:
        try:
            font_registry = get_font_registry()
            summary = font_registry.get_registration_summary()
            font_info = {
                "registered_fonts": summary['registered_fonts'],
                "font_families": summary['total_families'],
                "available_families": summary['available_families']
            }
        except Exception as e:
            logger.warning(f"Could not get font info for health check: {e}")
    
    return {
        "status": "healthy" if models_loaded else "loading",
        "device": str(DEVICE) if DEVICE else "unknown",
        "model": "indictrans2-en-indic-dist-200M",
        "components_loaded": {
            "tokenizer": tokenizer is not None,
            "model": model is not None,
            "processor": ip is not None,
            "font_system": font_system_initialized
        },
        "font_system": {
            "status": font_status,
            "error": font_initialization_error,
            "message": "Font system not yet initialized" if not font_system_initialized else "Font system initialized",
            **font_info
        }
    }


async def initialize_fonts_async():
    """
    Asynchronously initialize the font system to avoid blocking startup.
    """
    global font_system_initialized, font_initialization_error
    
    try:
        logger.info("🔤 Starting font system initialization...")
        
        # Load font configuration from environment variables
        from font_config import get_font_config
        font_config = get_font_config()
        logger.info(f"Font base path: {font_config.font_base_path}")
        
        # Run font initialization in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        successful, failed = await loop.run_in_executor(
            None, 
            initialize_font_system, 
            font_config.font_base_path,
            font_config
        )
        
        # Log registration results
        if successful > 0:
            logger.info(f"✅ Font system initialized: {successful} fonts registered successfully")
            if failed > 0:
                logger.warning(f"⚠️ Font registration issues: {failed} fonts failed to register")
        else:
            logger.warning("⚠️ No fonts were registered - PDF generation will use default fonts")
        
        # Get registration summary for detailed logging
        font_registry = get_font_registry()
        summary = font_registry.get_registration_summary()
        
        # logger.info(f"📊 Font registration summary:")
        # logger.info(f"  - Total fonts: {summary['total_fonts']}")
        # logger.info(f"  - Registered fonts: {summary['registered_fonts']}")
        # logger.info(f"  - Font families: {summary['total_families']}")
        # logger.info(f"  - Available families: {summary['available_families']}")
        
        # Log any registration errors
        if summary['registration_errors']:
            logger.warning(f"Font registration errors ({len(summary['registration_errors'])}):")
            for error in summary['registration_errors'][:5]:  # Log first 5 errors
                logger.warning(f"  - {error}")
            if len(summary['registration_errors']) > 5:
                logger.warning(f"  ... and {len(summary['registration_errors']) - 5} more errors")
        
        # Test language mapper
        language_mapper = get_language_mapper()
        mapping_summary = language_mapper.get_mapping_summary()
        # logger.info(f"🌐 Language mapping summary:")
        # logger.info(f"  - Total language mappings: {mapping_summary['total_language_mappings']}")
        # logger.info(f"  - Available mappings: {mapping_summary['available_language_mappings']}")
        # logger.info(f"  - Supported languages: {len(mapping_summary['supported_languages'])}")
        
        font_system_initialized = True
        logger.info("🎉 Font system initialization completed successfully!")
        
    except Exception as e:
        error_msg = f"Font system initialization failed: {e}"
        logger.error(f"❌ {error_msg}")
        font_initialization_error = error_msg
        # Don't raise - allow app to continue with default fonts


@app.on_event("startup")
async def startup_event():
    global tokenizer, model, ip, DEVICE

    try:
        logger.info("🚀 Starting application initialization...")
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {DEVICE}")

        # Start font system initialization asynchronously (non-blocking)
        asyncio.create_task(initialize_fonts_async())

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

        logger.info("🎉 All core components loaded successfully!")

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


def _process_pymupdf_layout(page_dict: Dict[str, Any], page_number: int) -> PdfLayoutData:
    """
    Processes the dictionary output from fitz.page.get_text("dict") to extract structured layout data.
    """
    page_layout = PdfLayoutData(page_number=page_number, width=page_dict['width'], height=page_dict['height'])
    
    for block in page_dict['blocks']:
        if block['type'] == 0:  # Text block
            for line in block['lines']:
                line_text = ""
                line_bbox = list(line['bbox'])
                for span in line['spans']:
                    # Accumulate text for the line
                    line_text += span['text']
                    
                    # Attempt to get font details from the first span in the line
                    # More sophisticated logic might average or take the most common font/size
                    font_name = span['font']
                    font_size = span['size']
                    color_rgb = fitz.utils.sRGB_to_rgb(span['color']) # Convert integer color to RGB tuple
                    
                    # Check for bold/italic - simplistic check, might need regex for more complex font names
                    is_bold = "bold" in font_name.lower()
                    is_italic = "italic" in font_name.lower()

                    page_layout.text_elements.append(
                        PdfTextElement(
                            text=span['text'],
                            bbox=tuple(span['bbox']), # Use span's bbox for granular placement
                            font_name=font_name,
                            font_size=font_size,
                            color=color_rgb,
                            is_bold=is_bold,
                            is_italic=is_italic
                        )
                    )

    # Note: This is a simplified extraction. For full fidelity, you'd need to handle
    # alignments, line spacing, super/subscripts, etc., which are complex.
    return page_layout


def extract_text_from_pdf(pdf_content: bytes, max_pages: int = 2) -> tuple[str, str, List[PdfLayoutData]]:
    """
    Enhanced PDF text extraction with OCR fallback and memory management
    Returns: (extracted_text, extraction_method, layout_data)
    """
    extraction_method = "Unknown"
    extracted_text = ""
    
    all_layout_data: List[PdfLayoutData] = []
    
    try:
        # Method 1: Try PyPDF2 first (fastest for text-based PDFs)
        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        # Limit pages
        max_pages_to_process = min(len(pdf_reader.pages), max_pages)
        
        for page_num in range(max_pages_to_process):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            if page_text.strip():
                extracted_text += page_text + "\n"
        
        if extracted_text.strip():
            extraction_method = "PyPDF2 (text-based)"
            # For PyPDF2, we don't have detailed layout, so return empty list for now
            logger.info(f"Extracted text using {extraction_method}")
            return extracted_text.strip(), extraction_method, []
            
    except Exception as e:
        logger.warning(f"PyPDF2 extraction failed: {e}")
    
    try:
        # Method 2: Try PyMuPDF (better for complex PDFs and layout extraction)
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        
        max_pages_to_process = min(len(pdf_document), max_pages)
        
        current_page_text = ""
        for page_num in range(max_pages_to_process):
            page = pdf_document[page_num]
            page_dict = page.get_text("dict")  # Get detailed layout as dictionary
            
            # Process layout data
            page_layout = _process_pymupdf_layout(page_dict, page_num)
            all_layout_data.append(page_layout)
            
            # Also extract plain text for translation
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
        all_layout_data = [] # Clear any partial layout data on failure
        # Fallback to simple text extraction from PyMuPDF if dict fails
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

    # Method 3: OCR fallback for scanned PDFs (with memory management)
    try:
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        
        max_pages_to_process = min(len(pdf_document), max_pages)
        
        current_page_text = ""
        for page_num in range(max_pages_to_process):
            page = pdf_document[page_num]
            
            # Convert page to image with lower resolution to save memory
            mat = fitz.Matrix(1.5, 1.5)  # Reduced from 2.0 to save memory
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Convert to PIL Image
            # No longer need PIL Image here if we are removing Pillow
            # image = Image.open(io.BytesIO(img_data))
            
            # Perform OCR (Pillow dependencies still needed for tesseract)
            # pytesseract.image_to_string expects PIL Image or path to image
            # Re-introduce Image import if OCR is to be used
            from PIL import Image # Temporarily re-import Image for OCR
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
                current_page_text += page_text + "\n"
            
            # Clean up memory
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
    
    # If all methods fail
    raise ValueError("Could not extract text from PDF. The file might be corrupted, password-protected, or contain only images without readable text.")


def chunk_text(text: str, max_chunk_size: int = 200) -> List[str]:
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
    """Extract text from PDF and translate it with enhanced extraction methods, memory management, and font support"""
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
        extracted_text, extraction_method, layout_data = extract_text_from_pdf(pdf_content, max_pages=2)
        
        logger.info(f"Text extracted using: {extraction_method}")
        
        # Clean up the extracted text
        extracted_text = extracted_text.strip()
        
        # Use word-based processing for optimal performance (matching Live Preview)
        logger.info(f"Text length: {len(extracted_text)} characters. Using word-based processing.")
        
        # Split into words like Live Preview
        words = extracted_text.split()
        text_chunks = [word.strip() for word in words if word.strip() and len(word.strip()) > 1]
        
        logger.info(f"Split into {len(text_chunks)} word elements")
        
        # Use same batch size as Live Preview
        all_translations = []
        batch_size = 5  # Same as Live Preview
        
        # Simple optimized sequential processing (avoiding threading issues)
        logger.info(f"Processing {len(text_chunks)} chunks with optimized sequential batching")
        
        # Use larger batch size for better performance without threading complexity
        optimized_batch_size = batch_size * 2  # Double the batch size for speed
        
        for i in range(0, len(text_chunks), optimized_batch_size):
            batch_chunks = text_chunks[i:i + optimized_batch_size]
            
            try:
                # Preprocess batch
                batch = ip.preprocess_batch(
                    batch_chunks,
                    src_lang="eng_Latn",
                    tgt_lang=target_language,
                )

                # Tokenize with standard settings
                inputs = tokenizer(
                    batch,
                    truncation=True,
                    padding="longest",
                    return_tensors="pt",
                    return_attention_mask=True,
                    max_length=512,
                ).to(DEVICE)

                # Generate translations
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

                # Decode
                decoded = tokenizer.batch_decode(
                    generated_tokens.detach().cpu().tolist(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )

                # Postprocess
                batch_translations = ip.postprocess_batch(decoded, lang=target_language)
                all_translations.extend(batch_translations)
                
                # Optimized memory clearing
                if DEVICE == "cuda" and i % (optimized_batch_size * 3) == 0:
                    torch.cuda.empty_cache()
                
                batch_num = i // optimized_batch_size + 1
                total_batches = (len(text_chunks) + optimized_batch_size - 1) // optimized_batch_size
                logger.info(f"Optimized: Processed batch {batch_num}/{total_batches}")
                
            except Exception as batch_error:
                logger.error(f"Error processing optimized batch {i//optimized_batch_size + 1}: {batch_error}")
                all_translations.extend(["[Translation failed]"] * len(batch_chunks))
        
        logger.info(f"Optimized sequential processing completed: {len(all_translations)} translations generated")
        
        # Join translated words back together with spaces
        translated_text = ' '.join(all_translations)

        # Get font information for the target language
        font_info = {}
        if font_system_initialized:
            try:
                language_mapper = get_language_mapper()
                font_info = language_mapper.get_font_info_for_language(target_language)
                logger.info(f"Font info for {target_language}: {font_info['selected_font']}")
            except Exception as e:
                logger.warning(f"Could not get font info for {target_language}: {e}")

        return {
            "success": True,
            "filename": file.filename,
            "pages_processed": 2,
            "extracted_text": extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text,  # Truncate for response
            "translated_text": translated_text,
            "target_language": target_language,
            "source_language": "eng_Latn",
            "language_code": target_language,  # Pass language code for PDF generation
            "extraction_method": extraction_method,
            "text_length": len(extracted_text),
            "chunks_processed": len(text_chunks),
            "memory_management": "chunking" if len(extracted_text) > 1000 else "standard",
            "font_support": {
                "enabled": font_system_initialized,
                "font_info": font_info
            }
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


@app.post("/benchmark-translation")
async def benchmark_translation(
    text: str = Form(...),
    target_language: str = Form(default="hin_Deva"),
    iterations: int = Form(default=3)
):
    """Benchmark translation performance with optimized settings"""
    global tokenizer, model, ip, DEVICE
    
    if not all([tokenizer, model, ip]):
        raise HTTPException(status_code=503, detail="Models are still loading")
    
    import time
    
    results = []
    total_start = time.time()
    
    for i in range(iterations):
        start_time = time.time()
        
        try:
            # Use smaller chunking for better performance
            chunks = chunk_text(text, max_chunk_size=150)
            
            # Translate with optimized batch size
            batch_size = 8
            all_translations = []
            
            for j in range(0, len(chunks), batch_size):
                batch = chunks[j:j+batch_size]
                
                pre = ip.preprocess_batch(batch, src_lang="eng_Latn", tgt_lang=target_language)
                inputs = tokenizer(
                    pre,
                    truncation=True,
                    padding="longest",
                    return_tensors="pt",
                    return_attention_mask=True,
                    max_length=400,
                ).to(DEVICE)
                
                with torch.no_grad():
                    generated_tokens = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        use_cache=False,
                        min_length=0,
                        max_length=180,
                        num_beams=2,
                        num_return_sequences=1,
                        do_sample=False,
                        early_stopping=True,
                    )
                
                decoded = tokenizer.batch_decode(
                    generated_tokens.detach().cpu().tolist(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                
                batch_trans = ip.postprocess_batch(decoded, lang=target_language)
                all_translations.extend(batch_trans)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            results.append({
                "iteration": i + 1,
                "processing_time": round(processing_time, 3),
                "chunks_processed": len(chunks),
                "words_per_second": round(len(text.split()) / processing_time, 2),
                "characters_per_second": round(len(text) / processing_time, 2)
            })
            
        except Exception as e:
            results.append({
                "iteration": i + 1,
                "error": str(e),
                "processing_time": None
            })
    
    total_time = time.time() - total_start
    successful_runs = [r for r in results if "error" not in r]
    
    if successful_runs:
        avg_time = sum(r["processing_time"] for r in successful_runs) / len(successful_runs)
        avg_wps = sum(r["words_per_second"] for r in successful_runs) / len(successful_runs)
    else:
        avg_time = None
        avg_wps = None
    
    return {
        "benchmark_results": results,
        "summary": {
            "total_time": round(total_time, 3),
            "successful_iterations": len(successful_runs),
            "failed_iterations": len(results) - len(successful_runs),
            "average_processing_time": round(avg_time, 3) if avg_time else None,
            "average_words_per_second": round(avg_wps, 2) if avg_wps else None,
            "text_length": len(text),
            "target_language": target_language
        },
        "optimization_info": {
            "batch_size": 8,
            "max_length": 180,
            "num_beams": 2,
            "early_stopping": True,
            "chunk_size": 800
        }
    }


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


@app.get("/performance-stats")
def get_performance_stats():
    """Get performance statistics and optimization info"""
    return {
        "optimization_status": "live_preview_optimized",
        "batch_sizes": {
            "translate_pdf": 5,
            "translate_pdf_enhanced": 5,
            "translate_pdf_enhanced_elements": 10,
            "live_preview": 5
        },
        "generation_params": {
            "max_length": "256 (same as Live Preview)",
            "num_beams": "3 (same as Live Preview)",
            "early_stopping": "disabled (matching Live Preview)",
            "memory_clearing": "every batch (stable)"
        },
        "processing_approach": {
            "method": "word_based (matching Live Preview)",
            "chunking": "individual words instead of text chunks",
            "optimization": "proven Live Preview parameters"
        },
        "performance_focus": {
            "speed": "optimized to match Live Preview performance",
            "quality": "maintained with proven parameters",
            "consistency": "same approach across all endpoints"
        }
    }


@app.get("/font-status")
async def get_font_status():
    """Get detailed font system status and registered fonts information"""
    global font_system_initialized, font_initialization_error
    
    try:
        if not font_system_initialized:
            return {
                "status": "error" if font_initialization_error else "loading",
                "error": font_initialization_error,
                "message": "Font system not yet initialized"
            }
        
        # Get font registry information
        font_registry = get_font_registry()
        registry_summary = font_registry.get_registration_summary()
        
        # Get language mapper information
        language_mapper = get_language_mapper()
        mapping_summary = language_mapper.get_mapping_summary()
        
        # Get detailed font information
        font_details = {}
        for family_name in registry_summary['available_families']:
            family_fonts = font_registry.get_font_family(family_name)
            if family_fonts:
                font_details[family_name] = []
                for font_name in family_fonts:
                    font_info = font_registry.get_font_info(font_name)
                    if font_info:
                        font_details[family_name].append({
                            "name": font_info.name,
                            "weight": font_info.weight,
                            "style": font_info.style,
                            "file_path": font_info.file_path,
                            "file_size": font_info.file_size,
                            "is_registered": font_info.is_registered
                        })
        
        return {
            "status": "initialized",
            "font_registry": {
                "total_fonts": registry_summary['total_fonts'],
                "registered_fonts": registry_summary['registered_fonts'],
                "failed_fonts": registry_summary['failed_fonts'],
                "total_families": registry_summary['total_families'],
                "available_families": registry_summary['available_families'],
                "fonts_by_family": registry_summary['fonts_by_family'],
                "registration_errors": registry_summary['registration_errors']
            },
            "language_mapping": {
                "total_mappings": mapping_summary['total_language_mappings'],
                "available_mappings": mapping_summary['available_language_mappings'],
                "supported_languages": mapping_summary['supported_languages'],
                "available_languages": mapping_summary['available_languages'],
                "languages_by_font": mapping_summary['languages_by_font_family']
            },
            "font_details": font_details,
            "environment": {
                "font_base_path": os.getenv('FONT_BASE_PATH', 'Pyback1/fonts'),
                "font_logging_enabled": os.getenv('ENABLE_FONT_LOGGING', 'true').lower() == 'true'
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting font status: {e}")
        return {
            "status": "error",
            "error": f"Failed to get font status: {str(e)}"
        }


@app.get("/font-health")
async def get_font_health():
    """Perform comprehensive font system health check"""
    global font_system_initialized, font_initialization_error
    
    try:
        if not font_system_initialized:
            return {
                "status": "error" if font_initialization_error else "loading",
                "error": font_initialization_error,
                "message": "Font system not yet initialized"
            }
        
        # Perform health check
        font_registry = get_font_registry()
        health_check = font_registry.perform_health_check()
        
        # Get usage statistics
        usage_stats = font_registry.get_font_usage_stats()
        
        # Get language coverage report
        language_mapper = get_language_mapper()
        coverage_report = language_mapper.get_language_coverage_report()
        
        return {
            "health_check": health_check,
            "usage_statistics": usage_stats,
            "language_coverage": coverage_report,
            "system_info": {
                "font_base_path": os.getenv('FONT_BASE_PATH', 'Pyback1/fonts'),
                "font_logging_enabled": os.getenv('ENABLE_FONT_LOGGING', 'true').lower() == 'true',
                "initialization_error": font_initialization_error
            }
        }
        
    except Exception as e:
        logger.error(f"Error performing font health check: {e}")
        return {
            "status": "error",
            "error": f"Failed to perform health check: {str(e)}"
        }


@app.get("/font-test/{language_code}")
async def test_font_mapping(language_code: str):
    """Test font mapping for a specific language code"""
    global font_system_initialized, font_initialization_error
    
    try:
        if not font_system_initialized:
            return {
                "status": "error" if font_initialization_error else "loading",
                "error": font_initialization_error,
                "message": "Font system not yet initialized"
            }
        
        # Test the language mapping
        language_mapper = get_language_mapper()
        test_result = language_mapper.test_language_mapping(language_code)
        
        # Get additional font info
        font_info = language_mapper.get_font_info_for_language(language_code)
        
        return {
            "language_code": language_code,
            "test_result": test_result,
            "font_info": font_info,
            "timestamp": str(os.times())
        }
        
    except Exception as e:
        logger.error(f"Error testing font mapping for {language_code}: {e}")
        return {
            "status": "error",
            "language_code": language_code,
            "error": f"Failed to test font mapping: {str(e)}"
        }


@app.get("/font-debug")
async def get_font_debug_info():
    """Get comprehensive font system debugging information"""
    global font_system_initialized, font_initialization_error
    
    try:
        debug_info = {
            "system_status": {
                "font_system_initialized": font_system_initialized,
                "initialization_error": font_initialization_error,
                "timestamp": str(os.times())
            },
            "environment": {
                "font_base_path": os.getenv('FONT_BASE_PATH', 'Pyback1/fonts'),
                "font_logging_enabled": os.getenv('ENABLE_FONT_LOGGING', 'true').lower() == 'true',
                "font_cache_size": os.getenv('FONT_CACHE_SIZE', 'not_set'),
                "working_directory": os.getcwd()
            }
        }
        
        if font_system_initialized:
            # Get detailed debugging information
            font_registry = get_font_registry()
            language_mapper = get_language_mapper()
            style_factory = get_style_factory()
            
            # Font registry debug info
            debug_info["font_registry"] = {
                "registration_summary": font_registry.get_registration_summary(),
                "usage_stats": font_registry.get_font_usage_stats(),
                "health_check": font_registry.perform_health_check()
            }
            
            # Language mapper debug info
            debug_info["language_mapper"] = {
                "mapping_summary": language_mapper.get_mapping_summary(),
                "coverage_report": language_mapper.get_language_coverage_report()
            }
            
            # Style factory debug info
            debug_info["style_factory"] = {
                "cache_info": style_factory.get_cache_info()
            }
            
            # Test common languages
            test_languages = ['hin_Deva', 'urd_Arab', 'sin_Sinh', 'hi', 'ur', 'si']
            debug_info["language_tests"] = {}
            
            for lang in test_languages:
                try:
                    debug_info["language_tests"][lang] = language_mapper.test_language_mapping(lang)
                except Exception as e:
                    debug_info["language_tests"][lang] = {"error": str(e)}
        
        return debug_info
        
    except Exception as e:
        logger.error(f"Error getting font debug info: {e}")
        return {
            "status": "error",
            "error": f"Failed to get debug info: {str(e)}"
        }


@app.post("/font-render-test")
async def test_font_rendering(
    language_code: str = Form(...),
    sample_text: str = Form(default="This is a sample text for font rendering test.")
):
    """Test font rendering by generating a sample PDF with the specified language"""
    global font_system_initialized, font_initialization_error
    
    try:
        if not font_system_initialized:
            return {
                "status": "error" if font_initialization_error else "loading",
                "error": font_initialization_error,
                "message": "Font system not yet initialized"
            }
        
        logger.info(f"🧪 Testing font rendering for language: {language_code}")
        
        # Test font mapping first
        language_mapper = get_language_mapper()
        mapping_test = language_mapper.test_language_mapping(language_code)
        
        # Create a test PDF
        try:
            pdf_buffer = create_pdf_from_text(
                original_text_chunks=["Original English text for testing"],
                translated_text_chunks=["Translated English text for testing"],
                filename="font_render_test.pdf",
                target_language=language_code,
                language_code=language_code
            )
            
            pdf_size = len(pdf_buffer.getvalue())
            
            return {
                "status": "success",
                "language_code": language_code,
                "sample_text": sample_text,
                "mapping_test": mapping_test,
                "pdf_generated": True,
                "pdf_size_bytes": pdf_size,
                "pdf_size_kb": round(pdf_size / 1024, 2),
                "font_system_status": "initialized",
                "timestamp": str(os.times())
            }
            
        except Exception as pdf_error:
            logger.error(f"Failed to generate test PDF: {pdf_error}")
            return {
                "status": "pdf_error",
                "language_code": language_code,
                "sample_text": sample_text,
                "mapping_test": mapping_test,
                "pdf_generated": False,
                "error": str(pdf_error),
                "timestamp": str(os.times())
            }
        
    except Exception as e:
        logger.error(f"Error in font rendering test: {e}")
        return {
            "status": "error",
            "language_code": language_code,
            "error": f"Font rendering test failed: {str(e)}"
        }


@app.get("/font-monitoring-summary")
async def get_font_monitoring_summary():
    """Get a summary of all font monitoring and debugging capabilities"""
    global font_system_initialized, font_initialization_error
    
    try:
        summary = {
            "monitoring_capabilities": {
                "font_status": {
                    "endpoint": "/font-status",
                    "description": "Detailed font system status and registered fonts information",
                    "available": True
                },
                "font_health": {
                    "endpoint": "/font-health", 
                    "description": "Comprehensive font system health check with recommendations",
                    "available": font_system_initialized
                },
                "font_test": {
                    "endpoint": "/font-test/{language_code}",
                    "description": "Test font mapping for specific language codes",
                    "available": font_system_initialized
                },
                "font_debug": {
                    "endpoint": "/font-debug",
                    "description": "Comprehensive debugging information for font system",
                    "available": font_system_initialized
                },
                "font_render_test": {
                    "endpoint": "/font-render-test",
                    "description": "Test font rendering by generating sample PDFs",
                    "available": font_system_initialized
                }
            },
            "system_status": {
                "font_system_initialized": font_system_initialized,
                "initialization_error": font_initialization_error,
                "logging_enabled": os.getenv('ENABLE_FONT_LOGGING', 'true').lower() == 'true'
            },
            "quick_stats": {}
        }
        
        # Add quick stats if font system is initialized
        if font_system_initialized:
            try:
                font_registry = get_font_registry()
                language_mapper = get_language_mapper()
                
                registry_summary = font_registry.get_registration_summary()
                mapping_summary = language_mapper.get_mapping_summary()
                
                summary["quick_stats"] = {
                    "registered_fonts": registry_summary['registered_fonts'],
                    "font_families": registry_summary['total_families'],
                    "supported_languages": len(mapping_summary['supported_languages']),
                    "available_languages": len(mapping_summary['available_languages']),
                    "registration_errors": len(registry_summary['registration_errors'])
                }
                
            except Exception as e:
                summary["quick_stats"] = {"error": f"Failed to get stats: {str(e)}"}
        
        # Add usage examples
        summary["usage_examples"] = {
            "check_font_health": "GET /font-health",
            "test_hindi_fonts": "GET /font-test/hin_Deva", 
            "test_urdu_fonts": "GET /font-test/urd_Arab",
            "test_sinhala_fonts": "GET /font-test/sin_Sinh",
            "render_test": "POST /font-render-test (with language_code and sample_text)",
            "full_debug_info": "GET /font-debug"
        }
        
        return summary
        
    except Exception as e:
        logger.error(f"Error getting font monitoring summary: {e}")
        return {
            "status": "error",
            "error": f"Failed to get monitoring summary: {str(e)}"
        }


def create_pdf_from_text(original_text_chunks: List[str], translated_text_chunks: List[str], filename: str, target_language: str, language_code: str = None, layout_data: List[PdfLayoutData] = None) -> io.BytesIO:
    """
    Create a PDF with original and translated text using appropriate fonts for the target language and preserving layout.
    
    Args:
        original_text_chunks: List of original text chunks
        translated_text_chunks: List of translated text chunks
        filename: Original filename for reference
        target_language: Target language name for display
        language_code: Language code for font selection (e.g., 'hin_Deva', 'urd_Arab')
        layout_data: Optional list of PdfLayoutData for layout preservation
    
    Returns:
        BytesIO buffer containing the generated PDF
    """
    buffer = io.BytesIO()
    
    from reportlab.pdfgen import canvas
    from reportlab.lib.colors import black
    from reportlab.lib.units import inch # Move inch import here

    c = canvas.Canvas(buffer, pagesize=A4)

    # Reconstruct full original and translated texts for approximate mapping
    full_original_text = " ".join(original_text_chunks).strip()
    full_translated_text = " ".join(translated_text_chunks).strip()
    
    # Keep track of the current position in the full original text to find segments
    current_original_char_pos = 0
    translated_segment_map = {}

    # Attempt to build a rough character-level mapping from original to translated text
    # This is a linear approximation and may not be perfect for complex sentence restructurings
    if full_original_text and full_translated_text:
        original_len = len(full_original_text)
        translated_len = len(full_translated_text)
        length_ratio = translated_len / original_len if original_len > 0 else 1
        
        current_original_idx = 0
        current_translated_idx = 0
        
        # Create a simplified character mapping for proportional scaling
        # This maps original_char_index -> translated_char_index
        char_map = {}
        for i in range(original_len):
            char_map[i] = int(i * length_ratio)
        char_map[original_len] = translated_len # Ensure end maps correctly

        # This map will store (original_text_start_index, original_text_end_index) -> translated_string
        # It will be populated as we iterate elements if we can accurately find them.
        # For now, we'll do a simple substring replacement based on the overall text.


    try:
        # Determine selected font family and variants
        selected_font_family = 'Helvetica' # Default fallback
        selected_font_regular = 'Helvetica'
        selected_font_bold = 'Helvetica-Bold'
        selected_font_italic = 'Helvetica-Oblique'
        selected_font_bold_italic = 'Helvetica-BoldOblique'
        
        if font_system_initialized:
            language_mapper = get_language_mapper()
            font_info = language_mapper.get_font_info_for_language(language_code or target_language)
            selected_font_family = font_info['selected_font']
            
            # Attempt to get specific font variants using FontRegistry's fallback mechanism
            # If a specific variant (e.g., Bold) is not found, it will fallback to Regular or default.
            selected_font_regular = language_mapper.font_registry.get_font_by_family_and_weight_with_fallback(selected_font_family, 'Regular') or 'Helvetica'
            selected_font_bold = language_mapper.font_registry.get_font_by_family_and_weight_with_fallback(selected_font_family, 'Bold') or selected_font_regular
            selected_font_italic = language_mapper.font_registry.get_font_by_family_and_weight_with_fallback(selected_font_family, 'Italic') or selected_font_regular
            selected_font_bold_italic = language_mapper.font_registry.get_font_by_family_and_weight_with_fallback(selected_font_family, 'BoldItalic') or selected_font_bold or selected_font_italic or selected_font_regular

            # Ensure these selected fonts are registered with ReportLab
            for font_name_to_check in [selected_font_family, selected_font_regular, selected_font_bold, selected_font_italic, selected_font_bold_italic]:
                if font_name_to_check and not pdfmetrics.getFont(font_name_to_check):
                    try:
                        font_info_to_register = language_mapper.font_registry.get_font_info(font_name_to_check)
                        if font_info_to_register and os.path.exists(font_info_to_register.file_path):
                            pdfmetrics.registerFont(TTFont(font_name_to_check, font_info_to_register.file_path))
                            logger.info(f"Registered dynamically: {font_name_to_check} from {font_info_to_register.file_path}")
                        else:
                            logger.warning(f"Could not find font file for dynamic registration: {font_name_to_check}")
                    except Exception as e:
                        logger.error(f"Failed to dynamically register {font_name_to_check}: {e}")

        # Function to get the appropriate font name based on style flags
        def get_current_font_name(is_bold: bool, is_italic: bool) -> str:
            if is_bold and is_italic: return selected_font_bold_italic
            if is_bold: return selected_font_bold
            if is_italic: return selected_font_italic
            return selected_font_regular

        # Use a simple ReportLab canvas for direct drawing if layout_data is available
        if layout_data:
            logger.info(f"📄 Starting PDF generation with font support and layout preservation")
            for page_idx, page_layout in enumerate(layout_data):
                c.setPageSize((page_layout.width, page_layout.height))
                
                # For debugging: draw original text for comparison
                # c.setFillColorRGB(0.5, 0.5, 0.5)
                # c.setFont('Helvetica', 8)
                # c.drawString(10, page_layout.height - 20, f"Original Page {page_layout.page_number + 1}")

                # Reset current_original_char_pos for each page if we want to align by page content
                # For simplicity, we assume continuous text for full_original_text and full_translated_text
                # If we need page-level translation, this would be more complex.

                for element in page_layout.text_elements:
                    original_text_segment = element.text
                    
                    # Find the start index of the current original_text_segment within the full_original_text
                    # This is still a heuristic. For truly robust layout, a token aligner is needed.
                    # For now, we'll use a simple find and assume order is preserved.
                    start_idx_in_full_original = full_original_text.find(original_text_segment, current_original_char_pos)

                    if start_idx_in_full_original != -1:
                        end_idx_in_full_original = start_idx_in_full_original + len(original_text_segment)

                        # Calculate corresponding indices in translated text using the char_map
                        translated_start_idx = char_map.get(start_idx_in_full_original, 0)
                        translated_end_idx = char_map.get(end_idx_in_full_original, translated_len)
                        
                        # Extract the translated segment
                        translated_text_segment = full_translated_text[translated_start_idx:translated_end_idx]
                        current_original_char_pos = end_idx_in_full_original # Update pointer

                        if translated_text_segment.strip():
                            current_font_name = get_current_font_name(element.is_bold, element.is_italic)

                            # ReportLab expects (R, G, B) in range (0,1) for setFillColorRGB
                            r, g, b = element.color
                            c.setFillColorRGB(r, g, b)
                            c.setFont(current_font_name, element.font_size)

                            # Adjust Y coordinate
                            x, y_top = element.bbox[0], element.bbox[1]
                            y_reportlab_baseline = page_layout.height - y_top - element.font_size
                            
                            # Implement basic text wrapping
                            words = translated_text_segment.strip().split(' ')
                            current_line_words = []
                            current_line_width = 0
                            available_width = element.bbox[2] - element.bbox[0]
                            line_height = element.font_size * 1.2 # 120% of font size for line spacing

                            for word in words:
                                # Calculate width of word plus a space
                                word_width = c.stringWidth(word + " ", current_font_name, element.font_size)

                                if current_line_width + word_width > available_width and current_line_words:
                                    # Draw the current line and start a new one
                                    c.drawString(x, y_reportlab_baseline, " ".join(current_line_words))
                                    y_reportlab_baseline -= line_height # Move down for the next line
                                    current_line_words = [word]
                                    current_line_width = c.stringWidth(word + " ", current_font_name, element.font_size)
                                else:
                                    current_line_words.append(word)
                                    current_line_width += word_width
                            
                            # Draw any remaining text in the current line
                            if current_line_words:
                                c.drawString(x, y_reportlab_baseline, " ".join(current_line_words))

                c.showPage()
            logger.info(f"✅ Successfully created PDF with layout preservation.")

        else:
            # Canvas-based fallback if no layout data is provided
            logger.warning("⚠️ No layout data provided, falling back to simple canvas rendering.")
            c.setPageSize(A4) # Reset to default A4 if no layout data
            width, height = A4
            margin = 1 * inch
            y_position = height - margin
            line_height = 14

            c.setFont(get_current_font_name(False, False), 16)
            c.drawString(margin, y_position, "PDF Translation Result (Simple Layout)")
            y_position -= 2 * line_height

            c.setFont(get_current_font_name(False, False), 12)
            c.drawString(margin, y_position, f"Original File: {filename}")
            y_position -= line_height
            c.drawString(margin, y_position, f"Target Language: {target_language}")
            y_position -= 2 * line_height

            c.setFont(get_current_font_name(True, False), 12) # Bold
            c.drawString(margin, y_position, "Original Text:")
            y_position -= line_height
            c.setFont(get_current_font_name(False, False), 10)
            for chunk in original_text_chunks:
                for line in chunk.split('\n'):
                    if y_position < margin:
                        c.showPage()
                        c.setPageSize(A4)
                        y_position = height - margin
                    c.drawString(margin, y_position, line.strip())
                    y_position -= line_height
            y_position -= line_height

            c.setFont(get_current_font_name(True, False), 12) # Bold
            c.drawString(margin, y_position, "Translated Text:")
            y_position -= line_height
            c.setFont(get_current_font_name(False, False), 10)
            for chunk in translated_text_chunks:
                for line in chunk.split('\n'):
                    if y_position < margin:
                        c.showPage()
                        c.setPageSize(A4)
                        y_position = height - margin
                    c.drawString(margin, y_position, line.strip())
                    y_position -= line_height
            y_position -= line_height
            c.showPage()
            logger.info(f"✅ Successfully created PDF with simple canvas rendering.")
            
        c.save()
        buffer.seek(0)
        
        # Log final PDF creation details
        pdf_size = len(buffer.getvalue())
        logger.info(f"✅ Successfully created PDF:")
        logger.info(f"  - Elements: {sum(len(p.text_elements) for p in layout_data) if layout_data else 0} text elements processed")
        logger.info(f"  - Size: {pdf_size} bytes ({pdf_size / 1024:.1f} KB)")
        logger.info(f"  - Language: {target_language}")
        logger.info(f"  - Font system used: {'Yes' if font_system_initialized else 'No'}")
        
        return buffer
        
    except Exception as e:
        logger.error(f"Error creating PDF: {e}")
        # Create a simple fallback PDF in case of errors
        try:
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch # Re-add inch import

            doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*inch, bottomMargin=1*inch)
            styles = getSampleStyleSheet()
            
            # Fallback to default ReportLab font for error document
            default_font = "Helvetica"
            if not pdfmetrics.getFont(default_font):
                pdfmetrics.registerFont(TTFont(default_font, "Helvetica")) # Ensure Helvetica is registered
            
            title_style = ParagraphStyle(
                'ErrorTitle',
                parent=styles['Title'],
                fontName=default_font,
                fontSize=16,
                textColor='red'
            )
            heading_style = ParagraphStyle(
                'ErrorHeading',
                parent=styles['Heading2'],
                fontName=default_font,
                fontSize=14,
                textColor='darkred'
            )
            body_style = ParagraphStyle(
                'ErrorNormal',
                parent=styles['Normal'],
                fontName=default_font,
                fontSize=11,
                textColor='black'
            )

            simple_content = [
                Paragraph("PDF Translation Result (Error Fallback)", title_style),
                Spacer(1, 12),
                Paragraph(f"Original File: {filename}", body_style),
                Paragraph(f"Target Language: {target_language}", body_style),
                Spacer(1, 20),
                Paragraph("Error Details:", heading_style),
                Paragraph(f"An error occurred during PDF generation: {e}", body_style),
                Spacer(1, 20),
                Paragraph("Original Text:", heading_style),
                Paragraph(full_original_text, body_style),
                Spacer(1, 20),
                Paragraph("Translated Text:", heading_style),
                Paragraph(full_translated_text, body_style)
            ]
            doc.build(simple_content)
            buffer.seek(0)
            logger.info("Created fallback PDF due to error in enhanced PDF generation")
            return buffer
        except Exception as fallback_error:
            logger.error(f"Failed to create fallback PDF: {fallback_error}")
            raise





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
    """Enhanced PDF translation with duplicate removal, font support, and download option"""
    global tokenizer, model, ip, DEVICE
    
    if not all([tokenizer, model, ip]):
        raise HTTPException(status_code=503, detail="Models are still loading. Please try again in a moment.")
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Read PDF content
        pdf_content = await file.read()
        
        # Enhanced text extraction with multiple methods, including layout data
        extracted_text, extraction_method, layout_data = extract_text_from_pdf(pdf_content, max_pages=2)
        logger.info(f"Text extracted using: {extraction_method}")
        
        # If layout_data is available, translate each text element individually
        if layout_data and any(p.text_elements for p in layout_data):
            all_elements = []
            for page in layout_data:
                all_elements.extend(page.text_elements)
            # Remove empty/duplicate texts
            seen = set()
            element_texts = []
            element_indices = []
            for idx, el in enumerate(all_elements):
                t = el.text.strip()
                if t and t not in seen:
                    element_texts.append(t)
                    element_indices.append(idx)
                    seen.add(t)
                else:
                    element_indices.append(None)  # Mark as duplicate/empty
            # Translate in optimized batches
            batch_size = 10  # Proven optimal size for element-wise processing
            translated_texts = []
            for i in range(0, len(element_texts), batch_size):
                batch = element_texts[i:i+batch_size]
                try:
                    pre = ip.preprocess_batch(batch, src_lang="eng_Latn", tgt_lang=target_language)
                    inputs = tokenizer(
                        pre,
                        truncation=True,
                        padding="longest",
                        return_tensors="pt",
                        return_attention_mask=True,
                        max_length=512,  # Standard input length
                    ).to(DEVICE)
                    with torch.no_grad():
                        generated_tokens = model.generate(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            use_cache=False,
                            min_length=0,
                            max_length=200,  # Balanced for quality vs speed
                            num_beams=3,     # Keep quality with reasonable speed
                            num_return_sequences=1,
                            do_sample=False,
                            early_stopping=True,  # Stop early when possible
                        )
                    decoded = tokenizer.batch_decode(
                        generated_tokens.detach().cpu().tolist(),
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )
                    batch_trans = ip.postprocess_batch(decoded, lang=target_language)
                    translated_texts.extend(batch_trans)
                except Exception as batch_error:
                    logger.error(f"Error translating element batch: {batch_error}")
                    translated_texts.extend(["[Translation failed]"] * len(batch))
            # Map translations back to all_elements order
            translated_elements = []
            trans_idx = 0
            for idx in element_indices:
                if idx is not None:
                    translated_elements.append(translated_texts[trans_idx])
                    trans_idx += 1
                else:
                    translated_elements.append("")
            # For download and PDF generation, pass these lists
            return {
                "success": True,
                "filename": file.filename,
                "pages_processed": len(layout_data),
                "extracted_text": extracted_text,
                "translated_text": " ".join(translated_elements),
                "target_language": target_language,
                "source_language": "eng_Latn",
                "language_code": target_language,
                "extraction_method": extraction_method,
                "text_length": len(extracted_text),
                "chunks_processed": len(translated_elements),
                "memory_management": "element-wise",
                "duplicates_removed": True,
                "download_available": True,
                "layout_data_available": True,
                "font_support": {
                    "enabled": font_system_initialized,
                },
                "original_text_elements": [el.text for el in all_elements],
                "translated_text_elements": translated_elements,
                "layout_data": [p.dict() for p in layout_data] if layout_data else []
            }
        # Fallback: old chunk-based translation
        # ... existing code ...

        # Clean up and remove duplicates from extracted text
        extracted_text = remove_duplicates_from_text(extracted_text.strip())
        
        # Use word-based processing like Live Preview for optimal performance
        logger.info(f"Text length: {len(extracted_text)} characters. Using word-based processing for maximum speed.")
        
        # Split into words (similar to Live Preview approach)
        words = extracted_text.split()
        # Filter and clean words
        text_chunks = [word.strip() for word in words if word.strip() and len(word.strip()) > 1]
        
        logger.info(f"Split into {len(text_chunks)} word elements")
        
        # Translate words in small batches (proven optimal from Live Preview)
        all_translations = []
        batch_size = 5  # Proven optimal size from Live Preview
        
        # Simple optimized sequential processing (avoiding threading issues)
        logger.info(f"Processing {len(text_chunks)} chunks with optimized sequential batching")
        
        # Use larger batch size for better performance without threading complexity
        optimized_batch_size = batch_size * 2  # Double the batch size for speed
        
        for i in range(0, len(text_chunks), optimized_batch_size):
            batch_chunks = text_chunks[i:i + optimized_batch_size]
            
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
                
                # Optimized memory clearing
                if DEVICE == "cuda" and i % (optimized_batch_size * 3) == 0:
                    torch.cuda.empty_cache()
                
                batch_num = i // optimized_batch_size + 1
                total_batches = (len(text_chunks) + optimized_batch_size - 1) // optimized_batch_size
                logger.info(f"Optimized: Processed batch {batch_num}/{total_batches}")
                
            except Exception as batch_error:
                logger.error(f"Error processing optimized batch {i//optimized_batch_size + 1}: {batch_error}")
                all_translations.extend(["[Translation failed]"] * len(batch_chunks))
        
        logger.info(f"Optimized sequential processing completed: {len(all_translations)} translations generated")
        
        # Join translated words back together with spaces
        translated_text = ' '.join(all_translations)

        # Remove duplicates from translated text as well
        translated_text = remove_duplicates_from_text(translated_text)

        # Get font information for the target language
        font_info = {}
        if font_system_initialized:
            try:
                language_mapper = get_language_mapper()
                font_info = language_mapper.get_font_info_for_language(target_language)
                logger.info(f"Font info for {target_language}: {font_info['selected_font']}")
            except Exception as e:
                logger.warning(f"Could not get font info for {target_language}: {e}")

        return {
            "success": True,
            "filename": file.filename,
            "pages_processed": 2,
            "extracted_text": extracted_text,
            "translated_text": translated_text,
            "target_language": target_language,
            "source_language": "eng_Latn",
            "language_code": target_language,  # Pass language code for PDF generation
            "extraction_method": extraction_method,
            "text_length": len(extracted_text),
            "chunks_processed": len(text_chunks),
            "memory_management": "chunking" if len(extracted_text) > 1000 else "standard",
            "duplicates_removed": True,
            "download_available": True,
            "layout_data_available": bool(layout_data),
            "font_support": {
                "enabled": font_system_initialized,
                "font_info": font_info
            },
            "original_text_chunks": text_chunks,
            "translated_text_chunks": all_translations,
            "layout_data": [p.dict() for p in layout_data] if layout_data else [] # Include layout data
        }

    except Exception as e:
        logger.error(f"❌ PDF translation error: {e}")
        raise HTTPException(status_code=500, detail=f"PDF translation failed: {str(e)}")


@app.post("/download-translated-pdf")
async def download_translated_pdf_endpoint(
    filename: str = Form(...),
    original_text_chunks_json: str = Form(...),
    translated_text_chunks_json: str = Form(...),
    target_language: str = Form(...),
    layout_data_json: str = Form(...), # Receive layout data as JSON string
    language_code: Optional[str] = Form(None)
):
    logger.info(f"Download PDF endpoint called for {filename}")

    try:
        original_text_chunks = json.loads(original_text_chunks_json)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding original_text_chunks_json: {e}")
        raise HTTPException(status_code=422, detail=f"Invalid original_text_chunks_json: {e}")

    try:
        translated_text_chunks = json.loads(translated_text_chunks_json)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding translated_text_chunks_json: {e}")
        raise HTTPException(status_code=422, detail=f"Invalid translated_text_chunks_json: {e}")
    
    # Deserialize layout_data
    try:
        # Directly deserialize to a list of dicts, then convert to PdfLayoutData
        layout_data_dicts = json.loads(layout_data_json)
        layout_data = [PdfLayoutData(**item) for item in layout_data_dicts]
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding layout_data_json: {e}")
        raise HTTPException(status_code=422, detail=f"Invalid layout_data_json: {e}")
    except TypeError as e:
        logger.error(f"Type error during layout_data deserialization: {e}")
        raise HTTPException(status_code=422, detail=f"Invalid layout_data structure: {e}")

    effective_language_code = language_code if language_code else target_language

    try:
        pdf_buffer = create_pdf_from_text(
            original_text_chunks=original_text_chunks,
            translated_text_chunks=translated_text_chunks,
            filename=filename,
            target_language=target_language,
            language_code=effective_language_code,
            layout_data=layout_data
        )

        pdf_buffer.seek(0)
        headers = {
            "Content-Disposition": f"attachment; filename=\"translated_{filename}\""
        }
        return StreamingResponse(iter([pdf_buffer.getvalue()]), media_type="application/pdf", headers=headers)
    except Exception as e:
        logger.error(f"Error in download_translated_pdf_endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {e}")


@app.post("/translate-and-download-pdf")
async def translate_and_download_pdf(
    file: UploadFile = File(...),
    target_language: str = Form(...)
):
    """Directly translate a PDF and return the translated PDF as a download, preserving layout."""
    global tokenizer, model, ip, DEVICE
    if not all([tokenizer, model, ip]):
        raise HTTPException(status_code=503, detail="Models are still loading. Please try again in a moment.")
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    try:
        pdf_content = await file.read()
        extracted_text, extraction_method, layout_data = extract_text_from_pdf(pdf_content, max_pages=2)
        if not layout_data or not any(p.text_elements for p in layout_data):
            raise HTTPException(status_code=422, detail="Could not extract layout from PDF. Only text-based PDFs are supported for layout-preserving translation.")
        # Flatten all text elements
        all_elements = []
        for page in layout_data:
            all_elements.extend(page.text_elements)
        # Remove empty/duplicate texts
        seen = set()
        element_texts = []
        element_indices = []
        for idx, el in enumerate(all_elements):
            t = el.text.strip()
            if t and t not in seen:
                element_texts.append(t)
                element_indices.append(idx)
                seen.add(t)
            else:
                element_indices.append(None)
        # Translate in batches
        batch_size = 10
        translated_texts = []
        for i in range(0, len(element_texts), batch_size):
            batch = element_texts[i:i+batch_size]
            try:
                pre = ip.preprocess_batch(batch, src_lang="eng_Latn", tgt_lang=target_language)
                inputs = tokenizer(
                    pre,
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
                batch_trans = ip.postprocess_batch(decoded, lang=target_language)
                translated_texts.extend(batch_trans)
            except Exception as batch_error:
                logger.error(f"Error translating element batch: {batch_error}")
                translated_texts.extend(["[Translation failed]"] * len(batch))
        # Map translations back to all_elements order
        translated_elements = []
        trans_idx = 0
        for idx in element_indices:
            if idx is not None:
                translated_elements.append(translated_texts[trans_idx])
                trans_idx += 1
            else:
                translated_elements.append("")
        # Generate the PDF directly
        pdf_buffer = create_pdf_from_text(
            original_text_chunks=[el.text for el in all_elements],
            translated_text_chunks=translated_elements,
            filename=file.filename,
            target_language=target_language,
            language_code=target_language,
            layout_data=layout_data
        )
        pdf_buffer.seek(0)
        headers = {
            "Content-Disposition": f"attachment; filename=translated_{file.filename}"
        }
        return StreamingResponse(iter([pdf_buffer.getvalue()]), media_type="application/pdf", headers=headers)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in direct PDF translation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate translated PDF: {e}")


@app.post("/translate-pdf-live-preview")
async def translate_pdf_live_preview(
    file: UploadFile = File(...),
    target_language: str = Form(...)
):
    """Stream translation results element-by-element for live preview (SSE) with layout preservation."""
    global tokenizer, model, ip, DEVICE
    if not all([tokenizer, model, ip]):
        raise HTTPException(status_code=503, detail="Models are still loading. Please try again in a moment.")
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    pdf_content = await file.read()
    # Process ALL pages for large files, not just first 2
    extracted_text, extraction_method, layout_data = extract_text_from_pdf(pdf_content, max_pages=10)
    
    # Handle both layout-based and simple text-based PDFs
    if layout_data and any(p.text_elements for p in layout_data):
        # Layout-based processing (preferred)
        all_elements = []
        for page in layout_data:
            all_elements.extend(page.text_elements)
        # Remove empty/duplicate texts
        seen = set()
        element_texts = []
        element_indices = []
        for idx, el in enumerate(all_elements):
            t = el.text.strip()
            if t and t not in seen:
                element_texts.append(t)
                element_indices.append(idx)
                seen.add(t)
            else:
                element_indices.append(None)
    else:
        # Fallback to simple text-based processing with proper document layout
        logger.info(f"Live preview: No layout data available, creating proper document layout for simple text")
        
        # Create proper document structure with paragraphs
        paragraphs = [p.strip() for p in extracted_text.split('\n\n') if p.strip()]
        if not paragraphs:
            paragraphs = [extracted_text] if extracted_text.strip() else ["No text found"]
        
        element_texts = []
        element_indices = []
        all_elements = []
        
        # Document layout parameters
        page_width = 500
        margin_x = 50
        margin_y = 100
        line_height = 18
        paragraph_spacing = 30
        current_y = margin_y
        
        for paragraph_idx, paragraph in enumerate(paragraphs):
            # Determine if this is a title, heading, or body text
            is_title = paragraph_idx == 0 and len(paragraph) < 100
            is_heading = paragraph.startswith('What is') or paragraph.startswith('How') or paragraph.startswith('Why')
            
            if is_title:
                # Center the title
                font_size = 20
                font_weight = 'bold'
                text_align = 'center'
                current_y += 20
            elif is_heading:
                # Left-align headings
                font_size = 16
                font_weight = 'bold'
                text_align = 'left'
                current_y += paragraph_spacing
            else:
                # Justify body text
                font_size = 12
                font_weight = 'normal'
                text_align = 'justify'
                current_y += paragraph_spacing
            
            # Split paragraph into words for word-by-word translation
            words = paragraph.split()
            current_x = margin_x
            current_line_y = current_y
            
            for word_idx, word in enumerate(words):
                if word.strip():
                    # Calculate word width (approximate)
                    word_width = len(word) * 8 + 5  # Approximate character width
                    
                    # Check if word fits on current line
                    if current_x + word_width > page_width - margin_x:
                        # Move to next line
                        current_x = margin_x
                        current_line_y += line_height
                    
                    element_texts.append(word)
                    element_indices.append(len(element_texts) - 1)
                    
                    mock_element = {
                        "text": word,
                        "bbox": [current_x, current_line_y, current_x + word_width, current_line_y + line_height],
                        "font_name": "Helvetica",
                        "font_size": font_size,
                        "color": [0, 0, 0],
                        "is_bold": font_weight == 'bold',
                        "is_italic": False,
                        "text_align": text_align,
                        "paragraph_index": paragraph_idx,
                        "word_index": word_idx,
                        "is_title": is_title,
                        "is_heading": is_heading
                    }
                    all_elements.append(mock_element)
                    
                    # Move to next word position
                    current_x += word_width + 5  # 5px space between words
            
            # Update current_y for next paragraph
            current_y = current_line_y + line_height
    async def event_generator() -> AsyncGenerator[str, None]:
        # First, send layout information
        layout_info = {
            "type": "layout_info",
            "total_elements": len(element_texts),
            "layout_data": []
        }
        
        # Add layout data for each element
        for i, (text, idx) in enumerate(zip(element_texts, element_indices)):
            if idx is not None and all_elements:
                element = all_elements[idx]
                layout_info["layout_data"].append({
                    "index": i,
                    "text": text,
                    "bbox": element.bbox if hasattr(element, 'bbox') else element.get('bbox', [0, 0, 100, 20]),
                    "font_name": element.font_name if hasattr(element, 'font_name') else element.get('font_name', 'Helvetica'),
                    "font_size": element.font_size if hasattr(element, 'font_size') else element.get('font_size', 12),
                    "color": element.color if hasattr(element, 'color') else element.get('color', [0, 0, 0]),
                    "is_bold": element.is_bold if hasattr(element, 'is_bold') else element.get('is_bold', False),
                    "is_italic": element.is_italic if hasattr(element, 'is_italic') else element.get('is_italic', False)
                })
            else:
                # Mock layout for simple text
                layout_info["layout_data"].append({
                    "index": i,
                    "text": text,
                    "bbox": [50 + (i * 60), 100, 50 + (i * 60) + 50, 120],
                    "font_name": "Helvetica",
                    "font_size": 12,
                    "color": [0, 0, 0],
                    "is_bold": False,
                    "is_italic": False
                })
        
        yield f"data: {json.dumps(layout_info)}\n\n"
        
        # Then stream translations word by word
        batch_size = 5  # Optimal batch size for large files
        trans_idx = 0
        total_batches = (len(element_texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(element_texts), batch_size):
            batch = element_texts[i:i+batch_size]
            try:
                pre = ip.preprocess_batch(batch, src_lang="eng_Latn", tgt_lang=target_language)
                inputs = tokenizer(
                    pre,
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
                batch_trans = ip.postprocess_batch(decoded, lang=target_language)
            except Exception as batch_error:
                logger.error(f"Error translating element batch: {batch_error}")
                batch_trans = ["[Translation failed]"] * len(batch)
            
            # Stream each element in the batch with layout info
            for j, orig in enumerate(batch):
                element_idx = i + j
                layout_element = layout_info["layout_data"][element_idx] if element_idx < len(layout_info["layout_data"]) else None
                
                event = {
                    "type": "translation_update",
                    "element_index": trans_idx,
                    "original_text": orig,
                    "translated_text": batch_trans[j],
                    "layout": layout_element,
                    "progress": {
                        "current": trans_idx + 1,
                        "total": len(element_texts),
                        "percentage": round(((trans_idx + 1) / len(element_texts)) * 100, 1),
                        "batch_progress": f"Batch {i//batch_size + 1}/{total_batches}"
                    }
                }
                yield f"data: {json.dumps(event)}\n\n"
                trans_idx += 1
                
                # Add small delay for better visual effect (reduced for large files)
                import asyncio
                await asyncio.sleep(0.05)
        
        # Send completion event
        completion_event = {
            "type": "translation_complete",
            "total_translated": trans_idx
        }
        yield f"data: {json.dumps(completion_event)}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")

app.include_router(pdf_translation_router)
app.include_router(pdf_simple_router)