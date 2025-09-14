from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
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

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logging.getLogger('font_management').setLevel(logging.DEBUG)

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
        
        logger.info(f"📊 Font registration summary:")
        logger.info(f"  - Total fonts: {summary['total_fonts']}")
        logger.info(f"  - Registered fonts: {summary['registered_fonts']}")
        logger.info(f"  - Font families: {summary['total_families']}")
        logger.info(f"  - Available families: {summary['available_families']}")
        
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
        logger.info(f"🌐 Language mapping summary:")
        logger.info(f"  - Total language mappings: {mapping_summary['total_language_mappings']}")
        logger.info(f"  - Available mappings: {mapping_summary['available_language_mappings']}")
        logger.info(f"  - Supported languages: {len(mapping_summary['supported_languages'])}")
        
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
                original_text="Original English text for testing",
                translated_text=sample_text,
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


def create_pdf_from_text(original_text: str, translated_text: str, filename: str, target_language: str, language_code: str = None) -> io.BytesIO:
    """
    Create a PDF with original and translated text using appropriate fonts for the target language.
    
    Args:
        original_text: Original text content
        translated_text: Translated text content
        filename: Original filename for reference
        target_language: Target language name for display
        language_code: Language code for font selection (e.g., 'hin_Deva', 'urd_Arab')
    
    Returns:
        BytesIO buffer containing the generated PDF
    """
    buffer = io.BytesIO()
    
    # Create PDF document
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*inch, bottomMargin=1*inch)
    
    try:
        # Get font system components if available
        if font_system_initialized:
            logger.info(f"📄 Starting PDF generation with font support")
            logger.info(f"📋 PDF Parameters: filename='{filename}', target_language='{target_language}', language_code='{language_code}'")
            
            style_factory = get_style_factory()
            language_mapper = get_language_mapper()
            
            # Use provided language_code or try to derive from target_language
            effective_language_code = language_code or target_language
            logger.info(f"🔤 Effective language code for font selection: '{effective_language_code}'")
            
            # Get detailed font info before style creation
            font_info = language_mapper.get_font_info_for_language(effective_language_code)
            logger.info(f"🎯 Font selection details:")
            logger.info(f"  - Language code: {font_info['language_code']}")
            logger.info(f"  - Normalized code: {font_info['normalized_code']}")
            logger.info(f"  - Preferred font: {font_info['preferred_font']}")
            logger.info(f"  - Selected font: {font_info['selected_font']}")
            logger.info(f"  - Is supported: {font_info['is_supported']}")
            logger.info(f"  - Is preferred available: {font_info['is_preferred_available']}")
            logger.info(f"  - Fallback fonts: {font_info['fallback_fonts']}")
            
            # Get language-specific styles
            try:
                logger.info(f"🎨 Creating language-specific styles for '{effective_language_code}'...")
                language_styles = style_factory.create_styles_for_language(effective_language_code)
                
                # Extract individual styles
                title_style = language_styles['title']
                heading_style = language_styles['heading1']
                body_style = language_styles['body']
                translated_style = language_styles['body']
                
                logger.info(f"✅ Successfully created {len(language_styles)} language-specific styles")
                logger.info(f"📝 Style details:")
                logger.info(f"  - Title style font: {title_style.fontName}")
                logger.info(f"  - Heading style font: {heading_style.fontName}")
                logger.info(f"  - Body style font: {body_style.fontName}")
                logger.info(f"  - Translated text style font: {translated_style.fontName}")
                
            except Exception as e:
                logger.error(f"❌ Failed to create language-specific styles for '{effective_language_code}': {e}")
                logger.warning(f"🔄 Falling back to default styles...")
                
                # Fall back to default styles
                language_styles = style_factory.create_fallback_styles()
                title_style = language_styles['title']
                heading_style = language_styles['heading1']
                body_style = language_styles['body']
                translated_style = language_styles['body']
                
                logger.info(f"✅ Created fallback styles:")
                logger.info(f"  - Title style font: {title_style.fontName}")
                logger.info(f"  - Heading style font: {heading_style.fontName}")
                logger.info(f"  - Body style font: {body_style.fontName}")
                logger.info(f"  - Translated text style font: {translated_style.fontName}")
        else:
            # Font system not initialized, use default ReportLab styles
            logger.warning("⚠️ Font system not initialized, using default ReportLab styles")
            logger.warning(f"📋 PDF will be generated without Indian language font support")
            logger.info(f"🔄 Using system fonts for language: '{target_language}'")
            styles = getSampleStyleSheet()
            
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
            
            translated_style = body_style
        
        # Build content
        content = []
        
        # Title
        content.append(Paragraph("PDF Translation Result", title_style))
        content.append(Spacer(1, 12))
        
        # File info
        content.append(Paragraph(f"<b>Original File:</b> {filename}", body_style))
        content.append(Paragraph(f"<b>Target Language:</b> {target_language}", body_style))
        if language_code:
            content.append(Paragraph(f"<b>Language Code:</b> {language_code}", body_style))
        content.append(Spacer(1, 20))
        
        # Original text section
        content.append(Paragraph("Original Text", heading_style))
        # Split long text into paragraphs
        original_paragraphs = original_text.split('\n\n') if '\n\n' in original_text else [original_text]
        for para in original_paragraphs:
            if para.strip():
                # Use regular body style for original English text
                content.append(Paragraph(para.strip(), body_style))
        
        content.append(Spacer(1, 20))
        
        # Translated text section
        content.append(Paragraph("Translated Text", heading_style))
        # Split long text into paragraphs
        translated_paragraphs = translated_text.split('\n\n') if '\n\n' in translated_text else [translated_text]
        
        # Use standard ReportLab Paragraphs for all translated text
        for para in translated_paragraphs:
            if para.strip():
                content.append(Paragraph(para.strip(), translated_style))
        
        # Build PDF
        logger.info(f"🔨 Building PDF document with {len(content)} elements...")
        doc.build(content)
        buffer.seek(0)
        
        # Log final PDF creation details
        pdf_size = len(buffer.getvalue())
        logger.info(f"✅ Successfully created PDF:")
        logger.info(f"  - Elements: {len(content)}")
        logger.info(f"  - Size: {pdf_size} bytes ({pdf_size / 1024:.1f} KB)")
        logger.info(f"  - Language: {target_language}")
        logger.info(f"  - Font system used: {'Yes' if font_system_initialized else 'No'}")
        
        return buffer
        
    except Exception as e:
        logger.error(f"Error creating PDF: {e}")
        # Create a simple fallback PDF in case of errors
        try:
            styles = getSampleStyleSheet()
            simple_content = [
                Paragraph("PDF Translation Result", styles['Title']),
                Spacer(1, 12),
                Paragraph(f"Original File: {filename}", styles['Normal']),
                Paragraph(f"Target Language: {target_language}", styles['Normal']),
                Spacer(1, 20),
                Paragraph("Original Text", styles['Heading2']),
                Paragraph(original_text, styles['Normal']),
                Spacer(1, 20),
                Paragraph("Translated Text", styles['Heading2']),
                Paragraph(translated_text, styles['Normal'])
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
            "font_support": {
                "enabled": font_system_initialized,
                "font_info": font_info
            }
        }

    except Exception as e:
        logger.error(f"❌ PDF translation error: {e}")
        raise HTTPException(status_code=500, detail=f"PDF translation failed: {str(e)}")


@app.post("/download-translated-pdf")
async def download_translated_pdf_endpoint(
    original_text: str = Form(...),
    translated_text: str = Form(...),
    filename: str = Form(...),
    target_language: str = Form(...),
    language_code: Optional[str] = Form(None)
):
    """Generate and download translated PDF with enhanced font support"""
    try:
        # Use target_language as language_code if not provided
        effective_language_code = language_code or target_language
        
        # Log font selection for debugging
        if font_system_initialized:
            try:
                language_mapper = get_language_mapper()
                font_info = language_mapper.get_font_info_for_language(effective_language_code)
                logger.info(f"Generating PDF for {effective_language_code} using font: {font_info['selected_font']}")
            except Exception as e:
                logger.warning(f"Could not get font info for PDF generation: {e}")
        
        # Create PDF with both original and translated text using font support
        pdf_buffer = create_pdf_from_text(
            original_text=original_text,
            translated_text=translated_text,
            filename=filename,
            target_language=target_language,
            language_code=effective_language_code
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