from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import torch
import logging
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor
import PyPDF2
import io

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
            torch_dtype=torch.float32,
            cache_dir=cache_dir,
            local_files_only=False
        ).to(DEVICE)
        logger.info("✅ Model loaded")

        logger.info("Loading IndicProcessor...")
        ip = IndicProcessor(inference=True)
        logger.info("✅ IndicProcessor loaded")

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

        # Step 3: Generate (⚡ FIXED)
        with torch.no_grad():
            generated_tokens = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,  # avoids "0 layers" cache bug
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
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

        # Step 3: Generate
        with torch.no_grad():
            generated_tokens = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
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


@app.post("/translate-pdf")
async def translate_pdf(
    file: UploadFile = File(...),
    target_language: str = Form(...)
):
    """Extract text from PDF and translate it"""
    global tokenizer, model, ip, DEVICE
    
    if not all([tokenizer, model, ip]):
        raise HTTPException(status_code=503, detail="Models are still loading. Please try again in a moment.")
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Read PDF content
        pdf_content = await file.read()
        pdf_file = io.BytesIO(pdf_content)
        
        # Extract text from PDF
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        extracted_text = ""
        
        # Limit to first 2 pages as requested
        max_pages = min(len(pdf_reader.pages), 2)
        
        for page_num in range(max_pages):
            page = pdf_reader.pages[page_num]
            extracted_text += page.extract_text() + "\n"
        
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")
        
        # Clean up the extracted text
        extracted_text = extracted_text.strip()
        
        # Split text into sentences for better translation
        sentences = [sent.strip() for sent in extracted_text.split('.') if sent.strip()]
        if not sentences:
            sentences = [extracted_text]
        
        # Translate the text
        batch = ip.preprocess_batch(
            sentences,
            src_lang="eng_Latn",
            tgt_lang=target_language,
        )

        inputs = tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)

        with torch.no_grad():
            generated_tokens = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        decoded = tokenizer.batch_decode(
            generated_tokens.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        translations = ip.postprocess_batch(decoded, lang=target_language)
        
        # Join translated sentences back together
        translated_text = '. '.join(translations) if len(translations) > 1 else translations[0]

        return {
            "success": True,
            "filename": file.filename,
            "pages_processed": max_pages,
            "extracted_text": extracted_text,
            "translated_text": translated_text,
            "target_language": target_language,
            "source_language": "eng_Latn"
        }

    except Exception as e:
        logger.error(f"❌ PDF translation error: {e}")
        raise HTTPException(status_code=500, detail=f"PDF translation failed: {str(e)}")


@app.get("/languages")
def get_supported_languages():
    """Get list of supported languages"""
    return {
        "supported_languages": {
            "asm_Beng": "Assamese",
            "ben_Beng": "Bengali", 
            "brx_Deva": "Bodo",
            "doi_Deva": "Dogri",
            "gom_Deva": "Goan Konkani",
            "guj_Gujr": "Gujarati",
            "hin_Deva": "Hindi",
            "hne_Deva": "Chhattisgarhi",
            "kan_Knda": "Kannada",
            "kas_Arab": "Kashmiri (Arabic)",
            "kas_Deva": "Kashmiri (Devanagari)",
            "kha_Latn": "Khasi",
            "lus_Latn": "Mizo",
            "mai_Deva": "Maithili",
            "mal_Mlym": "Malayalam",
            "mar_Deva": "Marathi",
            "mni_Beng": "Manipuri (Bengali)",
            "mni_Mtei": "Manipuri (Meetei Mayek)",
            "npi_Deva": "Nepali",
            "ory_Orya": "Odia",
            "pan_Guru": "Punjabi",
            "san_Deva": "Sanskrit",
            "sat_Olck": "Santali",
            "snd_Arab": "Sindhi (Arabic)",
            "snd_Deva": "Sindhi (Devanagari)",
            "tam_Taml": "Tamil",
            "tel_Telu": "Telugu",
            "urd_Arab": "Urdu"
        },
        "source_language": "eng_Latn"
    }
