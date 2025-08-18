from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
import logging
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="IndicTrans2 Translation API", version="1.0.0")

# Global holders
tokenizer = None
model = None
ip = None
DEVICE = None


class TranslationRequest(BaseModel):
    sentences: List[str]
    src_lang: str
    tgt_lang: str


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

        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        logger.info("✅ Tokenizer loaded")

        logger.info("Loading model...")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        ).to(DEVICE)
        logger.info("✅ Model loaded")

        logger.info("Loading IndicProcessor...")
        ip = IndicProcessor(inference=True)
        logger.info("✅ IndicProcessor loaded")

        logger.info("🎉 All components loaded successfully!")

    except Exception as e:
        logger.error(f"❌ Failed to load components: {e}")
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
