from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any, AsyncGenerator
import json
import io
import logging
from font_management import get_language_mapper
from translation_core import extract_text_from_pdf, ip, tokenizer, model, DEVICE
import torch
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/translate-and-download-pdf")
async def translate_and_download_pdf(
    file: UploadFile = File(...),
    target_language: str = Form(...)
):
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
        all_elements = []
        for page in layout_data:
            all_elements.extend(page.text_elements)
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
        translated_elements = []
        trans_idx = 0
        for idx in element_indices:
            if idx is not None:
                translated_elements.append(translated_texts[trans_idx])
                trans_idx += 1
            else:
                translated_elements.append("")
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

@router.post("/translate-pdf-live-preview")
async def translate_pdf_live_preview(
    file: UploadFile = File(...),
    target_language: str = Form(...)
):
    global tokenizer, model, ip, DEVICE
    if not all([tokenizer, model, ip]):
        raise HTTPException(status_code=503, detail="Models are still loading. Please try again in a moment.")
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    pdf_content = await file.read()
    extracted_text, extraction_method, layout_data = extract_text_from_pdf(pdf_content, max_pages=2)
    if not layout_data or not any(p.text_elements for p in layout_data):
        raise HTTPException(status_code=422, detail="Could not extract layout from PDF. Only text-based PDFs are supported for layout-preserving translation.")
    all_elements = []
    for page in layout_data:
        all_elements.extend(page.text_elements)
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
    async def event_generator() -> AsyncGenerator[str, None]:
        # First, send layout information
        layout_info_event = {
            "type": "layout_info",
            "total_elements": len(element_texts),
            "layout_data": [
                {
                    "index": idx,
                    "text": el.text,
                    "bbox": list(el.bbox),
                    "font_name": el.font_name,
                    "font_size": el.font_size,
                    "color": list(el.color),
                    "is_bold": el.is_bold,
                    "is_italic": el.is_italic,
                    "paragraph_index": idx // 10,  # Simple paragraph grouping
                    "word_index": idx,
                    "is_title": el.font_size > 16,
                    "is_heading": el.font_size > 14 and el.font_size <= 16
                }
                for idx, el in enumerate(all_elements) if element_indices[idx] is not None
            ]
        }
        yield f"data: {json.dumps(layout_info_event)}\n\n"
        
        batch_size = 5
        trans_idx = 0
        total_elements = len(element_texts)
        
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
            
            for j, orig in enumerate(batch):
                # Find the corresponding layout element
                layout_element = None
                actual_element_idx = None
                for idx, el in enumerate(all_elements):
                    if element_indices[idx] == trans_idx and el.text.strip() == orig:
                        layout_element = {
                            "index": idx,
                            "text": el.text,
                            "bbox": list(el.bbox),
                            "font_name": el.font_name,
                            "font_size": el.font_size,
                            "color": list(el.color),
                            "is_bold": el.is_bold,
                            "is_italic": el.is_italic,
                            "paragraph_index": idx // 10,
                            "word_index": idx,
                            "is_title": el.font_size > 16,
                            "is_heading": el.font_size > 14 and el.font_size <= 16
                        }
                        actual_element_idx = idx
                        break
                
                event = {
                    "type": "translation_update",
                    "element_index": trans_idx,
                    "original_text": orig,
                    "translated_text": batch_trans[j],
                    "layout": layout_element,
                    "progress": {
                        "current": trans_idx + 1,
                        "total": total_elements,
                        "percentage": ((trans_idx + 1) / total_elements) * 100,
                        "batch_progress": f"Batch {i//batch_size + 1}/{(len(element_texts) + batch_size - 1)//batch_size}"
                    }
                }
                yield f"data: {json.dumps(event)}\n\n"
                trans_idx += 1
        
        # Send completion event
        completion_event = {
            "type": "translation_complete",
            "total_translated": trans_idx,
            "message": "Translation completed successfully"
        }
        yield f"data: {json.dumps(completion_event)}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.post("/translate-and-download-docx")
async def translate_and_download_docx(
    file: UploadFile = File(...),
    target_language: str = Form(...)
):
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
        all_elements = []
        for page in layout_data:
            all_elements.extend(page.text_elements)
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
        translated_elements = []
        trans_idx = 0
        for idx in element_indices:
            if idx is not None:
                translated_elements.append(translated_texts[trans_idx])
                trans_idx += 1
            else:
                translated_elements.append("")
        # Build the .docx file
        doc = Document()
        doc.add_heading(f"Translated PDF: {file.filename}", 0)
        last_font_size = None
        for i, el in enumerate(all_elements):
            translated_text = translated_elements[i] if i < len(translated_elements) else ""
            if not translated_text.strip():
                continue
            p = doc.add_paragraph()
            run = p.add_run(translated_text)
            # Font size
            font_size = int(el.font_size) if hasattr(el, 'font_size') and el.font_size else 12
            run.font.size = Pt(font_size)
            # Bold/Italic
            run.bold = getattr(el, 'is_bold', False)
            run.italic = getattr(el, 'is_italic', False)
            # Color (convert 0-1 RGB to 0-255)
            if hasattr(el, 'color') and el.color:
                r, g, b = [int(255 * c) for c in el.color]
                run.font.color.rgb = RGBColor(r, g, b)
            # Alignment (left by default)
            p.alignment = WD_PARAGRAPH_ALIGNMENT.LEFT
        # Save to buffer
        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        headers = {
            "Content-Disposition": f"attachment; filename=translated_{file.filename.replace('.pdf', '.docx')}"
        }
        return StreamingResponse(iter([buffer.getvalue()]), media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", headers=headers)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in direct DOCX translation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate translated DOCX: {e}")

@router.post("/download-translated-pdf")
async def download_translated_pdf(
    file: UploadFile = File(...),
    target_language: str = Form(...),
    original_text_chunks_json: str = Form(...),
    translated_text_chunks_json: str = Form(...),
    layout_data_json: str = Form(...),
    language_code: str = Form(...)
):
    """Download translated document as PDF format from live preview"""
    try:
        # Parse the JSON data
        original_text_chunks = json.loads(original_text_chunks_json)
        translated_text_chunks = json.loads(translated_text_chunks_json)
        layout_data_raw = json.loads(layout_data_json)
        
        # Convert layout data to proper format for create_pdf_from_text
        from translation_core import PdfLayoutData, PdfTextElement
        layout_data = []
        
        # Group elements by page (assuming single page for now)
        page_layout = PdfLayoutData(page_number=1, width=600, height=800, text_elements=[])
        
        for i, element_data in enumerate(layout_data_raw):
            if isinstance(element_data, dict):
                text_element = PdfTextElement(
                    text=original_text_chunks[i] if i < len(original_text_chunks) else element_data.get('text', ''),
                    bbox=tuple(element_data.get('bbox', [0, 0, 100, 20])),
                    font_name=element_data.get('font_name', 'Arial'),
                    font_size=element_data.get('font_size', 12),
                    color=tuple(element_data.get('color', [0, 0, 0])),
                    is_bold=element_data.get('is_bold', False),
                    is_italic=element_data.get('is_italic', False)
                )
                page_layout.text_elements.append(text_element)
        
        layout_data.append(page_layout)
        
        # Import the function from main module
        from main import create_pdf_from_text
        
        # Create PDF using the existing function
        pdf_buffer = create_pdf_from_text(
            original_text_chunks=original_text_chunks,
            translated_text_chunks=translated_text_chunks,
            filename=file.filename,
            target_language=target_language,
            language_code=language_code,
            layout_data=layout_data
        )
        
        pdf_buffer.seek(0)
        
        # Generate filename
        filename = file.filename or "document.pdf"
        pdf_filename = filename.replace('.PDF', '.pdf')
        if not pdf_filename.endswith('.pdf'):
            pdf_filename += '.pdf'
        
        return StreamingResponse(
            iter([pdf_buffer.getvalue()]),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=translated_{pdf_filename}"}
        )
        
    except Exception as e:
        logger.error(f"Error creating PDF document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating PDF document: {str(e)}")

@router.post("/download-translated-word")
async def download_translated_word(
    file: UploadFile = File(...),
    target_language: str = Form(...),
    original_text_chunks_json: str = Form(...),
    translated_text_chunks_json: str = Form(...),
    layout_data_json: str = Form(...),
    language_code: str = Form(...)
):
    """Download translated document as Word format from live preview"""
    try:
        # Parse the JSON data
        original_text_chunks = json.loads(original_text_chunks_json)
        translated_text_chunks = json.loads(translated_text_chunks_json)
        layout_data = json.loads(layout_data_json)
        
        # Create Word document
        doc = Document()
        
        # Set document margins
        sections = doc.sections
        for section in sections:
            section.top_margin = Pt(72)  # 1 inch
            section.bottom_margin = Pt(72)
            section.left_margin = Pt(72)
            section.right_margin = Pt(72)
        
        # Add title
        title = doc.add_heading('Translated Document', 0)
        title.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        
        # Add language info
        lang_info = doc.add_paragraph(f'Translated to: {target_language}')
        lang_info.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        
        # Add separator
        doc.add_paragraph('─' * 50)
        
        # Group text by paragraphs if layout data is available
        if layout_data and len(layout_data) > 0:
            # Group elements by paragraph
            paragraph_groups = {}
            for element in layout_data:
                para_idx = element.get('paragraph_index', 0)
                if para_idx not in paragraph_groups:
                    paragraph_groups[para_idx] = []
                paragraph_groups[para_idx].append(element)
            
            # Add content by paragraphs
            for para_idx in sorted(paragraph_groups.keys()):
                elements = paragraph_groups[para_idx]
                if not elements:
                    continue
                
                first_element = elements[0]
                is_title = first_element.get('is_title', False)
                is_heading = first_element.get('is_heading', False)
                
                # Create paragraph
                if is_title:
                    para = doc.add_heading('', level=1)
                    para.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
                elif is_heading:
                    para = doc.add_heading('', level=2)
                else:
                    para = doc.add_paragraph()
                
                # Add translated text
                translated_text = ' '.join([translated_text_chunks[i] for i in range(len(elements)) if i < len(translated_text_chunks)])
                if translated_text.strip():
                    run = para.add_run(translated_text)
                    if is_title or is_heading:
                        run.bold = True
                        run.font.size = Pt(16 if is_heading else 20)
                    else:
                        run.font.size = Pt(12)
                
                # Add some spacing
                if not is_title:
                    doc.add_paragraph()
        else:
            # Fallback: add all translated text as one paragraph
            all_translated = ' '.join(translated_text_chunks)
            if all_translated.strip():
                para = doc.add_paragraph(all_translated)
                para.alignment = WD_PARAGRAPH_ALIGNMENT.JUSTIFY
        
        # Save to BytesIO
        doc_buffer = io.BytesIO()
        doc.save(doc_buffer)
        doc_buffer.seek(0)
        
        # Generate filename
        filename = file.filename or "document.pdf"
        word_filename = filename.replace('.pdf', '.docx').replace('.PDF', '.docx')
        if not word_filename.endswith('.docx'):
            word_filename += '.docx'
        
        return StreamingResponse(
            io.BytesIO(doc_buffer.getvalue()),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f"attachment; filename=translated_{word_filename}"}
        )
        
    except Exception as e:
        logger.error(f"Error creating Word document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating Word document: {str(e)}")
