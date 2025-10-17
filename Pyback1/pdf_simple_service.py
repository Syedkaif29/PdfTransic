"""
Simple PDF generation service that mirrors the Word document approach
with proper font support for Indian languages
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import json
import io
import logging
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from font_management import get_language_mapper, get_font_registry

router = APIRouter()
logger = logging.getLogger(__name__)

def get_font_system_status():
    """Get font system status from main app"""
    try:
        # Import the global font system status from main
        import main
        return getattr(main, 'font_system_initialized', False), getattr(main, 'font_initialization_error', None)
    except:
        return False, "Could not access main app font system"

@router.post("/download-pdf-simple")
async def download_pdf_simple(
    file: UploadFile = File(...),
    filename: str = Form(...),
    target_language: str = Form(...),
    original_text_chunks_json: str = Form(...),
    translated_text_chunks_json: str = Form(...),
    layout_data_json: str = Form(...),
    language_code: str = Form(...)
):
    """
    Simple PDF download endpoint that works exactly like the Word download
    but generates PDF instead of DOCX
    """
    logger.info(f"Simple PDF download request for: {filename}")
    
    try:
        # Parse JSON data (identical to Word endpoint)
        original_text_chunks = json.loads(original_text_chunks_json)
        translated_text_chunks = json.loads(translated_text_chunks_json)
        layout_data = json.loads(layout_data_json)
        
        # Validate data
        if not translated_text_chunks:
            raise HTTPException(status_code=422, detail="No translated text provided")
        
        # Get appropriate font for the target language
        selected_font_family = 'Helvetica'  # Default fallback
        selected_font_regular = 'Helvetica'
        selected_font_bold = 'Helvetica-Bold'
        
        try:
            # Check if font system is initialized (from main app)
            font_system_initialized, font_initialization_error = get_font_system_status()
            
            if font_system_initialized:
                language_mapper = get_language_mapper()
                font_info = language_mapper.get_font_info_for_language(language_code or target_language)
                selected_font_family = font_info['selected_font']
                
                logger.info(f"Font selection for {target_language}: {selected_font_family}")
                
                # Get specific font variants using FontRegistry
                font_registry = get_font_registry()
                selected_font_regular = font_registry.get_font_by_family_and_weight_with_fallback(selected_font_family, 'Regular') or 'Helvetica'
                selected_font_bold = font_registry.get_font_by_family_and_weight_with_fallback(selected_font_family, 'Bold') or selected_font_regular
                
                # Ensure fonts are registered with ReportLab
                for font_name_to_check in [selected_font_regular, selected_font_bold]:
                    if font_name_to_check and font_name_to_check not in ['Helvetica', 'Helvetica-Bold']:
                        try:
                            # Check if font is already registered
                            pdfmetrics.getFont(font_name_to_check)
                        except:
                            # Font not registered, try to register it
                            try:
                                font_info_to_register = font_registry.get_font_info(font_name_to_check)
                                if font_info_to_register and os.path.exists(font_info_to_register.file_path):
                                    pdfmetrics.registerFont(TTFont(font_name_to_check, font_info_to_register.file_path))
                                    logger.info(f"Registered font for PDF: {font_name_to_check}")
                                else:
                                    logger.warning(f"Could not find font file for: {font_name_to_check}")
                            except Exception as e:
                                logger.error(f"Failed to register font {font_name_to_check}: {e}")
                
                logger.info(f"Using fonts - Regular: {selected_font_regular}, Bold: {selected_font_bold}")
            else:
                logger.warning("Font system not initialized, using default fonts")
                if font_initialization_error:
                    logger.warning(f"Font initialization error: {font_initialization_error}")
                    
        except Exception as e:
            logger.error(f"Error in font selection: {e}")
            # Continue with default fonts
        
        # Create PDF buffer
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4
        margin = 1 * inch
        y_position = height - margin
        line_height = 14
        
        # Document header (same as Word doc)
        c.setFont(selected_font_bold, 20)
        title = "Translated Document"
        title_width = c.stringWidth(title, selected_font_bold, 20)
        c.drawString((width - title_width) / 2, y_position, title)
        y_position -= 2 * line_height
        
        # Language info
        c.setFont(selected_font_regular, 12)
        lang_info = f"Translated to: {target_language}"
        lang_width = c.stringWidth(lang_info, selected_font_regular, 12)
        c.drawString((width - lang_width) / 2, y_position, lang_info)
        y_position -= line_height
        
        # Separator
        separator = "─" * 50
        sep_width = c.stringWidth(separator, selected_font_regular, 12)
        c.drawString((width - sep_width) / 2, y_position, separator)
        y_position -= 2 * line_height
        
        # Content (same logic as Word doc)
        c.setFont(selected_font_regular, 12)
        
        if layout_data and len(layout_data) > 0:
            # Group by paragraphs (same as Word)
            paragraph_groups = {}
            for element in layout_data:
                para_idx = element.get('paragraph_index', 0)
                if para_idx not in paragraph_groups:
                    paragraph_groups[para_idx] = []
                paragraph_groups[para_idx].append(element)
            
            # Process paragraphs
            for para_idx in sorted(paragraph_groups.keys()):
                elements = paragraph_groups[para_idx]
                if not elements:
                    continue
                
                first_element = elements[0]
                is_title = first_element.get('is_title', False)
                is_heading = first_element.get('is_heading', False)
                
                # Set font based on element type
                if is_title:
                    current_font = selected_font_bold
                    font_size = 20
                elif is_heading:
                    current_font = selected_font_bold
                    font_size = 16
                else:
                    current_font = selected_font_regular
                    font_size = 12
                
                c.setFont(current_font, font_size)
                
                # Get translated text for this paragraph
                translated_text = ' '.join([
                    translated_text_chunks[i] 
                    for i in range(len(elements)) 
                    if i < len(translated_text_chunks)
                ])
                
                if translated_text.strip():
                    # Simple text wrapping
                    words = translated_text.split()
                    current_line = ""
                    
                    for word in words:
                        test_line = current_line + " " + word if current_line else word
                        
                        if c.stringWidth(test_line, current_font, font_size) < (width - 2 * margin):
                            current_line = test_line
                        else:
                            if current_line:
                                if y_position < margin:
                                    c.showPage()
                                    y_position = height - margin
                                
                                # Center titles and headings
                                if is_title or is_heading:
                                    line_width = c.stringWidth(current_line, current_font, font_size)
                                    x_pos = (width - line_width) / 2
                                else:
                                    x_pos = margin
                                
                                c.drawString(x_pos, y_position, current_line)
                                y_position -= line_height * (1.5 if is_title else 1.2 if is_heading else 1)
                            current_line = word
                    
                    # Draw remaining text
                    if current_line:
                        if y_position < margin:
                            c.showPage()
                            y_position = height - margin
                        
                        if is_title or is_heading:
                            line_width = c.stringWidth(current_line, current_font, font_size)
                            x_pos = (width - line_width) / 2
                        else:
                            x_pos = margin
                        
                        c.drawString(x_pos, y_position, current_line)
                        y_position -= line_height * (1.5 if is_title else 1.2 if is_heading else 1)
                
                # Add spacing between paragraphs
                if not is_title:
                    y_position -= line_height / 2
        else:
            # Fallback: all text as one block (same as Word)
            all_translated = ' '.join(translated_text_chunks)
            if all_translated.strip():
                c.setFont(selected_font_regular, 12)
                words = all_translated.split()
                current_line = ""
                
                for word in words:
                    test_line = current_line + " " + word if current_line else word
                    if c.stringWidth(test_line, selected_font_regular, 12) < (width - 2 * margin):
                        current_line = test_line
                    else:
                        if current_line:
                            if y_position < margin:
                                c.showPage()
                                y_position = height - margin
                            c.drawString(margin, y_position, current_line)
                            y_position -= line_height
                        current_line = word
                
                if current_line:
                    if y_position < margin:
                        c.showPage()
                        y_position = height - margin
                    c.drawString(margin, y_position, current_line)
        
        # Save PDF
        c.save()
        buffer.seek(0)
        
        # Generate filename
        pdf_filename = filename.replace('.PDF', '.pdf')
        if not pdf_filename.endswith('.pdf'):
            pdf_filename += '.pdf'
        
        logger.info(f"Generated PDF: {len(buffer.getvalue())} bytes")
        
        return StreamingResponse(
            iter([buffer.getvalue()]),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=translated_{pdf_filename}"}
        )
        
    except Exception as e:
        logger.error(f"PDF generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")