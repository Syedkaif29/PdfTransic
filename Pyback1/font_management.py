"""
Font Management Infrastructure for Indian Language Support

This module provides comprehensive font management capabilities for the PDF translation API,
including font discovery, registration, validation, and error handling.
"""

import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any
from pathlib import Path
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib import colors
from font_config import get_font_config, FontSystemConfig
import re

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class FontInfo:
    """Information about a registered font"""
    name: str
    file_path: str
    family: str
    weight: str
    style: str = "normal"
    is_registered: bool = False
    file_size: Optional[int] = None
    
    def __post_init__(self):
        """Calculate file size if not provided"""
        if self.file_size is None and os.path.exists(self.file_path):
            try:
                self.file_size = os.path.getsize(self.file_path)
            except OSError as e:
                logger.warning(f"Could not get file size for {self.file_path}: {e}")
                self.file_size = 0


class FontRegistry:
    """
    Centralized font registry for managing font discovery, registration, and lookup.
    
    This class handles:
    - Automatic font discovery from directory structures
    - Font file validation and registration with ReportLab
    - Error handling and logging for font operations
    - Font family and weight management
    - Comprehensive fallback mechanisms for missing or corrupted fonts
    """
    
    def __init__(self, config: Optional[FontSystemConfig] = None):
        """Initialize the font registry with configuration support"""
        self.config = config or get_font_config()
        self.registered_fonts: Dict[str, FontInfo] = {}
        self.font_families: Dict[str, List[str]] = {}
        self._supported_extensions: Set[str] = self.config.supported_extensions
        self._registration_errors: List[str] = []
        self._corrupted_fonts: Set[str] = set()
        self._missing_fonts: Set[str] = set()
        self._fallback_enabled: bool = self.config.preferences.enable_fallback
        self._max_retries: int = self.config.max_registration_retries
        self._max_file_size: int = self.config.max_file_size_mb * 1024 * 1024  # Convert to bytes
        self._min_file_size: int = self.config.min_file_size_bytes
        self._max_errors: int = self.config.max_errors_before_abort
        self._error_count: int = 0
        
        # Set up logging level if specified
        if self.config.enable_font_logging:
            font_logger = logging.getLogger(__name__)
            # font_logger.setLevel(getattr(logging, self.config.log_level, logging.INFO)) # Removed to inherit global config
        
        logger.info("FontRegistry initialized with configuration support")
    
    def register_fonts_from_directory(self, base_path: str) -> Tuple[int, int]:
        """
        Scan and register all fonts from the specified directory structure with comprehensive error handling.
        
        Args:
            base_path: Base directory path containing font files
            
        Returns:
            Tuple of (successful_registrations, failed_registrations)
        """
        logger.info(f"🔤 Starting font registration from directory: {base_path}")
        logger.debug(f"Current working directory: {os.getcwd()}")
        logger.debug(f"Absolute path being checked: {os.path.abspath(base_path)}")
        
        # Enhanced directory validation with graceful degradation
        if not base_path:
            error_msg = "Font directory path is empty or None"
            logger.warning(f"⚠️ {error_msg} - continuing with default fonts")
            self._registration_errors.append(error_msg)
            return 0, 0
            
        if not os.path.exists(base_path):
            error_msg = f"Font directory does not exist: {base_path}"
            logger.warning(f"⚠️ {error_msg} - PDF generation will use system default fonts")
            self._registration_errors.append(error_msg)
            return 0, 0
        
        if not os.path.isdir(base_path):
            error_msg = f"Font path is not a directory: {base_path}"
            logger.warning(f"⚠️ {error_msg} - continuing with default fonts")
            self._registration_errors.append(error_msg)
            return 0, 0
            
        # Check directory permissions
        if not os.access(base_path, os.R_OK):
            error_msg = f"Font directory is not readable: {base_path}"
            logger.warning(f"⚠️ {error_msg} - check directory permissions")
            self._registration_errors.append(error_msg)
            return 0, 0
        
        successful = 0
        failed = 0
        
        try:
            # Walk through all subdirectories to find font files with error recovery
            for root, dirs, files in os.walk(base_path):
                # Skip directories that can't be accessed
                try:
                    if not os.access(root, os.R_OK):
                        logger.warning(f"⚠️ Skipping unreadable directory: {root}")
                        continue
                        
                    for file in files:
                        try:
                            file_path = os.path.join(root, file)
                            file_ext = os.path.splitext(file)[1].lower()
                            
                            if file_ext in self._supported_extensions:
                                if self._register_single_font_with_recovery(file_path):
                                    successful += 1
                                else:
                                    failed += 1
                                    
                                # Check if we should abort due to too many errors
                                if not self.config.continue_on_errors and self._error_count >= self._max_errors:
                                    logger.error(f"❌ Aborting font registration: too many errors ({self._error_count})")
                                    return successful, failed
                                    
                        except Exception as file_error:
                            error_msg = f"Error processing font file {file}: {file_error}"
                            logger.warning(f"⚠️ {error_msg}")
                            self._registration_errors.append(error_msg)
                            self._error_count += 1
                            failed += 1
                            
                            # Check if we should abort due to too many errors
                            if not self.config.continue_on_errors and self._error_count >= self._max_errors:
                                logger.error(f"❌ Aborting font registration: too many errors ({self._error_count})")
                                return successful, failed
                            
                except Exception as dir_error:
                    error_msg = f"Error accessing directory {root}: {dir_error}"
                    logger.warning(f"⚠️ {error_msg}")
                    self._registration_errors.append(error_msg)
                    continue
            
            # Log comprehensive results
            if successful > 0:
                logger.info(f"✅ Font registration completed: {successful} successful, {failed} failed")
            elif failed > 0:
                logger.warning(f"⚠️ Font registration completed with issues: 0 successful, {failed} failed - PDF generation will use system fonts")
            else:
                logger.warning(f"⚠️ No font files found in directory: {base_path}")
            
        except Exception as e:
            error_msg = f"Critical error during font directory scan: {e}"
            logger.error(f"❌ {error_msg}")
            self._registration_errors.append(error_msg)
            failed += 1
        
        return successful, failed
    
    def _register_single_font(self, font_path: str) -> bool:
        """
        Register a single font file with ReportLab.
        
        Args:
            font_path: Path to the font file
            
        Returns:
            True if registration successful, False otherwise
        """
        return self._register_single_font_with_recovery(font_path)
    
    def _register_single_font_with_recovery(self, font_path: str) -> bool:
        """
        Register a single font file with ReportLab with comprehensive error recovery.
        
        Args:
            font_path: Path to the font file
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            # Enhanced font file validation with detailed error reporting
            validation_result = self._validate_font_file_comprehensive(font_path)
            if not validation_result['valid']:
                # Track specific failure reasons
                if validation_result['reason'] == 'missing':
                    self._missing_fonts.add(font_path)
                elif validation_result['reason'] == 'corrupted':
                    self._corrupted_fonts.add(font_path)
                return False
            
            # Extract font information from filename with error handling
            try:
                font_info = self._extract_font_info(font_path)
            except Exception as e:
                error_msg = f"Failed to extract font info from {font_path}: {e}"
                logger.warning(f"⚠️ {error_msg}")
                self._registration_errors.append(error_msg)
                return False
            
            # Register with ReportLab with multiple retry attempts (from configuration)
            registration_success = False
            retry_count = 0
            max_retries = self._max_retries
            
            while not registration_success and retry_count < max_retries:
                try:
                    # Attempt ReportLab registration
                    pdfmetrics.registerFont(TTFont(font_info.name, font_path))
                    font_info.is_registered = True
                    registration_success = True
                    
                    # Store in registry
                    self.registered_fonts[font_info.name] = font_info
                    
                    # Update font families with duplicate checking
                    if font_info.family not in self.font_families:
                        self.font_families[font_info.family] = []
                    
                    # Avoid duplicate entries
                    if font_info.name not in self.font_families[font_info.family]:
                        self.font_families[font_info.family].append(font_info.name)
                    
                    logger.info(f"✅ Successfully registered font: {font_info.name} from {font_path}")
                    
                    # After registering the font, update the font family mapping
                    # This is crucial for ReportLab to correctly resolve font variations (bold, italic)
                    self._update_font_family_registration(font_info)
                    
                    return True
                    
                except Exception as e:
                    retry_count += 1
                    error_msg = f"ReportLab registration attempt {retry_count} failed for {font_path}: {e}"
                    
                    if retry_count < max_retries:
                        logger.warning(f"⚠️ {error_msg} - retrying...")
                        # Brief pause before retry
                        import time
                        time.sleep(0.1)
                    else:
                        logger.error(f"❌ {error_msg} - all retry attempts exhausted")
                        self._registration_errors.append(error_msg)
                        self._corrupted_fonts.add(font_path)
                        return False
                
        except Exception as e:
            error_msg = f"Critical font registration error for {font_path}: {e}"
            logger.error(f"❌ {error_msg}")
            self._registration_errors.append(error_msg)
            self._corrupted_fonts.add(font_path)
            return False
        
        return False
    
    def _update_font_family_registration(self, font_info: FontInfo):
        """
        Updates ReportLab's font family registration with the given font_info.
        This ensures ReportLab can correctly find bold/italic versions of a font family.
        """
        family_name = font_info.family
        
        # Prepare arguments for registerFontFamily by checking all currently registered fonts
        regular = None
        bold = None
        italic = None
        bold_italic = None

        # Iterate through all already registered fonts to build the family definition
        # This ensures we are always providing the most complete family definition to ReportLab
        for f_name, f_info in self.registered_fonts.items():
            if f_info.family == family_name:
                lower_weight = f_info.weight.lower()
                lower_style = f_info.style.lower()
                
                if lower_weight == 'regular' and lower_style == 'normal':
                    regular = f_info.name
                elif lower_weight == 'bold' and lower_style == 'normal':
                    bold = f_info.name
                elif lower_weight == 'regular' and lower_style == 'italic':
                    italic = f_info.name
                elif lower_weight == 'bold' and lower_style == 'italic':
                    bold_italic = f_info.name
        
        # Register the font family if at least one variant is found
        # ReportLab requires at least 'normal' for registration, but we will register if any variant is present
        if regular or bold or italic or bold_italic:
            try:
                # Only call registerFontFamily if there are new or updated variants
                # ReportLab will update existing family definitions if called again with the same family name
                pdfmetrics.registerFontFamily(family_name, normal=regular, bold=bold, italic=italic, boldItalic=bold_italic)
                logger.info(f"✅ Successfully updated/registered font family: '{family_name}' with variants (R:{regular}, B:{bold}, I:{italic}, BI:{bold_italic})")
            except Exception as e:
                logger.error(f"❌ Failed to register font family '{family_name}': {e}")
    
    def _validate_font_file(self, font_path: str) -> bool:
        """
        Validate that a font file exists and is readable.
        
        Args:
            font_path: Path to the font file
            
        Returns:
            True if valid, False otherwise
        """
        result = self._validate_font_file_comprehensive(font_path)
        return result['valid']
    
    def _validate_font_file_comprehensive(self, font_path: str) -> Dict:
        """
        Comprehensive font file validation with detailed error reporting.
        
        Args:
            font_path: Path to the font file
            
        Returns:
            Dictionary with validation result and detailed information
        """
        validation_result = {
            'valid': False,
            'reason': 'unknown',
            'details': '',
            'file_size': 0,
            'readable': False,
            'format_supported': False
        }
        
        try:
            # Check if path is provided
            if not font_path:
                validation_result.update({
                    'reason': 'invalid_path',
                    'details': 'Font path is empty or None'
                })
                logger.warning(f"⚠️ {validation_result['details']}")
                return validation_result
            
            # Check if file exists
            if not os.path.exists(font_path):
                validation_result.update({
                    'reason': 'missing',
                    'details': f'Font file does not exist: {font_path}'
                })
                logger.warning(f"⚠️ {validation_result['details']}")
                return validation_result
            
            # Check if it's actually a file
            if not os.path.isfile(font_path):
                validation_result.update({
                    'reason': 'not_file',
                    'details': f'Path is not a file: {font_path}'
                })
                logger.warning(f"⚠️ {validation_result['details']}")
                return validation_result
            
            # Check file permissions
            if not os.access(font_path, os.R_OK):
                validation_result.update({
                    'reason': 'permission_denied',
                    'details': f'Font file is not readable (permission denied): {font_path}'
                })
                logger.warning(f"⚠️ {validation_result['details']}")
                return validation_result
            
            # Check file size
            try:
                file_size = os.path.getsize(font_path)
                validation_result['file_size'] = file_size
                
                if file_size == 0:
                    validation_result.update({
                        'reason': 'empty',
                        'details': f'Font file is empty: {font_path}'
                    })
                    logger.warning(f"⚠️ {validation_result['details']}")
                    return validation_result
                
                # Check reasonable file size limits (from configuration)
                min_size = self._min_file_size
                max_size = self._max_file_size
                
                if file_size < min_size:
                    validation_result.update({
                        'reason': 'too_small',
                        'details': f'Font file too small ({file_size} bytes, minimum {min_size}): {font_path}'
                    })
                    logger.warning(f"⚠️ {validation_result['details']}")
                    return validation_result
                
                if file_size > max_size:
                    validation_result.update({
                        'reason': 'too_large',
                        'details': f'Font file too large ({file_size} bytes, maximum {max_size}): {font_path}'
                    })
                    logger.warning(f"⚠️ {validation_result['details']}")
                    return validation_result
                    
            except OSError as e:
                validation_result.update({
                    'reason': 'size_error',
                    'details': f'Cannot get file size for {font_path}: {e}'
                })
                logger.warning(f"⚠️ {validation_result['details']}")
                return validation_result
            
            # Check file extension
            file_ext = os.path.splitext(font_path)[1].lower()
            validation_result['format_supported'] = file_ext in self._supported_extensions
            
            if not validation_result['format_supported']:
                validation_result.update({
                    'reason': 'unsupported_format',
                    'details': f'Unsupported font format {file_ext}: {font_path}'
                })
                logger.warning(f"⚠️ {validation_result['details']}")
                return validation_result
            
            # Try to read the file to ensure it's accessible and not corrupted
            try:
                with open(font_path, 'rb') as f:
                    # Read first chunk to verify file integrity
                    header = f.read(1024)
                    if len(header) < 4:
                        validation_result.update({
                            'reason': 'corrupted',
                            'details': f'Font file appears corrupted (insufficient header): {font_path}'
                        })
                        logger.warning(f"⚠️ {validation_result['details']}")
                        return validation_result
                    
                    # Basic TTF/OTF header validation
                    if file_ext == '.ttf':
                        # TTF files should start with specific signatures
                        ttf_signatures = [b'\x00\x01\x00\x00', b'OTTO', b'true', b'typ1']
                        if not any(header.startswith(sig) for sig in ttf_signatures):
                            validation_result.update({
                                'reason': 'corrupted',
                                'details': f'TTF file has invalid header signature: {font_path}'
                            })
                            logger.warning(f"⚠️ {validation_result['details']}")
                            return validation_result
                    
                    validation_result['readable'] = True
                    
            except IOError as e:
                validation_result.update({
                    'reason': 'read_error',
                    'details': f'Cannot read font file {font_path}: {e}'
                })
                logger.warning(f"⚠️ {validation_result['details']}")
                return validation_result
            except Exception as e:
                validation_result.update({
                    'reason': 'corrupted',
                    'details': f'Font file appears corrupted {font_path}: {e}'
                })
                logger.warning(f"⚠️ {validation_result['details']}")
                return validation_result
            
            # If we reach here, the file passed all validations
            validation_result['valid'] = True
            validation_result['reason'] = 'valid'
            validation_result['details'] = f'Font file validation successful: {font_path}'
            
            return validation_result
                
        except Exception as e:
            validation_result.update({
                'reason': 'validation_error',
                'details': f'Font validation error for {font_path}: {e}'
            })
            logger.error(f"❌ {validation_result['details']}")
            return validation_result
    
    def _extract_font_info(self, font_path: str) -> FontInfo:
        """
        Extracts font information from the font file name and metadata.
        This enhanced version attempts to be smarter about naming.
        """
        logger.debug(f"⚙️ Extracting font info for: {font_path}")
        
        # Use pathlib for robust path parsing
        p = Path(font_path)
        file_name_without_ext = p.stem
        
        # Attempt to extract family, weight, and style more robustly
        family = "Unknown"
        weight = "Regular"
        style = "Normal"
        
        # Heuristic for Noto fonts
        if "Noto" in file_name_without_ext:
            if "Devanagari" in file_name_without_ext:
                family = "NotoSansDevanagari"
            elif "NastaliqUrdu" in file_name_without_ext:
                family = "NotoNastaliqUrdu"
            elif "Sinhala" in file_name_without_ext:
                family = "NotoSansSinhala"
            else:
                # General Noto, try to find a base name like NotoSans or NotoSerif
                match = re.search(r"Noto(Sans|Serif)?([A-Za-z]+)?", file_name_without_ext)
                if match:
                    family_prefix = "Noto" + (match.group(1) or "") # Noto, NotoSans, NotoSerif
                    # Look for script specific part after the main family prefix
                    script_match = re.search(r"""(?:Devanagari|NastaliqUrdu|Sinhala)""", file_name_without_ext)
                    if script_match:
                        family = family_prefix + script_match.group(0)
                    elif match.group(2): # If no specific script, but there's a suffix
                         family = family_prefix + match.group(2)
                    else:
                         family = family_prefix

            # Extract weight and style for Noto fonts from common patterns
            lower_name = file_name_without_ext.lower()
            if "extrabold" in lower_name:
                weight = "ExtraBold"
            elif "bold" in lower_name:
                weight = "Bold"
            elif "medium" in lower_name:
                weight = "Medium"
            elif "light" in lower_name:
                weight = "Light"
            elif "thin" in lower_name:
                weight = "Thin"
            elif "semibold" in lower_name:
                weight = "SemiBold"
            elif "extralight" in lower_name:
                weight = "ExtraLight"
            elif "regular" in lower_name:
                weight = "Regular"
            
            if "italic" in lower_name:
                style = "Italic"

        else:
            # Fallback for non-Noto fonts or if Noto heuristics fail
            # Attempt to get font family from the file itself using fontTools.ttLib if possible
            # For now, rely on simpler name parsing
            if '-' in file_name_without_ext:
                parts = file_name_without_ext.split('-')
                family = parts[0]
                style_weight_part = parts[-1].lower()
                
                if "bolditalic" in style_weight_part:
                    weight = "Bold"
                    style = "Italic"
                elif "bold" in style_weight_part:
                    weight = "Bold"
                elif "italic" in style_weight_part:
                    style = "Italic"
                elif "regular" in style_weight_part:
                    weight = "Regular"
                elif "medium" in style_weight_part:
                    weight = "Medium"
                elif "light" in style_weight_part:
                    weight = "Light"
            else:
                family = file_name_without_ext # Assume the whole name is the family if no hyphen
        
        # Construct a ReportLab-compatible font name
        # ReportLab expects names like 'FamilyName-WeightStyle' (e.g., 'Helvetica-Bold', 'NotoSansDevanagari-Regular')
        reportlab_font_name = family
        # Special handling for variable fonts to prefer static variants
        if "VariableFont" in file_name_without_ext and family in ["NotoSansDevanagari", "NotoNastaliqUrdu", "NotoSansSinhala"]:
            logger.debug(f"Detected variable font for {family}. Attempting to use static variants if available.")
            # For variable fonts, we typically want to map them to a 'Regular' static name
            weight = "Regular" # Default to Regular for variable font if no specific static weight is extracted
            style = "Normal"
            # The actual file path will still be the variable font, but the registered name will be static-like.
            # This relies on _update_font_family_registration to handle the full family registration.

        if weight != "Regular" or style != "Normal": # Only append if it's not the default regular normal
            name_parts = []
            if weight != "Regular":
                name_parts.append(weight)
            if style != "Normal":
                name_parts.append(style)
            if name_parts:
                reportlab_font_name = f"{family}-{''.join(name_parts)}"
        
        logger.debug(f"  - Extracted: Family='{family}', Weight='{weight}', Style='{style}', ReportLab Name='{reportlab_font_name}'")
        
        return FontInfo(
            name=reportlab_font_name, # Use the ReportLab-compatible name here
            file_path=font_path,
            family=family,
            weight=weight,
            style=style
        )
    
    def get_font_family(self, family_name: str) -> Optional[List[str]]:
        """
        Get all registered fonts for a specific family.
        
        Args:
            family_name: Name of the font family
            
        Returns:
            List of font names in the family, or None if family not found
        """
        return self.font_families.get(family_name)
    
    def get_font_by_family_and_weight(self, family_name: str, weight: str = 'Regular') -> Optional[str]:
        """
        Get a specific font by family and weight with comprehensive fallback.
        
        Args:
            family_name: Name of the font family
            weight: Font weight (default: 'Regular')
            
        Returns:
            Font name if found, None otherwise
        """
        return self.get_font_by_family_and_weight_with_fallback(family_name, weight)
    
    def get_font_by_family_and_weight_with_fallback(self, family_name: str, weight: str = 'Regular') -> Optional[str]:
        """
        Get a specific font by family and weight with comprehensive fallback mechanisms.
        
        Args:
            family_name: Name of the font family
            weight: Font weight (default: 'Regular')
            
        Returns:
            Font name if found, fallback font name, or None if no fonts available
        """
        logger.debug(f"🔍 Looking for font: {family_name}-{weight}")
        
        # Check if family exists
        family_fonts = self.get_font_family(family_name)
        if not family_fonts:
            logger.warning(f"⚠️ Font family '{family_name}' not available")
            return None
        
        # Look for exact weight match
        for font_name in family_fonts:
            font_info = self.registered_fonts.get(font_name)
            if font_info and font_info.weight == weight and font_info.is_registered:
                logger.debug(f"✅ Found exact match: {font_name}")
                return font_name
        
        # Use configuration preferences for weight fallbacks
        if self.config.preferences.strict_weight_matching:
            # In strict mode, only try the exact weight
            fallback_weights = []
        else:
            # Weight fallback hierarchy with preference for configured preferred weights
            preferred_weights = self.config.preferences.preferred_weights
            weight_fallbacks = {
                'Thin': ['ExtraLight', 'Light'] + preferred_weights,
                'ExtraLight': ['Light'] + preferred_weights + ['Thin'],
                'Light': preferred_weights + ['ExtraLight', 'Medium'],
                'Regular': preferred_weights + ['Light', 'SemiBold'],
                'Medium': preferred_weights + ['SemiBold', 'Light'],
                'SemiBold': ['Medium', 'Bold'] + preferred_weights,
                'Bold': ['SemiBold', 'ExtraBold', 'Medium'],
                'ExtraBold': ['Bold', 'Black', 'SemiBold'],
                'Black': ['ExtraBold', 'Bold', 'SemiBold']
            }
            
            # Get fallback weights, defaulting to preferred weights
            fallback_weights = weight_fallbacks.get(weight, preferred_weights + ['Regular', 'Medium', 'Light'])
            
            # Remove duplicates while preserving order
            seen = set()
            fallback_weights = [w for w in fallback_weights if not (w in seen or seen.add(w))]
        
        # Try fallback weights
        fallback_weights = weight_fallbacks.get(weight, ['Regular', 'Medium', 'Light'])
        for fallback_weight in fallback_weights:
            for font_name in family_fonts:
                font_info = self.registered_fonts.get(font_name)
                if font_info and font_info.weight == fallback_weight and font_info.is_registered:
                    logger.info(f"📝 Using fallback weight '{fallback_weight}' instead of '{weight}' for family '{family_name}'")
                    return font_name
        
        # If no specific weight found, return first available registered font in family
        for font_name in family_fonts:
            font_info = self.registered_fonts.get(font_name)
            if font_info and font_info.is_registered:
                logger.info(f"📝 Using first available font '{font_name}' from family '{family_name}'")
                return font_name
        
        logger.warning(f"⚠️ No registered fonts found in family '{family_name}'")
        return None
    
    def is_font_available(self, font_name: str) -> bool:
        """
        Check if a font is registered and available.
        
        Args:
            font_name: Name of the font
            
        Returns:
            True if font is available, False otherwise
        """
        font_info = self.registered_fonts.get(font_name)
        return font_info is not None and font_info.is_registered
    
    def is_family_available(self, family_name: str) -> bool:
        """
        Check if a font family has any registered fonts.
        
        Args:
            family_name: Name of the font family
            
        Returns:
            True if family has registered fonts, False otherwise
        """
        return family_name in self.font_families and len(self.font_families[family_name]) > 0
    
    def get_registration_summary(self) -> Dict[str, Any]:
        """
        Provides a summary of registered fonts and families.
        """
        total_fonts = len(self.registered_fonts)
        registered_fonts = len([f for f in self.registered_fonts.values() if f.is_registered])
        failed_fonts = total_fonts - registered_fonts
        
        fonts_by_family = {family: len(names) for family, names in self.font_families.items()}
        
        return {
            'total_fonts': total_fonts,
            'registered_fonts': registered_fonts,
            'failed_fonts': failed_fonts,
            'total_families': len(self.font_families),
            'available_families': list(self.font_families.keys()),
            'registration_errors': self._registration_errors.copy(),
            'fonts_by_family': fonts_by_family
        }
    
    def get_font_info(self, font_name: str) -> Optional[FontInfo]:
        """
        Get detailed information about a specific font.
        
        Args:
            font_name: Name of the font
            
        Returns:
            FontInfo object if found, None otherwise
        """
        return self.registered_fonts.get(font_name)
    
    def perform_health_check(self) -> Dict:
        """
        Perform a comprehensive health check of the font system.
        
        Returns:
            Dictionary with health check results and recommendations
        """
        logger.info("🏥 Performing font system health check...")
        
        health_status = {
            'overall_status': 'healthy',
            'timestamp': str(os.times()),
            'checks': {},
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        # Check 1: Font registration status
        total_fonts = len(self.registered_fonts)
        registered_fonts = sum(1 for font in self.registered_fonts.values() if font.is_registered)
        
        health_status['checks']['font_registration'] = {
            'status': 'pass' if registered_fonts > 0 else 'fail',
            'total_fonts': total_fonts,
            'registered_fonts': registered_fonts,
            'registration_rate': (registered_fonts / total_fonts * 100) if total_fonts > 0 else 0
        }
        
        if registered_fonts == 0:
            health_status['errors'].append("No fonts are registered - PDF generation will use system defaults")
            health_status['overall_status'] = 'critical'
        elif registered_fonts < total_fonts:
            health_status['warnings'].append(f"{total_fonts - registered_fonts} fonts failed to register")
            if health_status['overall_status'] == 'healthy':
                health_status['overall_status'] = 'warning'
        
        # Check 2: Font family availability
        required_families = ['NotoSansDevanagari', 'NotoNastaliqUrdu', 'NotoSansSinhala']
        available_families = []
        missing_families = []
        
        for family in required_families:
            if self.is_family_available(family):
                available_families.append(family)
            else:
                missing_families.append(family)
        
        health_status['checks']['font_families'] = {
            'status': 'pass' if len(available_families) == len(required_families) else 'warning',
            'required_families': required_families,
            'available_families': available_families,
            'missing_families': missing_families,
            'availability_rate': (len(available_families) / len(required_families) * 100)
        }
        
        if missing_families:
            health_status['warnings'].append(f"Missing font families: {', '.join(missing_families)}")
            if health_status['overall_status'] == 'healthy':
                health_status['overall_status'] = 'warning'
        
        # Check 3: Font file integrity
        corrupted_fonts = []
        inaccessible_fonts = []
        
        for font_name, font_info in self.registered_fonts.items():
            if not os.path.exists(font_info.file_path):
                inaccessible_fonts.append(font_name)
            elif not font_info.is_registered:
                corrupted_fonts.append(font_name)
        
        health_status['checks']['font_integrity'] = {
            'status': 'pass' if not corrupted_fonts and not inaccessible_fonts else 'warning',
            'corrupted_fonts': corrupted_fonts,
            'inaccessible_fonts': inaccessible_fonts,
            'integrity_rate': ((total_fonts - len(corrupted_fonts) - len(inaccessible_fonts)) / total_fonts * 100) if total_fonts > 0 else 100
        }
        
        if corrupted_fonts:
            health_status['warnings'].append(f"Corrupted fonts detected: {', '.join(corrupted_fonts)}")
        if inaccessible_fonts:
            health_status['errors'].append(f"Inaccessible font files: {', '.join(inaccessible_fonts)}")
            health_status['overall_status'] = 'critical'
        
        # Check 4: Registration errors
        if self._registration_errors:
            health_status['checks']['registration_errors'] = {
                'status': 'warning',
                'error_count': len(self._registration_errors),
                'recent_errors': self._registration_errors[-5:] if len(self._registration_errors) > 5 else self._registration_errors
            }
            health_status['warnings'].append(f"{len(self._registration_errors)} font registration errors occurred")
            if health_status['overall_status'] == 'healthy':
                health_status['overall_status'] = 'warning'
        else:
            health_status['checks']['registration_errors'] = {
                'status': 'pass',
                'error_count': 0
            }
        
        # Generate recommendations
        if missing_families:
            health_status['recommendations'].append("Install missing Noto font families for better Indian language support")
        
        if corrupted_fonts:
            health_status['recommendations'].append("Re-download or replace corrupted font files")
        
        if len(self._registration_errors) > 0:
            health_status['recommendations'].append("Check font file permissions and formats")
        
        if registered_fonts == 0:
            health_status['recommendations'].append("Verify font directory path and ensure font files are accessible")
        
        logger.info(f"🏥 Font system health check completed: {health_status['overall_status']}")
        return health_status
    
    def get_font_usage_stats(self) -> Dict:
        """
        Get statistics about font usage and availability.
        
        Returns:
            Dictionary with font usage statistics
        """
        stats = {
            'total_fonts': len(self.registered_fonts),
            'registered_fonts': sum(1 for font in self.registered_fonts.values() if font.is_registered),
            'font_families': len(self.font_families),
            'fonts_by_weight': {},
            'fonts_by_family': {},
            'largest_font': None,
            'smallest_font': None,
            'total_font_size': 0
        }
        
        # Analyze fonts by weight
        for font_info in self.registered_fonts.values():
            weight = font_info.weight
            if weight not in stats['fonts_by_weight']:
                stats['fonts_by_weight'][weight] = 0
            stats['fonts_by_weight'][weight] += 1
        
        # Analyze fonts by family
        for family, fonts in self.font_families.items():
            stats['fonts_by_family'][family] = len(fonts)
        
        # Find largest and smallest fonts
        font_sizes = [(font.name, font.file_size or 0) for font in self.registered_fonts.values() if font.file_size]
        if font_sizes:
            font_sizes.sort(key=lambda x: x[1])
            stats['smallest_font'] = {'name': font_sizes[0][0], 'size': font_sizes[0][1]}
            stats['largest_font'] = {'name': font_sizes[-1][0], 'size': font_sizes[-1][1]}
            stats['total_font_size'] = sum(size for _, size in font_sizes)
        
        return stats
    
    def get_corrupted_fonts(self) -> Set[str]:
        """Get set of corrupted font file paths."""
        return self._corrupted_fonts.copy()
    
    def get_missing_fonts(self) -> Set[str]:
        """Get set of missing font file paths."""
        return self._missing_fonts.copy()
    
    def recover_from_font_errors(self) -> Dict:
        """
        Attempt to recover from font registration errors by re-validating and cleaning up.
        
        Returns:
            Dictionary with recovery results
        """
        logger.info("🔧 Starting font error recovery process...")
        
        recovery_results = {
            'corrupted_fonts_removed': 0,
            'missing_fonts_removed': 0,
            'invalid_registrations_cleaned': 0,
            'recovery_successful': False,
            'errors': []
        }
        
        try:
            # Clean up corrupted fonts from registry
            fonts_to_remove = []
            for font_name, font_info in self.registered_fonts.items():
                if font_info.file_path in self._corrupted_fonts:
                    fonts_to_remove.append(font_name)
                elif not os.path.exists(font_info.file_path):
                    fonts_to_remove.append(font_name)
                    self._missing_fonts.add(font_info.file_path)
            
            # Remove invalid fonts from registry
            for font_name in fonts_to_remove:
                font_info = self.registered_fonts.pop(font_name, None)
                if font_info:
                    # Remove from family listings
                    if font_info.family in self.font_families:
                        if font_name in self.font_families[font_info.family]:
                            self.font_families[font_info.family].remove(font_name)
                        # Remove empty families
                        if not self.font_families[font_info.family]:
                            del self.font_families[font_info.family]
                    
                    recovery_results['invalid_registrations_cleaned'] += 1
                    logger.info(f"🧹 Removed invalid font registration: {font_name}")
            
            recovery_results['corrupted_fonts_removed'] = len(self._corrupted_fonts)
            recovery_results['missing_fonts_removed'] = len(self._missing_fonts)
            recovery_results['recovery_successful'] = True
            
            logger.info(f"✅ Font error recovery completed: {recovery_results}")
            
        except Exception as e:
            error_msg = f"Font error recovery failed: {e}"
            logger.error(f"❌ {error_msg}")
            recovery_results['errors'].append(error_msg)
            recovery_results['recovery_successful'] = False
        
        return recovery_results
    
    def enable_fallback_mode(self, enabled: bool = True):
        """
        Enable or disable fallback mode for font operations.
        
        Args:
            enabled: Whether to enable fallback mode
        """
        self._fallback_enabled = enabled
        logger.info(f"🔄 Font fallback mode {'enabled' if enabled else 'disabled'}")
    
    def is_fallback_enabled(self) -> bool:
        """Check if fallback mode is enabled."""
        return self._fallback_enabled
    
    def clear_registry(self):
        """Clear all registered fonts and reset the registry."""
        logger.info("🧹 Clearing font registry")
        self.registered_fonts.clear()
        self.font_families.clear()
        self._registration_errors.clear()
        self._corrupted_fonts.clear()
        self._missing_fonts.clear()


class LanguageMapper:
    """
    Maps target language codes to appropriate font families for PDF generation.
    
    This class provides:
    - Predefined language-to-font mappings for Indian languages
    - Fallback font selection when preferred fonts are unavailable
    - Support for language code variations and aliases
    """
    
    # Primary language-to-font mappings based on script systems
    LANGUAGE_FONT_MAP = {
        # Devanagari script languages
        'hin_Deva': 'NotoSansDevanagari',  # Hindi
        'mar_Deva': 'NotoSansDevanagari',  # Marathi
        'nep_Deva': 'NotoSansDevanagari',  # Nepali
        'mai_Deva': 'NotoSansDevanagari',  # Maithili
        'bho_Deva': 'NotoSansDevanagari',  # Bhojpuri
        'mag_Deva': 'NotoSansDevanagari',  # Magahi
        'new_Deva': 'NotoSansDevanagari',  # Newari
        'gom_Deva': 'NotoSansDevanagari',  # Konkani (Devanagari)
        'sa_Deva': 'NotoSansDevanagari',   # Sanskrit (Devanagari)
        'snd_Deva': 'NotoSansDevanagari', # Sindhi (Devanagari script)
        
        # Arabic/Urdu script languages
        'urd_Arab': 'NotoNastaliqUrdu',    # Urdu
        'ur_Arab': 'NotoNastaliqUrdu',     # Urdu (alternative code)
        'snd_Arab': 'NotoNastaliqUrdu',     # Sindhi (Arabic script)
        'ps_Arab': 'NotoNastaliqUrdu',     # Pashto (can use Urdu font)
        'fas_Arab': 'NotoNastaliqUrdu',    # Persian (can use Urdu font)
        
        # Sinhala script languages
        'sin_Sinh': 'NotoSansSinhala',     # Sinhala
        'si_Sinh': 'NotoSansSinhala',      # Sinhala (alternative code)
        
        # Common language code variations (without script suffix)
        'hi': 'NotoSansDevanagari',        # Hindi
        'mr': 'NotoSansDevanagari',        # Marathi
        'ne': 'NotoSansDevanagari',        # Nepali
        'ur': 'NotoNastaliqUrdu',          # Urdu
        'si': 'NotoSansSinhala',           # Sinhala
        'sd': 'NotoNastaliqUrdu',          # Sindhi (general code, assuming Arabic script by default for now)
        
        # Kannada script languages
        'kan_Knda': 'NotoSansKannada',  # Kannada (now with dedicated font)
        'kn': 'NotoSansKannada',        # Kannada (alternative code, now with dedicated font)
        
        # Tamil script languages
        'tam_Taml': 'AnekTamil',         # Tamil
        'ta': 'AnekTamil',             # Tamil (alternative code)
        
        # Telugu script languages
        'tel_Telu': 'AnekTelugu',        # Telugu
        'te': 'AnekTelugu',             # Telugu (alternative code)
        
        # Bengali script languages
        'asm_Beng': 'NotoSansBengali',   # Assamese (Bengali script)
        'ben_Beng': 'NotoSansBengali',   # Bengali
        'bn': 'NotoSansBengali',         # Bengali (alternative code)
        
        # Newly added languages
        'urd_Arab': 'NotoSansArabic',    # Urdu (Arabic script, prioritizing NotoSansArabic)
        'ur_Arab': 'NotoSansArabic',     # Urdu (alternative code, Arabic script)
        'ur': 'NotoSansArabic',          # Urdu (base code, Arabic script)
        'mal_Mlym': 'NotoSansMalayalam', # Malayalam
        'ml': 'NotoSansMalayalam',       # Malayalam (alternative code)
        'mar_Deva': 'TiroDevanagariMarathi', # Marathi (Devanagari script)
        'mr': 'TiroDevanagariMarathi',   # Marathi (base code)
        'ory_Orya': 'AnekOdia',          # Odia (prioritizing AnekOdia)
        'or': 'AnekOdia',                # Odia (alternative code)
        'pan_Guru': 'BalooPaaji2',       # Punjabi (Gurmukhi script)
        'pa': 'BalooPaaji2',             # Punjabi (alternative code)
        'mni_Mtei': 'NotoSansMeeteiMayek', # Manipuri (Meitei Mayek script)
        'mni_Beng': 'NotoSansBengali',   # Manipuri (Bengali script, if applicable)
        'guj_Gujr': 'AnekGujarati',       # Gujarati
        'gu': 'AnekGujarati',            # Gujarati (alternative code)
        'kas_Arab': 'NotoSansArabic',    # Kashmiri (Arabic script)
        'kas_Deva': 'NotoSansDevanagari', # Kashmiri (Devanagari script)
        'brx_Deva': 'NotoSansDevanagari', # Bodo (Devanagari script)
        'doi_Deva': 'NotoSansDevanagari', # Dogri (Devanagari script)
        'sat_Olck': 'NotoSansDevanagari', # Santali (Olchiki script - fallback to Devanagari)
    }
    
    # Fallback font hierarchy for each script system
    FALLBACK_FONTS = {
        'NotoSansDevanagari': ['NotoSansDevanagari', 'DejaVuSans', 'Arial', 'Helvetica'],
        'NotoNastaliqUrdu': ['NotoNastaliqUrdu', 'DejaVuSans', 'Arial', 'Helvetica'], # Keep for fallback
        'NotoSansArabic': ['NotoSansArabic', 'NotoNastaliqUrdu', 'DejaVuSans', 'Arial', 'Helvetica'], # New Arabic fallback
        'NotoSansKannada': ['NotoSansKannada', 'DejaVuSans', 'Arial', 'Helvetica'],
        'NotoSansSinhala': ['NotoSansSinhala', 'DejaVuSans', 'Arial', 'Helvetica'],
        'AnekTamil': ['AnekTamil', 'DejaVuSans', 'Arial', 'Helvetica'],
        'AnekTelugu': ['AnekTelugu', 'DejaVuSans', 'Arial', 'Helvetica'],
        'NotoSansBengali': ['NotoSansBengali', 'DejaVuSans', 'Arial', 'Helvetica'],
        'NotoSansMalayalam': ['NotoSansMalayalam', 'DejaVuSans', 'Arial', 'Helvetica'], # New Malayalam fallback
        'TiroDevanagariMarathi': ['TiroDevanagariMarathi', 'NotoSansDevanagari', 'DejaVuSans', 'Arial', 'Helvetica'], # New Marathi fallback
        'AnekOdia': ['AnekOdia', 'NotoSansOriya', 'DejaVuSans', 'Arial', 'Helvetica'], # New Odia fallback
        'NotoSansOriya': ['NotoSansOriya', 'DejaVuSans', 'Arial', 'Helvetica'], # Ensure NotoSansOriya is also a fallback
        'BalooPaaji2': ['BalooPaaji2', 'DejaVuSans', 'Arial', 'Helvetica'], # New Punjabi fallback
        'NotoSansMeeteiMayek': ['NotoSansMeeteiMayek', 'DejaVuSans', 'Arial', 'Helvetica'], # New Manipuri fallback
        'AnekGujarati': ['AnekGujarati', 'DejaVuSans', 'Arial', 'Helvetica'], # New Gujarati fallback
    }
    
    # Default fallback for unsupported languages
    DEFAULT_FALLBACK = ['NotoSansDevanagari', 'DejaVuSans', 'Arial', 'Helvetica', 'Times-Roman']
    
    def __init__(self, font_registry: FontRegistry, config: Optional[FontSystemConfig] = None):
        """
        Initialize the language mapper with a font registry and configuration.
        
        Args:
            font_registry: FontRegistry instance for checking font availability
            config: Font system configuration (optional)
        """
        self.font_registry = font_registry
        self.config = config or get_font_config()
        
        # Update fallback fonts from configuration if provided
        if self.config.preferences.fallback_fonts:
            self.FALLBACK_FONTS.update(self.config.preferences.fallback_fonts)
        
        logger.info("LanguageMapper initialized with configuration support")
    
    def get_font_for_language(self, language_code: str) -> str:
        """
        Get the best available font for a given language code with comprehensive error handling.
        
        Args:
            language_code: Language code (e.g., 'hin_Deva', 'hi', 'urd_Arab')
            
        Returns:
            Font family name that should be used for the language
        """
        return self.get_font_for_language_with_fallback(language_code)
    
    def get_font_for_language_with_fallback(self, language_code: str) -> str:
        """
        Get the best available font for a given language code with comprehensive fallback mechanisms.
        
        Args:
            language_code: Language code (e.g., 'hin_Deva', 'hi', 'urd_Arab')
            
        Returns:
            Font family name that should be used for the language
        """
        try:
            # Enhanced logging for font selection process
            logger.info(f"🔤 Font selection process started for language: '{language_code}'")
            
            # Handle empty or invalid language codes
            if not language_code or not isinstance(language_code, str):
                logger.warning("⚠️ Invalid language code provided (empty, None, or not string), using default fallback")
                fallback_font = self._get_fallback_font(self.DEFAULT_FALLBACK)
                logger.info(f"📝 Font selection result: '{fallback_font}' (reason: invalid language code)")
                return fallback_font
            
            # Normalize language code with error handling
            try:
                normalized_code = language_code.lower().strip()
                if not normalized_code:
                    logger.warning("⚠️ Language code is empty after normalization, using default fallback")
                    fallback_font = self._get_fallback_font(self.DEFAULT_FALLBACK)
                    logger.info(f"📝 Font selection result: '{fallback_font}' (reason: empty after normalization)")
                    return fallback_font
            except Exception as e:
                logger.warning(f"⚠️ Error normalizing language code '{language_code}': {e}, using default fallback")
                fallback_font = self._get_fallback_font(self.DEFAULT_FALLBACK)
                logger.info(f"📝 Font selection result: '{fallback_font}' (reason: normalization error)")
                return fallback_font
            
            logger.debug(f"🔄 Normalized language code: '{normalized_code}'")
            
            # Try exact match first with error handling
            try:
                preferred_font = self.LANGUAGE_FONT_MAP.get(normalized_code)
                logger.debug(f"🎯 Preferred font mapping for '{normalized_code}': '{preferred_font}'")
                
                if preferred_font:
                    # Check if preferred font family is available with error handling
                    try:
                        if self.font_registry.is_family_available(preferred_font):
                            logger.debug(f"✅ Preferred font '{preferred_font}' is available.")
                            logger.info(f"✅ Using preferred font '{preferred_font}' for language '{language_code}'")
                            logger.info(f"📝 Font selection result: '{preferred_font}' (reason: exact match, font available)")
                            return preferred_font
                        else:
                            logger.warning(f"❌ Preferred font '{preferred_font}' not available for language '{language_code}', trying fallbacks")
                            fallback_font = self._get_fallback_font(self.FALLBACK_FONTS.get(preferred_font, self.DEFAULT_FALLBACK))
                            logger.info(f"📝 Font selection result: '{fallback_font}' (reason: preferred font unavailable, using fallback)")
                            return fallback_font
                    except Exception as e:
                        logger.warning(f"⚠️ Error checking font availability for '{preferred_font}': {e}")
                        # Continue to fallback mechanisms
                        
            except Exception as e:
                logger.warning(f"⚠️ Error during exact font mapping lookup for '{normalized_code}': {e}")
                # Continue to fallback mechanisms
            
            # Try partial matches for language codes with script suffixes
            try:
                if '_' in normalized_code:
                    base_lang = normalized_code.split('_')[0]
                    base_font = self.LANGUAGE_FONT_MAP.get(base_lang)
                    logger.debug(f"🔍 Trying base language '{base_lang}' -> font '{base_font}'")
                    
                    if base_font:
                        try:
                            if self.font_registry.is_family_available(base_font):
                                logger.debug(f"✅ Base language font '{base_font}' is available.")
                                logger.info(f"✅ Using base language font '{base_font}' for language '{language_code}'")
                                logger.info(f"📝 Font selection result: '{base_font}' (reason: base language match)")
                                return base_font
                        except Exception as e:
                            logger.warning(f"⚠️ Error checking base language font '{base_font}': {e}")
                            
            except Exception as e:
                logger.warning(f"⚠️ Error during base language matching for '{normalized_code}': {e}")
            
            # Try script-based matching with error handling
            try:
                script_font = self._get_font_by_script(normalized_code)
                logger.debug(f"🔍 Script-based font detection for '{normalized_code}': '{script_font}'")
                if script_font:
                    logger.info(f"✅ Using script-based font '{script_font}' for language '{language_code}'")
                    logger.info(f"📝 Font selection result: '{script_font}' (reason: script detection)")
                    return script_font
            except Exception as e:
                logger.warning(f"⚠️ Error during script-based font matching for '{normalized_code}': {e}")
            
            # No specific mapping found, use default fallback
            logger.warning(f"⚠️ No specific font mapping found for language '{language_code}', using default fallback")
            fallback_font = self._get_fallback_font(self.DEFAULT_FALLBACK)
            logger.info(f"📝 Font selection result: '{fallback_font}' (reason: no mapping found, default fallback)")
            return fallback_font
            
        except Exception as e:
            # Critical error in font selection - use emergency fallback
            error_msg = f"Critical error in font selection for language '{language_code}': {e}"
            logger.error(f"❌ {error_msg}")
            
            # Emergency fallback to most basic system font
            emergency_fallback = 'Helvetica'
            logger.error(f"🚨 Using emergency fallback font: {emergency_fallback}")
            logger.error("🚨 PDF generation may have significant rendering issues")
            
            return emergency_fallback
    
    def _get_font_by_script(self, language_code: str) -> Optional[str]:
        """
        Try to determine font based on script indicators in language code.
        
        Args:
            language_code: Normalized language code
            
        Returns:
            Font family name if script can be determined, None otherwise
        """
        # Check for script indicators in the language code
        if 'deva' in language_code:
            if self.font_registry.is_family_available('NotoSansDevanagari'):
                logger.debug(f"Detected Devanagari script in '{language_code}', using NotoSansDevanagari")
                return 'NotoSansDevanagari'
        elif 'knda' in language_code:
            if self.font_registry.is_family_available('NotoSansKannada'):
                logger.debug(f"Detected Kannada script in '{language_code}', using NotoSansKannada")
                return 'NotoSansKannada'
        elif 'taml' in language_code:
            if self.font_registry.is_family_available('AnekTamil'):
                logger.debug(f"Detected Tamil script in '{language_code}', using AnekTamil")
                return 'AnekTamil'
        elif 'telu' in language_code:
            if self.font_registry.is_family_available('AnekTelugu'):
                logger.debug(f"Detected Telugu script in '{language_code}', using AnekTelugu")
                return 'AnekTelugu'
        elif 'beng' in language_code:
            if self.font_registry.is_family_available('NotoSansBengali'):
                logger.debug(f"Detected Bengali script in '{language_code}', using NotoSansBengali")
                return 'NotoSansBengali'
        elif 'mlym' in language_code:
            if self.font_registry.is_family_available('NotoSansMalayalam'):
                logger.debug(f"Detected Malayalam script in '{language_code}', using NotoSansMalayalam")
                return 'NotoSansMalayalam'
        elif 'orya' in language_code:
            if self.font_registry.is_family_available('AnekOdia'): # Prioritize AnekOdia
                logger.debug(f"Detected Odia script in '{language_code}', using AnekOdia")
                return 'AnekOdia'
            elif self.font_registry.is_family_available('NotoSansOriya'):
                logger.debug(f"Detected Odia script in '{language_code}', using NotoSansOriya fallback")
                return 'NotoSansOriya'
        elif 'guru' in language_code:
            if self.font_registry.is_family_available('BalooPaaji2'):
                logger.debug(f"Detected Gurumukhi script in '{language_code}', using BalooPaaji2")
                return 'BalooPaaji2'
        elif 'mtei' in language_code:
            if self.font_registry.is_family_available('NotoSansMeeteiMayek'):
                logger.debug(f"Detected Meitei Mayek script in '{language_code}', using NotoSansMeeteiMayek")
                return 'NotoSansMeeteiMayek'
        elif 'gujr' in language_code:
            if self.font_registry.is_family_available('AnekGujarati'):
                logger.debug(f"Detected Gujarati script in '{language_code}', using AnekGujarati")
                return 'AnekGujarati'
        elif 'olck' in language_code:
            # No specific Olchiki font, fallback to Devanagari for now
            if self.font_registry.is_family_available('NotoSansDevanagari'):
                logger.debug(f"Detected Olchiki script in '{language_code}', falling back to NotoSansDevanagari")
                return 'NotoSansDevanagari'
        elif 'arab' in language_code:
            if self.font_registry.is_family_available('NotoSansArabic'): # Prioritize NotoSansArabic
                logger.debug(f"Detected Arabic script in '{language_code}', using NotoSansArabic")
                return 'NotoSansArabic'
            elif self.font_registry.is_family_available('NotoNastaliqUrdu'): # Fallback to Nastaliq
                logger.debug(f"Detected Arabic script in '{language_code}', falling back to NotoNastaliqUrdu")
                return 'NotoNastaliqUrdu'
        elif 'sinh' in language_code:
            if self.font_registry.is_family_available('NotoSansSinhala'):
                logger.debug(f"Detected Sinhala script in '{language_code}', using NotoSansSinhala")
                return 'NotoSansSinhala'
        
        logger.debug(f"No script-based font detected for '{language_code}'")
        return None
    
    def _get_fallback_font(self, fallback_list: List[str]) -> str:
        """
        Get the first available font from a fallback list with enhanced error handling.
        
        Args:
            fallback_list: List of font family names in preference order
            
        Returns:
            First available font family name, or system fallback if none available
        """
        logger.debug(f"🔍 Searching for fallback font from list: {fallback_list}")
        
        # Check if fallback is enabled
        if not self.font_registry.is_fallback_enabled():
            logger.warning("⚠️ Font fallback is disabled, using system default")
            return 'Helvetica'
        
        # Try each font in the fallback list
        for i, font_family in enumerate(fallback_list):
            try:
                if self.font_registry.is_family_available(font_family):
                    logger.debug(f"✅ Using fallback font #{i+1}: {font_family}")
                    return font_family
                else:
                    logger.debug(f"❌ Fallback font #{i+1} not available: {font_family}")
            except Exception as e:
                logger.warning(f"⚠️ Error checking fallback font {font_family}: {e}")
                continue
        
        # Enhanced system font fallback with platform detection
        system_fallbacks = self._get_system_fallback_fonts()
        
        for font_family in system_fallbacks:
            try:
                if self.font_registry.is_family_available(font_family):
                    logger.info(f"📝 Using system fallback font: {font_family}")
                    return font_family
            except Exception as e:
                logger.warning(f"⚠️ Error checking system font {font_family}: {e}")
                continue
        
        # Final fallback - use the last font from original list or default
        final_fallback = fallback_list[-1] if fallback_list else 'Helvetica'
        logger.warning(f"⚠️ No fallback fonts available, using final fallback: {final_fallback}")
        logger.warning("⚠️ PDF generation may have rendering issues with unsupported characters")
        
        return final_fallback
    
    def _get_system_fallback_fonts(self) -> List[str]:
        """
        Get platform-appropriate system fallback fonts.
        
        Returns:
            List of system font names in preference order
        """
        import platform
        system = platform.system().lower()
        
        if system == 'windows':
            return ['Arial', 'Calibri', 'Segoe UI', 'Tahoma', 'Verdana', 'Times New Roman']
        elif system == 'darwin':  # macOS
            return ['Helvetica', 'Arial', 'San Francisco', 'Lucida Grande', 'Times']
        else:  # Linux and others
            return ['DejaVu Sans', 'Liberation Sans', 'Ubuntu', 'Helvetica', 'Arial']
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of all supported language codes.
        
        Returns:
            List of supported language codes
        """
        return list(self.LANGUAGE_FONT_MAP.keys())
    
    def get_available_languages(self) -> List[str]:
        """
        Get list of language codes that have available fonts.
        
        Returns:
            List of language codes with available fonts
        """
        available_languages = []
        
        for lang_code, font_family in self.LANGUAGE_FONT_MAP.items():
            if self.font_registry.is_family_available(font_family):
                available_languages.append(lang_code)
        
        return available_languages
    
    def get_font_info_for_language(self, language_code: str) -> Dict:
        """
        Get detailed font information for a language.
        
        Args:
            language_code: Language code to get info for
            
        Returns:
            Dictionary with font information
        """
        normalized_code = language_code.lower().strip()
        
        # Try to get preferred font through the same logic as get_font_for_language
        preferred_font = self.LANGUAGE_FONT_MAP.get(normalized_code)
        
        # If no direct match, try base language
        if not preferred_font and '_' in normalized_code:
            base_lang = normalized_code.split('_')[0]
            preferred_font = self.LANGUAGE_FONT_MAP.get(base_lang)
        
        # If still no match, try script-based detection
        if not preferred_font:
            if 'deva' in normalized_code:
                preferred_font = 'NotoSansDevanagari'
            elif 'arab' in normalized_code:
                preferred_font = 'NotoNastaliqUrdu'
            elif 'sinh' in normalized_code:
                preferred_font = 'NotoSansSinhala'
        
        selected_font = self.get_font_for_language(language_code)
        is_supported = self.is_language_supported(language_code)
        
        return {
            'language_code': language_code,
            'normalized_code': normalized_code,
            'preferred_font': preferred_font,
            'selected_font': selected_font,
            'is_preferred_available': preferred_font and self.font_registry.is_family_available(preferred_font),
            'is_supported': is_supported,
            'fallback_fonts': self.FALLBACK_FONTS.get(preferred_font, self.DEFAULT_FALLBACK) if preferred_font else self.DEFAULT_FALLBACK
        }
    
    def is_language_supported(self, language_code: str) -> bool:
        """
        Check if a language code has explicit font support.
        
        Args:
            language_code: Language code to check
            
        Returns:
            True if language has explicit support, False otherwise
        """
        normalized_code = language_code.lower().strip()
        
        # Check direct mapping
        if normalized_code in self.LANGUAGE_FONT_MAP:
            return True
        
        # Check base language for codes with script suffixes
        if '_' in normalized_code:
            base_lang = normalized_code.split('_')[0]
            if base_lang in self.LANGUAGE_FONT_MAP:
                return True
        
        # Check script-based support
        return self._get_font_by_script(normalized_code) is not None
    
    def get_font_with_weight(self, language_code: str, weight: str = 'Regular') -> Optional[str]:
        """
        Get a specific font name with weight for a language.
        
        Args:
            language_code: Language code
            weight: Font weight (e.g., 'Regular', 'Bold', 'Medium')
            
        Returns:
            Specific font name if available, None otherwise
        """
        font_family = self.get_font_for_language(language_code)
        return self.font_registry.get_font_by_family_and_weight(font_family, weight)
    
    def get_mapping_summary(self) -> Dict:
        """
        Get a summary of language mappings and availability.
        
        Returns:
            Dictionary with mapping statistics and information
        """
        total_mappings = len(self.LANGUAGE_FONT_MAP)
        available_mappings = len(self.get_available_languages())
        
        # Group languages by font family
        languages_by_font = {}
        for lang_code, font_family in self.LANGUAGE_FONT_MAP.items():
            if font_family not in languages_by_font:
                languages_by_font[font_family] = []
            languages_by_font[font_family].append(lang_code)
        
        return {
            'total_language_mappings': total_mappings,
            'available_language_mappings': available_mappings,
            'unavailable_mappings': total_mappings - available_mappings,
            'supported_languages': self.get_supported_languages(),
            'available_languages': self.get_available_languages(),
            'languages_by_font_family': languages_by_font,
            'fallback_fonts': self.FALLBACK_FONTS,
            'default_fallback': self.DEFAULT_FALLBACK
        }
    
    def test_language_mapping(self, language_code: str) -> Dict:
        """
        Test font mapping for a specific language and return detailed results.
        
        Args:
            language_code: Language code to test
            
        Returns:
            Dictionary with detailed test results
        """
        logger.info(f"🧪 Testing font mapping for language: '{language_code}'")
        
        test_result = {
            'language_code': language_code,
            'test_timestamp': str(os.times()),
            'mapping_steps': [],
            'final_result': {},
            'warnings': [],
            'errors': []
        }
        
        # Step 1: Input validation
        if not language_code:
            test_result['errors'].append("Empty language code provided")
            test_result['final_result'] = {'status': 'error', 'font': None}
            return test_result
        
        test_result['mapping_steps'].append({
            'step': 'input_validation',
            'status': 'pass',
            'details': f"Language code '{language_code}' is valid"
        })
        
        # Step 2: Normalization
        normalized_code = language_code.lower().strip()
        test_result['mapping_steps'].append({
            'step': 'normalization',
            'status': 'pass',
            'details': f"Normalized to '{normalized_code}'"
        })
        
        # Step 3: Direct mapping lookup
        preferred_font = self.LANGUAGE_FONT_MAP.get(normalized_code)
        if preferred_font:
            test_result['mapping_steps'].append({
                'step': 'direct_mapping',
                'status': 'found',
                'details': f"Direct mapping found: '{normalized_code}' -> '{preferred_font}'"
            })
            
            # Check font availability
            if self.font_registry.is_family_available(preferred_font):
                test_result['mapping_steps'].append({
                    'step': 'font_availability',
                    'status': 'available',
                    'details': f"Font family '{preferred_font}' is available"
                })
                test_result['final_result'] = {
                    'status': 'success',
                    'font': preferred_font,
                    'method': 'direct_mapping'
                }
                return test_result
            else:
                test_result['mapping_steps'].append({
                    'step': 'font_availability',
                    'status': 'unavailable',
                    'details': f"Font family '{preferred_font}' is not available"
                })
                test_result['warnings'].append(f"Preferred font '{preferred_font}' not available")
        else:
            test_result['mapping_steps'].append({
                'step': 'direct_mapping',
                'status': 'not_found',
                'details': f"No direct mapping found for '{normalized_code}'"
            })
        
        # Step 4: Base language lookup
        if '_' in normalized_code:
            base_lang = normalized_code.split('_')[0]
            base_font = self.LANGUAGE_FONT_MAP.get(base_lang)
            test_result['mapping_steps'].append({
                'step': 'base_language_lookup',
                'status': 'found' if base_font else 'not_found',
                'details': f"Base language '{base_lang}' -> '{base_font}'" if base_font else f"No mapping for base language '{base_lang}'"
            })
            
            if base_font and self.font_registry.is_family_available(base_font):
                test_result['final_result'] = {
                    'status': 'success',
                    'font': base_font,
                    'method': 'base_language_mapping'
                }
                return test_result
        else:
            test_result['mapping_steps'].append({
                'step': 'base_language_lookup',
                'status': 'skipped',
                'details': "No script suffix found in language code"
            })
        
        # Step 5: Script detection
        script_font = self._get_font_by_script(normalized_code)
        if script_font:
            test_result['mapping_steps'].append({
                'step': 'script_detection',
                'status': 'found',
                'details': f"Script-based font detected: '{script_font}'"
            })
            test_result['final_result'] = {
                'status': 'success',
                'font': script_font,
                'method': 'script_detection'
            }
            return test_result
        else:
            test_result['mapping_steps'].append({
                'step': 'script_detection',
                'status': 'not_found',
                'details': "No script-based font mapping found"
            })
        
        # Step 6: Fallback
        fallback_font = self._get_fallback_font(self.DEFAULT_FALLBACK)
        test_result['mapping_steps'].append({
            'step': 'fallback',
            'status': 'applied',
            'details': f"Using default fallback: '{fallback_font}'"
        })
        
        test_result['warnings'].append("No specific font mapping found, using fallback")
        test_result['final_result'] = {
            'status': 'fallback',
            'font': fallback_font,
            'method': 'default_fallback'
        }
        
        return test_result
    
    def get_language_coverage_report(self) -> Dict:
        """
        Generate a comprehensive report of language coverage and font availability.
        
        Returns:
            Dictionary with detailed coverage report
        """
        logger.info("📊 Generating language coverage report...")
        
        report = {
            'report_timestamp': str(os.times()),
            'summary': {},
            'by_script': {},
            'by_availability': {},
            'recommendations': []
        }
        
        # Summary statistics
        total_languages = len(self.LANGUAGE_FONT_MAP)
        available_languages = len(self.get_available_languages())
        
        report['summary'] = {
            'total_supported_languages': total_languages,
            'languages_with_available_fonts': available_languages,
            'languages_without_fonts': total_languages - available_languages,
            'coverage_percentage': (available_languages / total_languages * 100) if total_languages > 0 else 0
        }
        
        # Group by script
        script_groups = {
            'Devanagari': [],
            'Arabic/Urdu': [],
            'Sinhala': [],
            'Other': []
        }
        
        for lang_code, font_family in self.LANGUAGE_FONT_MAP.items():
            if 'Devanagari' in font_family:
                script_groups['Devanagari'].append(lang_code)
            elif 'Urdu' in font_family or 'Arab' in lang_code:
                script_groups['Arabic/Urdu'].append(lang_code)
            elif 'Sinhala' in font_family:
                script_groups['Sinhala'].append(lang_code)
            else:
                script_groups['Other'].append(lang_code)
        
        for script, languages in script_groups.items():
            if languages:
                available_in_script = [lang for lang in languages if lang in self.get_available_languages()]
                report['by_script'][script] = {
                    'total_languages': len(languages),
                    'available_languages': len(available_in_script),
                    'language_codes': languages,
                    'available_codes': available_in_script,
                    'coverage_percentage': (len(available_in_script) / len(languages) * 100) if languages else 0
                }
        
        # Group by availability
        available_langs = self.get_available_languages()
        unavailable_langs = [lang for lang in self.LANGUAGE_FONT_MAP.keys() if lang not in available_langs]
        
        report['by_availability'] = {
            'available': {
                'count': len(available_langs),
                'languages': available_langs
            },
            'unavailable': {
                'count': len(unavailable_langs),
                'languages': unavailable_langs,
                'missing_fonts': list(set(self.LANGUAGE_FONT_MAP[lang] for lang in unavailable_langs))
            }
        }
        
        # Generate recommendations
        if unavailable_langs:
            missing_fonts = set(self.LANGUAGE_FONT_MAP[lang] for lang in unavailable_langs)
            report['recommendations'].append(f"Install missing fonts: {', '.join(missing_fonts)}")
        
        if report['summary']['coverage_percentage'] < 100:
            report['recommendations'].append("Improve font coverage by installing all required Noto fonts")
        
        if report['summary']['coverage_percentage'] < 50:
            report['recommendations'].append("Critical: Less than 50% language coverage - check font installation")
        
        return report


# Global font registry instance (initialized with configuration)
font_registry = FontRegistry()


def initialize_font_system(font_base_path: str = None, config: Optional[FontSystemConfig] = None) -> Tuple[int, int]:
    """
    Initialize the font system by registering fonts from the base directory with comprehensive error handling.
    
    Args:
        font_base_path: Base path for font directory (uses configuration default if None)
        config: Font system configuration (loads from environment if None)
        
    Returns:
        Tuple of (successful_registrations, failed_registrations)
    """
    successful = 0
    failed = 0
    
    try:
        # Load configuration if not provided
        if config is None:
            config = get_font_config()
        
        if font_base_path is None:
            # Use configured font path
            font_base_path = config.font_base_path
        
        logger.info(f"🔤 Initializing font system with base path: {font_base_path}")
        
        # Validate font base path before proceeding
        if not font_base_path:
            logger.warning("⚠️ Font base path is empty, skipping font registration")
            return 0, 0
        
        # Register fonts with comprehensive error handling
        try:
            successful, failed = font_registry.register_fonts_from_directory(font_base_path)
            
            # Attempt error recovery if there were failures
            if failed > 0:
                logger.info("🔧 Attempting font error recovery...")
                recovery_results = font_registry.recover_from_font_errors()
                if recovery_results['recovery_successful']:
                    logger.info(f"✅ Font error recovery completed: cleaned {recovery_results['invalid_registrations_cleaned']} invalid registrations")
                else:
                    logger.warning(f"⚠️ Font error recovery had issues: {recovery_results.get('errors', [])}")
            
            # Log final status
            if successful > 0:
                logger.info(f"✅ Font system initialization completed: {successful} fonts registered, {failed} failed")
            elif failed > 0:
                logger.warning(f"⚠️ Font system initialization completed with issues: 0 fonts registered, {failed} failed")
                logger.warning("⚠️ PDF generation will use system default fonts")
            else:
                logger.warning("⚠️ No fonts found in directory - PDF generation will use system default fonts")
            
            # Perform health check
            try:
                health_check = font_registry.perform_health_check()
                if health_check['overall_status'] != 'healthy':
                    logger.warning(f"⚠️ Font system health check indicates issues: {health_check['overall_status']}")
                    for warning in health_check.get('warnings', []):
                        logger.warning(f"⚠️ Health check warning: {warning}")
                    for error in health_check.get('errors', []):
                        logger.error(f"❌ Health check error: {error}")
            except Exception as e:
                logger.warning(f"⚠️ Font system health check failed: {e}")
            
            return successful, failed
            
        except Exception as e:
            error_msg = f"Font registration failed: {e}"
            logger.error(f"❌ {error_msg}")
            
            # Enable fallback mode for graceful degradation
            try:
                font_registry.enable_fallback_mode(True)
                logger.info("🔄 Enabled font fallback mode for graceful degradation")
            except Exception as fallback_error:
                logger.error(f"❌ Could not enable fallback mode: {fallback_error}")
            
            return 0, 1
        
    except Exception as e:
        error_msg = f"Critical failure in font system initialization: {e}"
        logger.error(f"❌ {error_msg}")
        
        # Emergency fallback setup
        try:
            logger.info("🚨 Attempting emergency font system setup...")
            font_registry.enable_fallback_mode(True)
            font_registry.clear_registry()  # Clear any partial state
            logger.warning("⚠️ Emergency font system setup completed - limited functionality available")
            logger.warning("⚠️ PDF generation will use system default fonts only")
        except Exception as emergency_error:
            logger.error(f"❌ Emergency font system setup also failed: {emergency_error}")
            logger.error("❌ Font system is not available - PDF generation may fail")
        
        return 0, 1


def get_font_registry() -> FontRegistry:
    """
    Get the global font registry instance.
    
    Returns:
        FontRegistry instance
    """
    return font_registry


def create_language_mapper(config: Optional[FontSystemConfig] = None) -> LanguageMapper:
    """
    Create a new LanguageMapper instance using the global font registry and configuration.
    
    Args:
        config: Font system configuration (loads from environment if None)
    
    Returns:
        LanguageMapper instance
    """
    return LanguageMapper(font_registry, config)


# Global language mapper instance (created after font registry)
_language_mapper = None


def get_language_mapper() -> LanguageMapper:
    """
    Get the global language mapper instance, creating it if necessary.
    
    Returns:
        LanguageMapper instance
    """
    global _language_mapper
    if _language_mapper is None:
        _language_mapper = create_language_mapper()
    return _language_mapper


@dataclass
class StyleConfig:
    """Configuration for creating paragraph styles"""
    font_family: str
    font_size: int
    leading: int
    space_after: int
    text_color: str = '#000000'
    alignment: int = TA_LEFT
    space_before: int = 0
    left_indent: int = 0
    right_indent: int = 0
    first_line_indent: int = 0
    bold: bool = False
    italic: bool = False


class EnhancedStyleFactory:
    """
    Factory class for creating ReportLab paragraph styles with font-aware capabilities.
    
    This class provides:
    - Font-aware style creation for different text types (title, heading, body)
    - Automatic font selection based on language codes
    - Fallback style creation when fonts are not available
    - Consistent styling across different languages
    """
    
    # Default style configurations for different text types
    DEFAULT_STYLE_CONFIGS = {
        'title': StyleConfig(
            font_family='Helvetica-Bold',
            font_size=18,
            leading=22,
            space_after=20,
            space_before=0,
            text_color='#1f2937',
            alignment=TA_CENTER,
            bold=True
        ),
        'heading1': StyleConfig(
            font_family='Helvetica-Bold',
            font_size=16,
            leading=20,
            space_after=16,
            space_before=12,
            text_color='#374151',
            alignment=TA_LEFT,
            bold=True
        ),
        'heading2': StyleConfig(
            font_family='Helvetica-Bold',
            font_size=14,
            leading=18,
            space_after=12,
            space_before=10,
            text_color='#4b5563',
            alignment=TA_LEFT,
            bold=True
        ),
        'heading3': StyleConfig(
            font_family='Helvetica-Bold',
            font_size=12,
            leading=16,
            space_after=10,
            space_before=8,
            text_color='#6b7280',
            alignment=TA_LEFT,
            bold=True
        ),
        'body': StyleConfig(
            font_family='Helvetica',
            font_size=11,
            leading=16,
            space_after=12,
            space_before=0,
            text_color='#000000',
            alignment=TA_LEFT
        ),
        'body_indent': StyleConfig(
            font_family='Helvetica',
            font_size=11,
            leading=16,
            space_after=12,
            space_before=0,
            text_color='#000000',
            alignment=TA_LEFT,
            left_indent=20
        ),
        'caption': StyleConfig(
            font_family='Helvetica',
            font_size=9,
            leading=12,
            space_after=8,
            space_before=4,
            text_color='#6b7280',
            alignment=TA_CENTER,
            italic=True
        ),
        'quote': StyleConfig(
            font_family='Helvetica',
            font_size=10,
            leading=14,
            space_after=12,
            space_before=8,
            text_color='#374151',
            alignment=TA_LEFT,
            left_indent=30,
            right_indent=30,
            italic=True
        )
    }
    
    def __init__(self, font_registry: FontRegistry, language_mapper: LanguageMapper = None, config: Optional[FontSystemConfig] = None):
        """
        Initialize the enhanced style factory with configuration support.
        
        Args:
            font_registry: FontRegistry instance for checking font availability
            language_mapper: LanguageMapper instance for language-to-font mapping
            config: Font system configuration (optional)
        """
        self.font_registry = font_registry
        self.language_mapper = language_mapper or get_language_mapper()
        self.config = config or get_font_config()
        self._style_cache = {}
        
        # Configure cache size based on configuration
        self._max_cache_size = self.config.cache_config.max_cache_size if self.config.cache_config.enable_font_caching else 0
        
        logger.info("EnhancedStyleFactory initialized with configuration support")
    
    def create_styles_for_language(self, language_code: str) -> Dict[str, ParagraphStyle]:
        """
        Create a complete set of paragraph styles for a specific language with comprehensive error handling.
        
        Args:
            language_code: Language code (e.g., 'hin_Deva', 'urd_Arab', 'sin_Sinh')
            
        Returns:
            Dictionary mapping style names to ParagraphStyle objects
        """
        try:
            # Validate language code
            if not language_code or not isinstance(language_code, str):
                logger.warning("⚠️ Invalid language code provided, using 'unknown'")
                language_code = "unknown"
            
            # Sanitize language code for cache key
            safe_language_code = language_code.replace(' ', '_').replace('-', '_')
            cache_key = f"styles_{safe_language_code}"
            
            # Check cache first with error handling (if caching is enabled)
            if self._max_cache_size > 0:
                try:
                    if cache_key in self._style_cache:
                        logger.debug(f"Using cached styles for language: {language_code}")
                        return self._style_cache[cache_key]
                except Exception as e:
                    logger.warning(f"⚠️ Error accessing style cache for {language_code}: {e}")
                    # Clear corrupted cache entry
                    self._style_cache.pop(cache_key, None)
            
            logger.info(f"🎨 Creating styles for language: {language_code}")
            
            # Get the appropriate font family for this language with error handling
            try:
                if self.language_mapper:
                    font_info = self.language_mapper.get_font_info_for_language(language_code)
                    selected_font = font_info['selected_font']
                    is_supported = font_info['is_supported']
                    logger.info(f"🌐 Language Mapper suggests font: '{selected_font}' (Supported: {is_supported})")
                    
                    # If the selected font is not explicitly registered, it might be a ReportLab built-in or a fallback handled by ReportLab itself.
                    # We will proceed with this selected_font and let ReportLab's style creation handle the final resolution.
                    if not pdfmetrics.getFont(selected_font):
                        logger.warning(f"⚠️ Selected font '{selected_font}' from Language Mapper is not directly registered. It might be a ReportLab built-in or a family name.")

                else:
                    logger.warning("⚠️ Language mapper not available, using system fallback 'Helvetica'")
                    selected_font = "Helvetica"
                    is_supported = False # Assume not supported if mapper isn't there
            except Exception as e:
                logger.warning(f"⚠️ Error getting font info for language {language_code}: {e}, using system fallback 'Helvetica'")
                selected_font = "Helvetica"
                is_supported = False
            
            # Create styles for each text type with error handling
            styles = {}
            successful_styles = 0
            failed_styles = 0
            
            for style_name, base_config in self.DEFAULT_STYLE_CONFIGS.items():
                try:
                    style = self._create_single_style(
                        style_name=style_name,
                        config=base_config,
                        font_family=selected_font, # Use selected_font as the primary font family for style creation
                        language_code=language_code
                    )
                    styles[style_name] = style
                    successful_styles += 1
                    
                except Exception as e:
                    logger.error(f"❌ Failed to create style '{style_name}' for {language_code}: {e}")
                    # Create emergency fallback style
                    try:
                        styles[style_name] = self._create_emergency_style(style_name, language_code)
                        logger.warning(f"⚠️ Using emergency style for '{style_name}'")
                        failed_styles += 1
                    except Exception as emergency_error:
                        logger.error(f"❌ Even emergency style creation failed for '{style_name}': {emergency_error}")
                        # Skip this style entirely
                        failed_styles += 1
            
            # Ensure we have at least basic styles
            if not styles:
                logger.error(f"❌ No styles could be created for {language_code}, creating minimal fallback set")
                styles = self._create_minimal_fallback_styles(language_code)
            
            # Cache the styles with error handling and size management
            try:
                if self._max_cache_size > 0:
                    # Check if cache is full and needs cleanup
                    if len(self._style_cache) >= self._max_cache_size:
                        self._cleanup_cache()
                    
                    self._style_cache[cache_key] = styles
                    logger.debug(f"Cached styles for {language_code} (cache size: {len(self._style_cache)})")
                else:
                    logger.debug(f"Style caching disabled by configuration")
            except Exception as e:
                logger.warning(f"⚠️ Could not cache styles for {language_code}: {e}")
            
            logger.info(f"✅ Created {successful_styles} styles for language {language_code} using font family: {selected_font}")
            if failed_styles > 0:
                logger.warning(f"⚠️ {failed_styles} styles had issues and used fallbacks")
            
            return styles
            
        except Exception as e:
            logger.error(f"❌ Critical error creating styles for language '{language_code}': {e}")
            # Return minimal emergency styles
            return self._create_minimal_fallback_styles(language_code or "unknown")
    
    def _create_minimal_fallback_styles(self, language_code: str) -> Dict[str, ParagraphStyle]:
        """
        Create a minimal set of fallback styles when all other methods fail.
        
        Args:
            language_code: Language code
            
        Returns:
            Dictionary with minimal style set
        """
        try:
            from reportlab.lib.styles import getSampleStyleSheet
            sample_styles = getSampleStyleSheet()
            
            return {
                'title': sample_styles['Title'],
                'heading1': sample_styles['Heading1'],
                'heading2': sample_styles['Heading2'],
                'body': sample_styles['Normal'],
                'caption': sample_styles['Normal']
            }
        except Exception as e:
            logger.error(f"❌ Even minimal fallback styles creation failed: {e}")
            # Return absolutely minimal style
            try:
                minimal_style = ParagraphStyle(
                    name=f"minimal_{language_code}",
                    fontName='Helvetica',
                    fontSize=11,
                    leading=16
                )
                return {
                    'title': minimal_style,
                    'heading1': minimal_style,
                    'heading2': minimal_style,
                    'body': minimal_style,
                    'caption': minimal_style
                }
            except Exception:
                logger.error("❌ Absolutely minimal style creation failed - returning empty dict")
                return {}
    
    def _create_single_style(self, style_name: str, config: StyleConfig, 
                           font_family: str, language_code: str) -> ParagraphStyle:
        """
        Create a single paragraph style with the specified configuration and comprehensive error handling.
        
        Args:
            style_name: Name of the style
            config: StyleConfig with style parameters
            font_family: Font family to use
            language_code: Language code for context
            
        Returns:
            ParagraphStyle object
        """
        try:
            logger.debug(f"🎨 Creating style '{style_name}' for language '{language_code}'")
            logger.debug(f"  - Requested font family: {font_family}")
            logger.debug(f"  - Style config: bold={getattr(config, 'bold', False)}, italic={getattr(config, 'italic', False)}, size={getattr(config, 'font_size', 11)}")
            
            # Validate inputs with error handling
            if not style_name:
                logger.warning("⚠️ Style name is empty, using default")
                style_name = "default"
            
            if not language_code:
                logger.warning("⚠️ Language code is empty, using 'unknown'")
                language_code = "unknown"
            
            if not font_family:
                logger.warning("⚠️ Font family is empty, using system fallback")
                font_family = "Helvetica"
            
            # Determine the actual font to use with error handling
            try:
                actual_font = self._get_actual_font_with_fallback(font_family, config, language_code)
                logger.debug(f"  - Resolved font: {actual_font}")
            except Exception as e:
                logger.warning(f"⚠️ Error resolving font for {font_family}: {e}, using emergency fallback")
                actual_font = self._get_emergency_fallback_font(config)
            
            # Convert color string to ReportLab color with error handling
            try:
                text_color = self._parse_color(getattr(config, 'text_color', '#000000'))
            except Exception as e:
                logger.warning(f"⚠️ Error parsing color '{getattr(config, 'text_color', 'unknown')}': {e}, using black")
                text_color = colors.black
            
            # Validate and sanitize style parameters
            try:
                font_size = max(6, min(72, getattr(config, 'font_size', 11)))  # Clamp between 6-72pt
                leading = max(font_size, getattr(config, 'leading', font_size * 1.2))  # Leading should be >= font size
                space_after = max(0, getattr(config, 'space_after', 12))
                space_before = max(0, getattr(config, 'space_before', 0))
                left_indent = max(0, getattr(config, 'left_indent', 0))
                right_indent = max(0, getattr(config, 'right_indent', 0))
                first_line_indent = getattr(config, 'first_line_indent', 0)
                alignment = getattr(config, 'alignment', TA_LEFT)
                
                # Validate alignment value
                if alignment not in [TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY]:
                    logger.warning(f"⚠️ Invalid alignment value {alignment}, using left alignment")
                    alignment = TA_LEFT
                    
            except Exception as e:
                logger.warning(f"⚠️ Error validating style parameters: {e}, using defaults")
                font_size = 11
                leading = 16
                space_after = 12
                space_before = 0
                left_indent = 0
                right_indent = 0
                first_line_indent = 0
                alignment = TA_LEFT
            
            # Create the style with error handling
            try:
                style_name_safe = f"{style_name}_{language_code}".replace(' ', '_').replace('-', '_')
                
                style = ParagraphStyle(
                    name=style_name_safe,
                    fontName=actual_font,
                    fontSize=font_size,
                    leading=leading,
                    spaceAfter=space_after,
                    spaceBefore=space_before,
                    leftIndent=left_indent,
                    rightIndent=right_indent,
                    firstLineIndent=first_line_indent,
                    alignment=alignment,
                    textColor=text_color
                )
                
                logger.info(f"✅ Created style '{style_name}' for {language_code}:")
                logger.info(f"  - Font: {actual_font}")
                logger.info(f"  - Size: {font_size}pt")
                logger.info(f"  - Leading: {leading}pt")
                logger.info(f"  - Color: {getattr(config, 'text_color', '#000000')}")
                
                return style
                
            except Exception as e:
                logger.error(f"❌ Failed to create ParagraphStyle: {e}")
                # Create minimal emergency style
                return self._create_emergency_style(style_name, language_code)
                
        except Exception as e:
            logger.error(f"❌ Critical error creating style '{style_name}' for '{language_code}': {e}")
            # Return emergency fallback style
            return self._create_emergency_style(style_name, language_code)
    
    def _get_actual_font_with_fallback(self, font_family: str, config: StyleConfig, language_code: str) -> str:
        """
        Determine the actual font name to use with comprehensive fallback mechanisms.
        Prioritizes the font suggested by the LanguageMapper, then checks font family registrations,
        and finally falls back to system defaults.
        
        Args:
            font_family: The primary font family/name to attempt to use (usually the selected_font from LanguageMapper)
            config: Style configuration, including bold/italic flags
            language_code: Language code for context and logging
            
        Returns:
            Actual font name to use for ReportLab
        """
        logger.debug(f"🔍 Resolving actual font for family/name '{font_family}' (language: {language_code})")
        is_bold = getattr(config, 'bold', False)
        is_italic = getattr(config, 'italic', False)

        # --- Strategy 1: Try the exact font_family/name provided (from LanguageMapper's selected_font) ---
        # This handles cases where LanguageMapper already provides a fully qualified font name like 'NotoSansDevanagari-Regular'
        try:
            if pdfmetrics.getFont(font_family):
                logger.info(f"✅ Using directly registered font: '{font_family}'")
                return font_family
        except Exception:
            logger.debug(f"➡️ Font '{font_family}' not directly registered or accessible. Trying family resolution.")

        # --- Strategy 2: Resolve using ReportLab's font family mechanism ---
        # This relies on pdfmetrics.registerFontFamily being correctly set up in FontRegistry
        try:
            # Try to get the font family definition
            family_info = pdfmetrics.getFontFamily(font_family)
            if family_info:
                logger.debug(f"Found ReportLab font family definition for '{font_family}'")
                target_font = None
                if is_bold and is_italic:
                    target_font = family_info.boldItalic
                elif is_bold:
                    target_font = family_info.bold
                elif is_italic:
                    target_font = family_info.italic
                else:
                    target_font = family_info.normal
                
                if target_font and pdfmetrics.getFont(target_font):
                    logger.info(f"✅ Using font family variant '{target_font}' for '{font_family}' (bold:{is_bold}, italic:{is_italic})")
                    return target_font
                else:
                    logger.warning(f"⚠️ Specific variant (bold:{is_bold}, italic:{is_italic}) not found in family '{font_family}' or not registered. Trying regular variant.")
                    if family_info.normal and pdfmetrics.getFont(family_info.normal):
                        logger.info(f"✅ Falling back to regular variant '{family_info.normal}' for family '{font_family}'")
                        return family_info.normal
        except Exception as e:
            logger.debug(f"⚠️ Error retrieving font family '{font_family}' from ReportLab: {e}. Trying generic fallback.")

        # --- Strategy 3: Attempt to resolve using FontRegistry and preferred weights (original logic, adapted) ---
        # This is a more direct check against our internal registry if ReportLab's family mechanism failed
        if self.font_registry.is_family_available(font_family):
            logger.debug(f"Internal registry shows family '{font_family}' is available. Attempting direct lookup.")
            # Determine weight based on style requirements
            if is_bold:
                for weight in ['Bold', 'SemiBold', 'Medium', 'Regular']:
                    font_name = self.font_registry.get_font_by_family_and_weight_with_fallback(font_family, weight)
                    if font_name:
                        logger.info(f"✅ Using {font_family}-{weight} (from internal registry) for bold style in {language_code}")
                        return font_name
            else: # Not bold
                font_name = self.font_registry.get_font_by_family_and_weight_with_fallback(font_family, 'Regular')
                if font_name:
                    logger.info(f"✅ Using {font_family}-Regular (from internal registry) for {language_code}")
                    return font_name
        
        # Fallback to any available font in the family if 'Regular' not found
        family_fonts = self.font_registry.get_font_family(font_family)
        if family_fonts:
            logger.warning(f"⚠️ Regular weight not found in internal registry, using fallback font {family_fonts[0]} from {font_family} family")
            return family_fonts[0]
        
        logger.warning(f"❌ Could not resolve primary font for '{font_family}' (bold:{is_bold}, italic:{is_italic}).")
            
        # --- Final Fallback: System default fonts ---
        fallback_font = self._get_fallback_font_enhanced(config)
        logger.warning(f"🔄 Falling back to system default font: '{fallback_font}'")
        return fallback_font
            
    def _get_fallback_font_enhanced(self, config: StyleConfig) -> str:
        """
        Provides a robust fallback font, ensuring it is a ReportLab registered font.
        Prioritizes 'Helvetica' (with bold if needed), then 'Times-Roman'.
        """
        is_bold = getattr(config, 'bold', False)

        # Try Helvetica variants
        if is_bold:
            if pdfmetrics.getFont('Helvetica-Bold'):
                return 'Helvetica-Bold'
            elif pdfmetrics.getFont('Helvetica'):
                logger.warning("⚠️ Helvetica-Bold not found, falling back to Helvetica-Regular for bold style.")
                return 'Helvetica'
        else:
            if pdfmetrics.getFont('Helvetica'):
                return 'Helvetica'
        
        # Try Times-Roman variants
        if is_bold:
            if pdfmetrics.getFont('Times-Bold'):
                return 'Times-Bold'
            elif pdfmetrics.getFont('Times-Roman'):
                logger.warning("⚠️ Times-Bold not found, falling back to Times-Roman for bold style.")
                return 'Times-Roman'
        else:
            if pdfmetrics.getFont('Times-Roman'):
                return 'Times-Roman'

        logger.critical("❌ No standard fallback fonts (Helvetica, Times-Roman) could be found or are registered. PDF generation may fail.")
        return 'Helvetica' # Ultimate emergency fallback, should ideally never be reached
    
    def _get_emergency_fallback_font(self, config: StyleConfig) -> str:
        """
        Get an emergency fallback font when all other methods fail.
        
        Args:
            config: Style configuration
            
        Returns:
            Emergency fallback font name
        """
        try:
            is_bold = getattr(config, 'bold', False)
            if is_bold:
                return 'Helvetica-Bold'
            else:
                return 'Helvetica'
        except Exception:
            # Ultimate fallback
            return 'Helvetica'
    
    def _create_emergency_style(self, style_name: str, language_code: str) -> ParagraphStyle:
        """
        Create an emergency minimal style when all other creation methods fail.
        
        Args:
            style_name: Name of the style
            language_code: Language code
            
        Returns:
            Minimal ParagraphStyle object
        """
        try:
            safe_name = f"emergency_{style_name}_{language_code}".replace(' ', '_').replace('-', '_')
            return ParagraphStyle(
                name=safe_name,
                fontName='Helvetica',
                fontSize=11,
                leading=16,
                spaceAfter=12,
                spaceBefore=0,
                leftIndent=0,
                rightIndent=0,
                firstLineIndent=0,
                alignment=TA_LEFT,
                textColor=colors.black
            )
        except Exception as e:
            logger.error(f"❌ Even emergency style creation failed: {e}")
            # Return the most basic style possible
            from reportlab.lib.styles import getSampleStyleSheet
            return getSampleStyleSheet()['Normal']
    
    def _get_fallback_font(self, config: StyleConfig) -> str:
        """
        Get a fallback font when the preferred font is not available.
        
        Args:
            config: Style configuration
            
        Returns:
            Fallback font name
        """
        return self._get_fallback_font_enhanced(config)
    
    def _parse_color(self, color_str: str):
        """
        Parse color string to ReportLab color object.
        
        Args:
            color_str: Color string (hex format like '#000000')
            
        Returns:
            ReportLab color object
        """
        try:
            if color_str.startswith('#'):
                # Remove # and convert hex to RGB
                hex_color = color_str[1:]
                if len(hex_color) == 6:
                    r = int(hex_color[0:2], 16) / 255.0
                    g = int(hex_color[2:4], 16) / 255.0
                    b = int(hex_color[4:6], 16) / 255.0
                    return colors.Color(r, g, b)
            
            # Fallback to black if parsing fails
            return colors.black
            
        except Exception as e:
            logger.warning(f"Failed to parse color '{color_str}': {e}, using black")
            return colors.black
    
    def create_fallback_styles(self) -> Dict[str, ParagraphStyle]:
        """
        Create fallback styles using system fonts when no language-specific fonts are available.
        
        Returns:
            Dictionary mapping style names to fallback ParagraphStyle objects
        """
        cache_key = "fallback_styles"
        if self._max_cache_size > 0 and cache_key in self._style_cache:
            return self._style_cache[cache_key]
        
        logger.info("Creating fallback styles using system fonts")
        
        styles = {}
        
        for style_name, config in self.DEFAULT_STYLE_CONFIGS.items():
            # Use system fonts for fallback
            fallback_font = self._get_fallback_font(config)
            text_color = self._parse_color(config.text_color)
            
            style = ParagraphStyle(
                name=f"fallback_{style_name}",
                fontName=fallback_font,
                fontSize=config.font_size,
                leading=config.leading,
                spaceAfter=config.space_after,
                spaceBefore=config.space_before,
                leftIndent=config.left_indent,
                rightIndent=config.right_indent,
                firstLineIndent=config.first_line_indent,
                alignment=config.alignment,
                textColor=text_color
            )
            
            styles[style_name] = style
        
        # Cache the fallback styles (if caching is enabled)
        if self._max_cache_size > 0:
            if len(self._style_cache) >= self._max_cache_size:
                self._cleanup_cache()
            self._style_cache[cache_key] = styles
        
        logger.info(f"Created {len(styles)} fallback styles")
        return styles
    
    def get_style_for_text_type(self, text_type: str, language_code: str = None) -> ParagraphStyle:
        """
        Get a specific style for a text type and language.
        
        Args:
            text_type: Type of text ('title', 'heading1', 'heading2', 'body', etc.)
            language_code: Language code (optional, uses fallback if not provided)
            
        Returns:
            ParagraphStyle object
        """
        if language_code:
            styles = self.create_styles_for_language(language_code)
        else:
            styles = self.create_fallback_styles()
        
        return styles.get(text_type, styles.get('body', self._create_emergency_fallback_style()))
    
    def _create_emergency_fallback_style(self) -> ParagraphStyle:
        """
        Create an emergency fallback style if all else fails.
        
        Returns:
            Basic ParagraphStyle object
        """
        return ParagraphStyle(
            name="emergency_fallback",
            fontName="Helvetica",
            fontSize=11,
            leading=16,
            spaceAfter=12,
            alignment=TA_LEFT,
            textColor=colors.black
        )
    
    def create_custom_style(self, name: str, language_code: str, **kwargs) -> ParagraphStyle:
        """
        Create a custom style with specific parameters.
        
        Args:
            name: Name for the custom style
            language_code: Language code for font selection
            **kwargs: Style parameters (fontSize, leading, spaceAfter, etc.)
            
        Returns:
            Custom ParagraphStyle object
        """
        # Get base font for the language
        font_family = self.language_mapper.get_font_for_language(language_code)
        
        # Create base config and override with provided parameters
        base_config = self.DEFAULT_STYLE_CONFIGS.get('body')
        
        # Build style parameters
        style_params = {
            'name': f"custom_{name}_{language_code}",
            'fontName': self._get_actual_font_with_fallback(font_family, base_config, language_code),
            'fontSize': kwargs.get('fontSize', base_config.font_size),
            'leading': kwargs.get('leading', base_config.leading),
            'spaceAfter': kwargs.get('spaceAfter', base_config.space_after),
            'spaceBefore': kwargs.get('spaceBefore', base_config.space_before),
            'leftIndent': kwargs.get('leftIndent', base_config.left_indent),
            'rightIndent': kwargs.get('rightIndent', base_config.right_indent),
            'firstLineIndent': kwargs.get('firstLineIndent', base_config.first_line_indent),
            'alignment': kwargs.get('alignment', base_config.alignment),
            'textColor': self._parse_color(kwargs.get('textColor', base_config.text_color))
        }
        
        return ParagraphStyle(**style_params)
    
    def _cleanup_cache(self):
        """Clean up cache when it reaches the configured threshold"""
        if not self._style_cache:
            return
        
        cleanup_threshold = self.config.cache_config.cache_cleanup_threshold
        target_size = int(self._max_cache_size * cleanup_threshold)
        
        if len(self._style_cache) > target_size:
            # Remove oldest entries (simple FIFO cleanup)
            items_to_remove = len(self._style_cache) - target_size
            keys_to_remove = list(self._style_cache.keys())[:items_to_remove]
            
            for key in keys_to_remove:
                self._style_cache.pop(key, None)
            
            logger.debug(f"Cache cleanup: removed {items_to_remove} entries, new size: {len(self._style_cache)}")
    
    def clear_cache(self):
        """Clear the style cache to force recreation of styles."""
        logger.info("Clearing style cache")
        self._style_cache.clear()
    
    def get_cache_info(self) -> Dict:
        """
        Get information about the style cache.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'cached_style_sets': len(self._style_cache),
            'cache_keys': list(self._style_cache.keys()),
            'total_cached_styles': sum(len(styles) for styles in self._style_cache.values())
        }


# Global style factory instance
_style_factory = None


def get_style_factory() -> EnhancedStyleFactory:
    """
    Get the global style factory instance, creating it if necessary.
    
    Returns:
        EnhancedStyleFactory instance
    """
    global _style_factory
    if _style_factory is None:
        _style_factory = EnhancedStyleFactory(font_registry, get_language_mapper())
    return _style_factory


def create_style_factory(font_registry: FontRegistry = None, 
                        language_mapper: LanguageMapper = None,
                        config: Optional[FontSystemConfig] = None) -> EnhancedStyleFactory:
    """
    Create a new EnhancedStyleFactory instance with configuration support.
    
    Args:
        font_registry: FontRegistry instance (uses global if not provided)
        language_mapper: LanguageMapper instance (uses global if not provided)
        config: Font system configuration (loads from environment if None)
        
    Returns:
        EnhancedStyleFactory instance
    """
    if font_registry is None:
        font_registry = get_font_registry()
    if language_mapper is None:
        language_mapper = get_language_mapper()
    
    return EnhancedStyleFactory(font_registry, language_mapper, config)