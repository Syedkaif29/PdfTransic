"""
Font Configuration Module

This module provides configuration management for the font system,
including environment variable support, font preferences, and caching settings.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FontCacheConfig:
    """Configuration for font caching and memory management"""
    max_cache_size: int = 100
    enable_font_caching: bool = True
    cache_cleanup_threshold: float = 0.8  # Cleanup when 80% full
    memory_limit_mb: int = 50  # Maximum memory for font cache in MB
    
    def __post_init__(self):
        """Validate cache configuration"""
        if self.max_cache_size < 1:
            logger.warning("Font cache size must be at least 1, setting to 1")
            self.max_cache_size = 1
        
        if self.cache_cleanup_threshold < 0.1 or self.cache_cleanup_threshold > 1.0:
            logger.warning("Cache cleanup threshold must be between 0.1 and 1.0, setting to 0.8")
            self.cache_cleanup_threshold = 0.8
        
        if self.memory_limit_mb < 1:
            logger.warning("Memory limit must be at least 1MB, setting to 10MB")
            self.memory_limit_mb = 10


@dataclass
class FontPreferences:
    """Font preferences and fallback configuration"""
    preferred_weights: List[str] = field(default_factory=lambda: ['Regular', 'Medium', 'Light'])
    fallback_fonts: Dict[str, List[str]] = field(default_factory=dict)
    enable_fallback: bool = True
    strict_weight_matching: bool = False
    
    def __post_init__(self):
        """Set up default fallback fonts if not provided"""
        if not self.fallback_fonts:
            self.fallback_fonts = {
                'NotoSansDevanagari': ['DejaVu Sans', 'Arial Unicode MS', 'Times-Roman'],
                'NotoNastaliqUrdu': ['DejaVu Sans', 'Arial Unicode MS', 'Times-Roman'],
                'NotoSansSinhala': ['DejaVu Sans', 'Arial Unicode MS', 'Times-Roman'],
                'default': ['Helvetica', 'Times-Roman', 'Courier']
            }


@dataclass
class FontSystemConfig:
    """Main font system configuration"""
    # Directory settings
    font_base_path: str = 'fonts'
    additional_font_paths: List[str] = field(default_factory=list)
    
    # File handling
    supported_extensions: Set[str] = field(default_factory=lambda: {'.ttf', '.otf'})
    max_file_size_mb: int = 50
    min_file_size_bytes: int = 1024
    
    # Registration settings
    max_registration_retries: int = 3
    registration_timeout_seconds: float = 30.0
    enable_async_registration: bool = True
    
    # Logging and monitoring
    enable_font_logging: bool = True
    log_level: str = 'DEBUG'
    enable_performance_monitoring: bool = False
    
    # Error handling
    continue_on_errors: bool = True
    max_errors_before_abort: int = 100
    
    # Font preferences and caching
    preferences: FontPreferences = field(default_factory=FontPreferences)
    cache_config: FontCacheConfig = field(default_factory=FontCacheConfig)
    
    def __post_init__(self):
        """Validate and normalize configuration"""
        # Normalize paths
        self.font_base_path = os.path.normpath(self.font_base_path)
        self.additional_font_paths = [os.path.normpath(p) for p in self.additional_font_paths]
        
        # Validate file size limits
        if self.max_file_size_mb < 1:
            logger.warning("Max file size must be at least 1MB, setting to 10MB")
            self.max_file_size_mb = 10
        
        if self.min_file_size_bytes < 100:
            logger.warning("Min file size must be at least 100 bytes, setting to 1024 bytes")
            self.min_file_size_bytes = 1024
        
        # Validate retry settings
        if self.max_registration_retries < 1:
            logger.warning("Max retries must be at least 1, setting to 1")
            self.max_registration_retries = 1
        
        if self.registration_timeout_seconds < 1.0:
            logger.warning("Registration timeout must be at least 1 second, setting to 5 seconds")
            self.registration_timeout_seconds = 5.0
        
        # Validate error handling
        if self.max_errors_before_abort < 1:
            logger.warning("Max errors before abort must be at least 1, setting to 10")
            self.max_errors_before_abort = 10
        
        # Validate log level
        valid_log_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        if self.log_level.upper() not in valid_log_levels:
            logger.warning(f"Invalid log level '{self.log_level}', setting to 'DEBUG'")
            self.log_level = 'DEBUG'


class FontConfigManager:
    """
    Font configuration manager that loads settings from environment variables
    and provides configuration access throughout the font system.
    """
    
    def __init__(self):
        """Initialize configuration manager"""
        self._config: Optional[FontSystemConfig] = None
        self._env_prefix = 'FONT_'
        
    def load_config(self) -> FontSystemConfig:
        """
        Load configuration from environment variables and defaults.
        
        Returns:
            FontSystemConfig object with loaded settings
        """
        if self._config is not None:
            return self._config
        
        logger.info("🔧 Loading font system configuration from environment variables...")
        
        # Load basic settings
        font_base_path = os.getenv('FONT_BASE_PATH', 'fonts')
        additional_paths = self._parse_path_list(os.getenv('FONT_ADDITIONAL_PATHS', ''))
        
        # Load file handling settings
        supported_extensions = self._parse_extension_list(
            os.getenv('FONT_SUPPORTED_EXTENSIONS', '.ttf,.otf')
        )
        max_file_size_mb = self._parse_int(os.getenv('FONT_MAX_FILE_SIZE_MB', '50'), 50)
        min_file_size_bytes = self._parse_int(os.getenv('FONT_MIN_FILE_SIZE_BYTES', '1024'), 1024)
        
        # Load registration settings
        max_retries = self._parse_int(os.getenv('FONT_MAX_RETRIES', '3'), 3)
        timeout_seconds = self._parse_float(os.getenv('FONT_REGISTRATION_TIMEOUT', '30.0'), 30.0)
        enable_async = self._parse_bool(os.getenv('FONT_ENABLE_ASYNC_REGISTRATION', 'true'))
        
        # Load logging settings
        enable_logging = self._parse_bool(os.getenv('FONT_ENABLE_LOGGING', 'true'))
        log_level = os.getenv('FONT_LOG_LEVEL', 'DEBUG').upper()
        enable_monitoring = self._parse_bool(os.getenv('FONT_ENABLE_MONITORING', 'false'))
        
        # Load error handling settings
        continue_on_errors = self._parse_bool(os.getenv('FONT_CONTINUE_ON_ERRORS', 'true'))
        max_errors = self._parse_int(os.getenv('FONT_MAX_ERRORS_BEFORE_ABORT', '100'), 100)
        
        # Load cache configuration
        cache_config = self._load_cache_config()
        
        # Load font preferences
        preferences = self._load_font_preferences()
        
        # Create configuration object
        self._config = FontSystemConfig(
            font_base_path=font_base_path,
            additional_font_paths=additional_paths,
            supported_extensions=supported_extensions,
            max_file_size_mb=max_file_size_mb,
            min_file_size_bytes=min_file_size_bytes,
            max_registration_retries=max_retries,
            registration_timeout_seconds=timeout_seconds,
            enable_async_registration=enable_async,
            enable_font_logging=enable_logging,
            log_level=log_level,
            enable_performance_monitoring=enable_monitoring,
            continue_on_errors=continue_on_errors,
            max_errors_before_abort=max_errors,
            preferences=preferences,
            cache_config=cache_config
        )
        
        # Log configuration summary
        self._log_config_summary()
        
        return self._config
    
    def _load_cache_config(self) -> FontCacheConfig:
        """Load font cache configuration from environment variables"""
        max_cache_size = self._parse_int(os.getenv('FONT_CACHE_SIZE', '100'), 100)
        enable_caching = self._parse_bool(os.getenv('FONT_ENABLE_CACHING', 'true'))
        cleanup_threshold = self._parse_float(os.getenv('FONT_CACHE_CLEANUP_THRESHOLD', '0.8'), 0.8)
        memory_limit = self._parse_int(os.getenv('FONT_CACHE_MEMORY_LIMIT_MB', '50'), 50)
        
        return FontCacheConfig(
            max_cache_size=max_cache_size,
            enable_font_caching=enable_caching,
            cache_cleanup_threshold=cleanup_threshold,
            memory_limit_mb=memory_limit
        )
    
    def _load_font_preferences(self) -> FontPreferences:
        """Load font preferences from environment variables"""
        # Parse preferred weights
        preferred_weights_str = os.getenv('FONT_PREFERRED_WEIGHTS', 'Regular,Medium,Light')
        preferred_weights = [w.strip() for w in preferred_weights_str.split(',') if w.strip()]
        
        # Parse fallback settings
        enable_fallback = self._parse_bool(os.getenv('FONT_ENABLE_FALLBACK', 'true'))
        strict_matching = self._parse_bool(os.getenv('FONT_STRICT_WEIGHT_MATCHING', 'false'))
        
        # Parse custom fallback fonts (format: family1:font1,font2;family2:font3,font4)
        fallback_fonts = {}
        fallback_str = os.getenv('FONT_FALLBACK_FONTS', '')
        if fallback_str:
            try:
                for family_mapping in fallback_str.split(';'):
                    if ':' in family_mapping:
                        family, fonts = family_mapping.split(':', 1)
                        font_list = [f.strip() for f in fonts.split(',') if f.strip()]
                        if font_list:
                            fallback_fonts[family.strip()] = font_list
            except Exception as e:
                logger.warning(f"Failed to parse FONT_FALLBACK_FONTS: {e}")
        
        preferences = FontPreferences(
            preferred_weights=preferred_weights,
            enable_fallback=enable_fallback,
            strict_weight_matching=strict_matching
        )
        
        # Override default fallbacks if custom ones provided
        if fallback_fonts:
            preferences.fallback_fonts.update(fallback_fonts)
        
        return preferences
    
    def _parse_path_list(self, path_str: str) -> List[str]:
        """Parse comma-separated list of paths"""
        if not path_str:
            return []
        return [p.strip() for p in path_str.split(',') if p.strip()]
    
    def _parse_extension_list(self, ext_str: str) -> Set[str]:
        """Parse comma-separated list of file extensions"""
        if not ext_str:
            return {'.ttf', '.otf'}
        
        extensions = set()
        for ext in ext_str.split(','):
            ext = ext.strip().lower()
            if ext and not ext.startswith('.'):
                ext = '.' + ext
            if ext:
                extensions.add(ext)
        
        return extensions if extensions else {'.ttf', '.otf'}
    
    def _parse_int(self, value: str, default: int) -> int:
        """Parse integer value with fallback to default"""
        try:
            return int(value)
        except (ValueError, TypeError):
            logger.warning(f"Invalid integer value '{value}', using default {default}")
            return default
    
    def _parse_float(self, value: str, default: float) -> float:
        """Parse float value with fallback to default"""
        try:
            return float(value)
        except (ValueError, TypeError):
            logger.warning(f"Invalid float value '{value}', using default {default}")
            return default
    
    def _parse_bool(self, value: str) -> bool:
        """Parse boolean value from string"""
        if not value:
            return False
        return value.lower() in ('true', '1', 'yes', 'on', 'enabled')
    
    def _log_config_summary(self):
        """Log configuration summary for debugging"""
        if not self._config or not self._config.enable_font_logging:
            return
        
        logger.info("📋 Font system configuration summary:")
        logger.info(f"  Font base path: {self._config.font_base_path}")
        
        if self._config.additional_font_paths:
            logger.info(f"  Additional paths: {len(self._config.additional_font_paths)}")
            for path in self._config.additional_font_paths:
                logger.info(f"    - {path}")
        
        logger.info(f"  Supported extensions: {', '.join(self._config.supported_extensions)}")
        logger.info(f"  File size limits: {self._config.min_file_size_bytes} - {self._config.max_file_size_mb}MB")
        logger.info(f"  Registration retries: {self._config.max_registration_retries}")
        logger.info(f"  Async registration: {self._config.enable_async_registration}")
        logger.info(f"  Font caching: {self._config.cache_config.enable_font_caching}")
        
        if self._config.cache_config.enable_font_caching:
            logger.info(f"  Cache size: {self._config.cache_config.max_cache_size}")
            logger.info(f"  Memory limit: {self._config.cache_config.memory_limit_mb}MB")
        
        logger.info(f"  Fallback enabled: {self._config.preferences.enable_fallback}")
        logger.info(f"  Preferred weights: {', '.join(self._config.preferences.preferred_weights)}")
    
    def get_config(self) -> FontSystemConfig:
        """Get current configuration, loading if necessary"""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def reload_config(self) -> FontSystemConfig:
        """Reload configuration from environment variables"""
        self._config = None
        return self.load_config()
    
    def update_config(self, **kwargs) -> FontSystemConfig:
        """
        Update specific configuration values.
        
        Args:
            **kwargs: Configuration values to update
            
        Returns:
            Updated configuration object
        """
        if self._config is None:
            self.load_config()
        
        # Update configuration attributes
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
                logger.info(f"Updated font config: {key} = {value}")
            else:
                logger.warning(f"Unknown configuration key: {key}")
        
        return self._config


# Global configuration manager instance
_config_manager = FontConfigManager()


def get_font_config() -> FontSystemConfig:
    """
    Get the global font system configuration.
    
    Returns:
        FontSystemConfig object with current settings
    """
    return _config_manager.get_config()


def reload_font_config() -> FontSystemConfig:
    """
    Reload font configuration from environment variables.
    
    Returns:
        Reloaded FontSystemConfig object
    """
    return _config_manager.reload_config()


def update_font_config(**kwargs) -> FontSystemConfig:
    """
    Update specific font configuration values.
    
    Args:
        **kwargs: Configuration values to update
        
    Returns:
        Updated FontSystemConfig object
    """
    return _config_manager.update_config(**kwargs)