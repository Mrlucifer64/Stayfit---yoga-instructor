"""
Configuration file for AI Yoga Instructor MVP
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists
load_dotenv()

class Config:
    """Application configuration class"""
    
    # Flask configuration
    SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'dev-key-change-this-in-prod')
    DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    # Database configuration
    DB_CONFIG = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'database': os.getenv('DB_NAME', 'yoga_progress'),
        'user': os.getenv('DB_USER', 'root'),
        'passwd': os.getenv('DB_PASSWORD', ''), 
        'port': int(os.getenv('DB_PORT', '3306'))
    }
    
    # Camera configuration
    CAMERA_INDEX = int(os.getenv('CAMERA_INDEX', '0'))
    CAMERA_WIDTH = int(os.getenv('CAMERA_WIDTH', '640'))
    CAMERA_HEIGHT = int(os.getenv('CAMERA_HEIGHT', '480'))
    CAMERA_FPS = int(os.getenv('CAMERA_FPS', '30'))
    
    # Pose detection configuration
    POSE_DETECTION_CONFIDENCE = float(os.getenv('POSE_DETECTION_CONFIDENCE', '0.5'))
    POSE_TRACKING_CONFIDENCE = float(os.getenv('POSE_TRACKING_CONFIDENCE', '0.5'))
    POSE_MODEL_COMPLEXITY = int(os.getenv('POSE_MODEL_COMPLEXITY', '1'))  # 0=Lite, 1=Full, 2=Heavy
    
    # Voice feedback configuration
    VOICE_ENABLED = os.getenv('VOICE_ENABLED', 'True').lower() == 'true'
    VOICE_RATE = int(os.getenv('VOICE_RATE', '150'))  # Words per minute
    VOICE_VOLUME = float(os.getenv('VOICE_VOLUME', '0.8'))  # 0.0 to 1.0
    
    # Session configuration
    DEFAULT_POSE_DURATION = int(os.getenv('DEFAULT_POSE_DURATION', '30'))  # seconds
    SESSION_TIMEOUT = int(os.getenv('SESSION_TIMEOUT', '3600'))  # seconds
    
    # UI configuration
    REFRESH_RATE = int(os.getenv('REFRESH_RATE', '1000'))  # milliseconds
    MAX_POSE_HISTORY = int(os.getenv('MAX_POSE_HISTORY', '100'))
    
    # Development configuration
    USE_FALLBACK_STORAGE = os.getenv('USE_FALLBACK_STORAGE', 'False').lower() == 'true'
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    @classmethod
    def get_db_url(cls) -> str:
        """Get database connection URL"""
        config = cls.DB_CONFIG
        password_part = f":{config['passwd']}" if config['passwd'] else ""
        return f"mysql+pymysql://{config['user']}{password_part}@{config['host']}:{config['port']}/{config['database']}"

class DevelopmentConfig(Config):
    """Development-specific configuration"""
    DEBUG = True
    USE_FALLBACK_STORAGE = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production-specific configuration"""
    DEBUG = False
    USE_FALLBACK_STORAGE = False
    LOG_LEVEL = 'WARNING'

class TestingConfig(Config):
    """Testing-specific configuration"""
    TESTING = True
    USE_FALLBACK_STORAGE = True
    DEBUG = True

# Configuration mapping
config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config(config_name: str = None) -> Config:
    """Get configuration based on environment"""
    if config_name is None:
        config_name = os.getenv('FLASK_ENV', 'default')
    
    config_class = config_map.get(config_name, config_map['default'])
    
    # PRODUCTION SAFETY CHECK
    # Only check for the weak secret key if we are actually IN production mode
    if config_name == 'production' and config_class.SECRET_KEY == 'dev-key-change-this-in-prod':
        raise ValueError("CRITICAL: You are running in Production with a default SECRET_KEY. Set FLASK_SECRET_KEY in .env")

    return config_class()