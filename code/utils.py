import json
import pickle
import logging
from pathlib import Path
from datetime import datetime
from config import LOGS_DIR

# إعداد نظام السجلات (Logging)
def setup_logger(name):
    """إعداد مسجل الأحداث لحفظ سجلات العمليات"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # تجنب تكرار إضافة المعالجات
    if not logger.handlers:
        # صيغة السجل
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # حفظ في ملف
        log_file = LOGS_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        
        # عرض في الشاشة
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    return logger

# دوال حفظ وتحميل البيانات
def save_json(data, filepath):
    """حفظ البيانات بصيغة JSON"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(filepath):
    """تحميل البيانات من ملف JSON"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_pickle(data, filepath):
    """حفظ البيانات (مثل الفهارس) بصيغة Pickle"""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(filepath):
    """تحميل البيانات من ملف Pickle"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)
