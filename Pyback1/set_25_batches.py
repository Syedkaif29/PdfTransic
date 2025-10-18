#!/usr/bin/env python3
"""
Set batch size to create exactly 25 batches
493 words ÷ 25 batches = ~20 words per batch
"""

import os
import re

def set_batch_size_for_25_batches():
    """Set batch size to ~20 to get 25 total batches"""
    
    words = 493
    target_batches = 25
    batch_size = words // target_batches  # 493 ÷ 25 = 19.72, so use 20
    
    print(f"🎯 SETTING BATCH SIZE FOR 25 BATCHES")
    print("=" * 40)
    print(f"Words: {words}")
    print(f"Target batches: {target_batches}")
    print(f"Calculated batch size: {batch_size}")
    print()
    
    # Update main.py (in same directory)
    main_py_path = "main.py"
    if os.path.exists(main_py_path):
        with open(main_py_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        content = re.sub(r'optimized_batch_size = \d+', f'optimized_batch_size = {batch_size}', content)
        # Try multiple patterns to catch different comment styles
        content = re.sub(r'batch_size = \d+  # Aggressive optimization for speed', f'batch_size = {batch_size}  # Set for 25 batches', content)
        content = re.sub(r'batch_size = \d+  # For 25 batches', f'batch_size = {batch_size}  # Set for 25 batches', content)
        content = re.sub(r'batch_size = \d+  # Optimized batch size', f'batch_size = {batch_size}  # Set for 25 batches', content)
        
        with open(main_py_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Updated main.py: batch_size = {batch_size}")
    
    # Update pdf_translation_api.py (in same directory)
    api_py_path = "pdf_translation_api.py"
    if os.path.exists(api_py_path):
        with open(api_py_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        content = re.sub(r'batch_size = \d+', f'batch_size = {batch_size}', content)
        
        with open(api_py_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Updated pdf_translation_api.py: batch_size = {batch_size}")
    
    actual_batches = (words + batch_size - 1) // batch_size
    print(f"\n🎯 Batch size set to {batch_size}")
    print(f"Expected batches: {actual_batches}")
    print(f"Words per batch: ~{words/actual_batches:.1f}")

if __name__ == "__main__":
    set_batch_size_for_25_batches()
    print(f"\n🚀 Restart server to get ~25 batches!")
