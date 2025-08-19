#!/usr/bin/env python3
"""
Test script for memory-optimized PDF processing
"""
import requests
import json

def test_memory_endpoints():
    """Test the new memory management endpoints"""
    
    base_url = "http://localhost:8000"  # Change to your deployed URL
    
    print("🧪 Testing Memory Management Features...")
    print("=" * 50)
    
    # Test memory info endpoint
    try:
        response = requests.get(f"{base_url}/memory-info")
        if response.status_code == 200:
            memory_info = response.json()
            print("📊 Memory Info:")
            print(json.dumps(memory_info, indent=2))
        else:
            print(f"❌ Memory info failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Memory info error: {e}")
    
    print("\n" + "=" * 50)
    
    # Test memory clearing
    try:
        response = requests.post(f"{base_url}/clear-memory")
        if response.status_code == 200:
            clear_result = response.json()
            print("🧹 Memory Clear Result:")
            print(json.dumps(clear_result, indent=2))
        else:
            print(f"❌ Memory clear failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Memory clear error: {e}")
    
    print("\n" + "=" * 50)
    print("🚀 Memory Management Features Added:")
    print("✅ Text chunking for large PDFs")
    print("✅ Batch processing to manage memory")
    print("✅ GPU memory clearing between batches")
    print("✅ Reduced beam search (3 instead of 5)")
    print("✅ Float16 precision for GPU (saves ~50% memory)")
    print("✅ Memory monitoring endpoints")
    print("✅ OCR image resizing for memory efficiency")
    print("✅ Low CPU memory usage during model loading")

def test_simple_translation():
    """Test simple translation with memory management"""
    
    base_url = "http://localhost:8000"
    
    test_data = {
        "text": "This is a test sentence for translation.",
        "target_language": "hin_Deva",
        "source_language": "eng_Latn"
    }
    
    try:
        response = requests.post(f"{base_url}/translate-simple", json=test_data)
        if response.status_code == 200:
            result = response.json()
            print("✅ Simple Translation Test:")
            print(f"Original: {result.get('original_text', 'N/A')}")
            print(f"Translated: {result.get('translated_text', 'N/A')}")
        else:
            print(f"❌ Translation failed: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ Translation error: {e}")

if __name__ == "__main__":
    test_memory_endpoints()
    print("\n" + "=" * 50)
    test_simple_translation()