#!/usr/bin/env python3
"""
Test script to verify Hugging Face Spaces deployment
"""
import requests
import json
import time

# Your HF Space URL
BASE_URL = "https://syedkaif29-pdftransic.hf.space"

def test_endpoint(url, method="GET", data=None, timeout=30):
    """Test an API endpoint"""
    try:
        if method == "GET":
            response = requests.get(url, timeout=timeout)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=timeout)
        
        return response.status_code, response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
    except requests.exceptions.RequestException as e:
        return None, str(e)

def main():
    print(f"🧪 Testing deployment at: {BASE_URL}")
    print("=" * 50)
    
    # Test 1: Root endpoint
    print("1. Testing root endpoint...")
    status, response = test_endpoint(f"{BASE_URL}/")
    if status == 200:
        print("   ✅ Root endpoint working")
        print(f"   📝 Response: {response}")
    else:
        print(f"   ❌ Root endpoint failed: {status} - {response}")
    
    print()
    
    # Test 2: Health check
    print("2. Testing health endpoint...")
    status, response = test_endpoint(f"{BASE_URL}/health")
    if status == 200:
        print("   ✅ Health endpoint working")
        if isinstance(response, dict):
            print(f"   📊 Status: {response.get('status', 'unknown')}")
            print(f"   🖥️  Device: {response.get('device', 'unknown')}")
            components = response.get('components_loaded', {})
            for component, loaded in components.items():
                status_icon = "✅" if loaded else "⏳"
                print(f"   {status_icon} {component}: {'loaded' if loaded else 'loading'}")
        else:
            print(f"   📝 Response: {response}")
    else:
        print(f"   ❌ Health endpoint failed: {status} - {response}")
    
    print()
    
    # Test 3: Languages endpoint
    print("3. Testing languages endpoint...")
    status, response = test_endpoint(f"{BASE_URL}/languages")
    if status == 200:
        print("   ✅ Languages endpoint working")
        if isinstance(response, dict) and 'supported_languages' in response:
            lang_count = len(response['supported_languages'])
            print(f"   🌍 Supported languages: {lang_count}")
        else:
            print(f"   📝 Response: {response}")
    else:
        print(f"   ❌ Languages endpoint failed: {status} - {response}")
    
    print()
    
    # Test 4: Simple translation (only if models are loaded)
    print("4. Testing simple translation...")
    
    # First check if models are ready
    health_status, health_response = test_endpoint(f"{BASE_URL}/health")
    models_ready = False
    
    if health_status == 200 and isinstance(health_response, dict):
        components = health_response.get('components_loaded', {})
        models_ready = all(components.values())
    
    if models_ready:
        translation_data = {
            "text": "Hello, how are you?",
            "target_language": "hin_Deva",
            "source_language": "eng_Latn"
        }
        
        status, response = test_endpoint(f"{BASE_URL}/translate-simple", "POST", translation_data, timeout=60)
        if status == 200:
            print("   ✅ Translation working!")
            if isinstance(response, dict):
                print(f"   📝 Original: {response.get('original_text', 'N/A')}")
                print(f"   🔄 Translated: {response.get('translated_text', 'N/A')}")
            else:
                print(f"   📝 Response: {response}")
        else:
            print(f"   ❌ Translation failed: {status} - {response}")
    else:
        print("   ⏳ Models still loading, skipping translation test")
        print("   💡 Try again in a few minutes when models are ready")
    
    print()
    print("🎯 Testing complete!")
    print(f"📖 API Documentation: {BASE_URL}/docs")

if __name__ == "__main__":
    main()