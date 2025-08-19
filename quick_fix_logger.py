#!/usr/bin/env python3
"""
Quick fix for logger issue in HF spaces
"""
import os

def fix_logger_issue():
    """Fix the logger undefined error in all HF spaces"""
    
    print("🔧 Fixing logger issue in HF spaces...")
    
    # Corrected cache setup code with proper logging
    cache_fix_code = '''
import os
import tempfile
import logging

# Set up logging first
logging.basicConfig(level=logging.INFO)
cache_logger = logging.getLogger(__name__)

# Set up cache directories with fallback
def setup_cache_dirs():
    """Setup cache directories with proper permissions"""
    cache_dirs = [
        os.environ.get('TRANSFORMERS_CACHE', '/tmp/transformers_cache'),
        os.environ.get('HF_HOME', '/tmp/huggingface_cache'),
        os.environ.get('TORCH_HOME', '/tmp/torch_cache')
    ]
    
    for cache_dir in cache_dirs:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            # Test write permissions
            test_file = os.path.join(cache_dir, 'test_write')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            cache_logger.info(f"✅ Cache directory ready: {cache_dir}")
        except Exception as e:
            cache_logger.warning(f"⚠️ Cache directory issue {cache_dir}: {e}")
            # Fallback to temp directory
            fallback_dir = tempfile.mkdtemp()
            if 'TRANSFORMERS_CACHE' in cache_dir:
                os.environ['TRANSFORMERS_CACHE'] = fallback_dir
            elif 'HF_HOME' in cache_dir:
                os.environ['HF_HOME'] = fallback_dir
            elif 'TORCH_HOME' in cache_dir:
                os.environ['TORCH_HOME'] = fallback_dir
            cache_logger.info(f"📁 Using fallback cache: {fallback_dir}")

# Call setup at startup
setup_cache_dirs()
'''

    # Update all HF spaces
    hf_spaces = [
        "hf_space_deploy",
        "hf_space_fix", 
        "hf_space_memory_fix",
        "hf_space_requirements_fix"
    ]
    
    for space in hf_spaces:
        main_py_path = f"{space}/main.py"
        
        try:
            # Read existing main.py
            with open(main_py_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Remove old cache setup if present
            if 'setup_cache_dirs' in content:
                lines = content.split('\n')
                new_lines = []
                skip_cache_block = False
                
                for line in lines:
                    if 'def setup_cache_dirs' in line or 'setup_cache_dirs()' in line:
                        skip_cache_block = True
                        continue
                    elif skip_cache_block and (line.startswith('def ') or line.startswith('class ') or line.startswith('app = ')):
                        skip_cache_block = False
                        new_lines.append(line)
                    elif not skip_cache_block:
                        new_lines.append(line)
                
                content = '\n'.join(new_lines)
            
            # Find position after imports to insert cache setup
            lines = content.split('\n')
            insert_pos = 0
            
            # Find the end of imports
            for i, line in enumerate(lines):
                if line.startswith('from ') or line.startswith('import '):
                    insert_pos = i + 1
                elif line.strip() == '' and insert_pos > 0:
                    continue
                elif insert_pos > 0 and not (line.startswith('from ') or line.startswith('import ') or line.strip() == ''):
                    break
            
            # Insert cache setup code
            lines.insert(insert_pos, cache_fix_code)
            
            # Write back
            with open(main_py_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            
            print(f"✅ Fixed logger issue in {space}/main.py")
            
        except Exception as e:
            print(f"❌ Failed to fix {space}: {e}")

if __name__ == "__main__":
    fix_logger_issue()
    print("\n🎉 Logger issue fixed!")
    print("🚀 Ready to redeploy spaces")