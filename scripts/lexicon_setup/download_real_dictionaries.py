#!/usr/bin/env python3
"""
Download and setup real English Scrabble dictionaries.
No more placeholders or fake dictionaries!
"""

import os
import sys
import requests
import zipfile
import tempfile
from pathlib import Path
import subprocess
import hashlib
from typing import Dict, List, Optional

class RealDictionaryDownloader:
    """Downloads and sets up real English Scrabble dictionaries."""
    
    def __init__(self, cache_dir: str = "lexica_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.third_party_dir = Path("third_party")
        self.third_party_dir.mkdir(exist_ok=True)
        
        # Dictionary sources
        self.dictionaries = {
            'enable1': {
                'url': 'https://norvig.com/ngrams/enable1.txt',
                'filename': 'enable1.txt',
                'description': 'ENABLE1 - Official English word list (172,000+ words)',
                'verified_words': ['HELLO', 'WORLD', 'SCRABBLE', 'QUACKLE', 'PYTHON']
            },
            'twl06': {
                'url': 'https://www.wordgamedictionary.com/twl06/download/twl06.txt',
                'filename': 'twl06.txt', 
                'description': 'TWL06 - Tournament Word List (178,000+ words)',
                'verified_words': ['HELLO', 'WORLD', 'SCRABBLE', 'QUACKLE', 'PYTHON']
            },
            'sowpods': {
                'url': 'https://www.wordgamedictionary.com/sowpods/download/sowpods.txt',
                'filename': 'sowpods.txt',
                'description': 'SOWPODS - International English word list (267,000+ words)',
                'verified_words': ['HELLO', 'WORLD', 'SCRABBLE', 'QUACKLE', 'PYTHON']
            }
        }
    
    def download_dictionary(self, dict_name: str) -> bool:
        """Download a specific dictionary."""
        if dict_name not in self.dictionaries:
            print(f"❌ Unknown dictionary: {dict_name}")
            return False
        
        dict_info = self.dictionaries[dict_name]
        file_path = self.cache_dir / dict_info['filename']
        
        if file_path.exists():
            print(f"✅ {dict_info['filename']} already exists")
            return True
        
        print(f"📥 Downloading {dict_info['description']}...")
        
        try:
            response = requests.get(dict_info['url'], stream=True, timeout=30)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✅ Downloaded {dict_info['filename']}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to download {dict_info['filename']}: {e}")
            return False
    
    def download_all_dictionaries(self) -> bool:
        """Download all available dictionaries."""
        print("🚀 Downloading real English Scrabble dictionaries...")
        
        success = True
        for dict_name in self.dictionaries:
            if not self.download_dictionary(dict_name):
                success = False
        
        return success
    
    def setup_quackle(self) -> bool:
        """Setup Quackle for dictionary compilation."""
        quackle_path = self.third_party_dir / "quackle"
        
        if quackle_path.exists():
            print("✅ Quackle already exists")
            return True
        
        print("📥 Cloning Quackle...")
        try:
            subprocess.run([
                'git', 'clone', '--depth', '1', 
                'https://github.com/quackle/quackle.git', 
                str(quackle_path)
            ], check=True)
            
            print("🔨 Building Quackle...")
            build_path = quackle_path / "build"
            build_path.mkdir(exist_ok=True)
            
            # Configure with CMake
            subprocess.run([
                'cmake', 
                '-DCMAKE_BUILD_TYPE=Release',
                '-DCMAKE_POSITION_INDEPENDENT_CODE=ON',
                '-DCMAKE_CXX_FLAGS=-fPIC -O3',
                '..'
            ], cwd=build_path, check=True)
            
            # Build
            subprocess.run(['make', '-j'], cwd=build_path, check=True)
            
            print("✅ Quackle built successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to setup Quackle: {e}")
            return False
    
    def compile_dawg(self, dict_name: str) -> bool:
        """Compile dictionary to DAWG format."""
        if dict_name not in self.dictionaries:
            return False
        
        dict_info = self.dictionaries[dict_name]
        txt_file = self.cache_dir / dict_info['filename']
        dawg_file = self.cache_dir / f"{dict_name}.dawg"
        
        if dawg_file.exists():
            print(f"✅ {dawg_file.name} already exists")
            return True
        
        quackle_path = self.third_party_dir / "quackle"
        makedawg = quackle_path / "build" / "src" / "makedawg"
        
        if not makedawg.exists():
            print("❌ makedawg not found. Please build Quackle first.")
            return False
        
        print(f"🔨 Compiling {dict_name} to DAWG...")
        
        try:
            with open(dawg_file, 'w') as f:
                subprocess.run([
                    str(makedawg),
                    str(txt_file),
                    'english.quackle_alphabet'
                ], stdout=f, check=True)
            
            print(f"✅ Compiled {dawg_file.name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to compile DAWG: {e}")
            return False
    
    def compile_gaddag(self, dict_name: str) -> bool:
        """Compile dictionary to GADDAG format."""
        if dict_name not in self.dictionaries:
            return False
        
        dawg_file = self.cache_dir / f"{dict_name}.dawg"
        gaddag_file = self.cache_dir / f"{dict_name}.gaddag"
        
        if gaddag_file.exists():
            print(f"✅ {gaddag_file.name} already exists")
            return True
        
        quackle_path = self.third_party_dir / "quackle"
        makegaddag = quackle_path / "build" / "src" / "makegaddag"
        
        if not makegaddag.exists():
            print("❌ makegaddag not found. Please build Quackle first.")
            return False
        
        print(f"🔨 Compiling {dict_name} to GADDAG...")
        
        try:
            with open(gaddag_file, 'w') as f:
                subprocess.run([
                    str(makegaddag),
                    str(dawg_file)
                ], stdout=f, check=True)
            
            print(f"✅ Compiled {gaddag_file.name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to compile GADDAG: {e}")
            return False
    
    def verify_dictionary(self, dict_name: str) -> bool:
        """Verify dictionary is working correctly."""
        if dict_name not in self.dictionaries:
            return False
        
        dict_info = self.dictionaries[dict_name]
        txt_file = self.cache_dir / dict_info['filename']
        
        if not txt_file.exists():
            print(f"❌ Dictionary file not found: {txt_file}")
            return False
        
        print(f"🔍 Verifying {dict_name} dictionary...")
        
        try:
            with open(txt_file, 'r') as f:
                words = [line.strip().upper() for line in f if line.strip()]
            
            print(f"📊 Dictionary contains {len(words)} words")
            
            # Verify expected words are present
            for word in dict_info['verified_words']:
                if word in words:
                    print(f"✅ Found expected word: {word}")
                else:
                    print(f"❌ Missing expected word: {word}")
                    return False
            
            # Check for common invalid entries
            invalid_count = 0
            for word in words:
                if not word.isalpha() or len(word) < 2:
                    invalid_count += 1
            
            if invalid_count > 0:
                print(f"⚠️  Found {invalid_count} potentially invalid words")
            
            print(f"✅ {dict_name} dictionary verified successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to verify dictionary: {e}")
            return False
    
    def setup_complete_lexicon(self, dict_name: str = 'enable1') -> bool:
        """Complete setup of a dictionary."""
        print(f"🚀 Setting up complete lexicon: {dict_name}")
        
        # Download dictionary
        if not self.download_dictionary(dict_name):
            return False
        
        # Verify dictionary
        if not self.verify_dictionary(dict_name):
            return False
        
        # Setup Quackle
        if not self.setup_quackle():
            return False
        
        # Compile DAWG
        if not self.compile_dawg(dict_name):
            return False
        
        # Compile GADDAG
        if not self.compile_gaddag(dict_name):
            return False
        
        print(f"🎉 Complete lexicon setup finished: {dict_name}")
        return True
    
    def get_lexicon_info(self) -> Dict:
        """Get information about available lexicons."""
        info = {}
        
        for dict_name, dict_info in self.dictionaries.items():
            txt_file = self.cache_dir / dict_info['filename']
            dawg_file = self.cache_dir / f"{dict_name}.dawg"
            gaddag_file = self.cache_dir / f"{dict_name}.gaddag"
            
            info[dict_name] = {
                'description': dict_info['description'],
                'txt_exists': txt_file.exists(),
                'dawg_exists': dawg_file.exists(),
                'gaddag_exists': gaddag_file.exists(),
                'txt_size': txt_file.stat().st_size if txt_file.exists() else 0,
                'dawg_size': dawg_file.stat().st_size if dawg_file.exists() else 0,
                'gaddag_size': gaddag_file.stat().st_size if gaddag_file.exists() else 0
            }
        
        return info


def main():
    """Main function."""
    downloader = RealDictionaryDownloader()
    
    if len(sys.argv) > 1:
        dict_name = sys.argv[1]
        if dict_name == 'all':
            downloader.download_all_dictionaries()
        elif dict_name in downloader.dictionaries:
            downloader.setup_complete_lexicon(dict_name)
        else:
            print(f"❌ Unknown dictionary: {dict_name}")
            print(f"Available: {list(downloader.dictionaries.keys())}")
    else:
        # Default: setup ENABLE1
        downloader.setup_complete_lexicon('enable1')
    
    # Show info
    info = downloader.get_lexicon_info()
    print("\n📊 Lexicon Status:")
    for name, data in info.items():
        status = "✅" if data['txt_exists'] and data['dawg_exists'] and data['gaddag_exists'] else "❌"
        print(f"  {status} {name}: {data['description']}")


if __name__ == "__main__":
    main()
