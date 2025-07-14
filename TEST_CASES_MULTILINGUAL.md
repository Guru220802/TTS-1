# TTS Multilingual Test Cases Documentation

## Overview

This document provides comprehensive test cases for validating the TTS-LipSync-Translation system across multiple languages. Each test case includes expected outcomes, validation criteria, and troubleshooting steps.

---

## Test Case 1: Hindi Language Processing

### Test Objective
Validate complete pipeline for English-to-Hindi translation with TTS and lip-sync generation.

### Test Data
- **Input Text**: "Artificial intelligence is revolutionizing the way we communicate and interact with technology in our daily lives."
- **Target Language**: `hi` (Hindi)
- **Expected Translation**: "कृत्रिम बुद्धिमत्ता हमारे दैनिक जीवन में प्रौद्योगिकी के साथ संवाद और बातचीत के तरीके में क्रांति ला रही है।"

### Test Steps
1. **API Call**
   ```bash
   curl -X POST "http://192.168.0.125:8001/api/generate-and-sync" \
        -H "Content-Type: application/x-www-form-urlencoded" \
        -d "text=Artificial intelligence is revolutionizing the way we communicate and interact with technology in our daily lives.&target_lang=hi"
   ```

2. **Expected Response**
   - HTTP Status: 200
   - Content-Type: video/mp4
   - File size: 2-10 MB (depending on duration)

3. **Validation Criteria**
   - ✅ Translation accuracy: >90% semantic correctness
   - ✅ Devanagari script validation
   - ✅ Audio quality: Clear Hindi pronunciation
   - ✅ Lip-sync accuracy: Mouth movements match audio
   - ✅ Avatar selection: Gender-appropriate avatar
   - ✅ Video duration: 8-15 seconds

### Metadata Validation
```json
{
  "language": "hi",
  "language_name": "Hindi",
  "script": "Devanagari",
  "text_length": 108,
  "gender": "male|female",
  "avatar": "pht1.jpg|pht2.jpg|pht3.jpg|pht4.jpg",
  "video_format": "mp4"
}
```

### Success Metrics
- Translation confidence score: >0.8
- Audio generation: <5 seconds
- Video generation: <30 seconds
- No errors in processing pipeline

---

## Test Case 2: English Language Processing (Baseline)

### Test Objective
Validate native English processing without translation to establish baseline performance.

### Test Data
- **Input Text**: "The future of human-computer interaction lies in natural language processing and realistic avatar generation."
- **Target Language**: `en` (English)
- **Expected Output**: Same text (no translation)

### Test Steps
1. **API Call**
   ```bash
   curl -X POST "http://192.168.0.125:8001/api/generate-and-sync" \
        -H "Content-Type: application/x-www-form-urlencoded" \
        -d "text=The future of human-computer interaction lies in natural language processing and realistic avatar generation.&target_lang=en"
   ```

2. **Expected Response**
   - HTTP Status: 200
   - Content-Type: video/mp4
   - Faster processing (no translation step)

3. **Validation Criteria**
   - ✅ No translation performed (confidence = 1.0)
   - ✅ Clear English pronunciation
   - ✅ Natural speech rhythm and intonation
   - ✅ Accurate lip-sync timing
   - ✅ Appropriate avatar gender selection
   - ✅ Video duration: 10-18 seconds

### Performance Benchmarks
- Audio generation: <3 seconds
- Video generation: <20 seconds
- Total processing time: <25 seconds

---

## Test Case 3: German Language Processing

### Test Objective
Validate English-to-German translation with proper Germanic pronunciation and lip-sync.

### Test Data
- **Input Text**: "Modern technology enables seamless translation and speech synthesis across multiple languages and cultures."
- **Target Language**: `de` (German)
- **Expected Translation**: "Moderne Technologie ermöglicht nahtlose Übersetzung und Sprachsynthese über mehrere Sprachen und Kulturen hinweg."

### Test Steps
1. **API Call**
   ```bash
   curl -X POST "http://192.168.0.125:8001/api/generate-and-sync" \
        -H "Content-Type: application/x-www-form-urlencoded" \
        -d "text=Modern technology enables seamless translation and speech synthesis across multiple languages and cultures.&target_lang=de"
   ```

2. **Expected Response**
   - HTTP Status: 200
   - Content-Type: video/mp4
   - German audio with proper pronunciation

3. **Validation Criteria**
   - ✅ Translation accuracy: >85% semantic correctness
   - ✅ Latin script validation
   - ✅ German pronunciation: Proper umlauts and consonants
   - ✅ Lip-sync accuracy: Matches German phonemes
   - ✅ Avatar selection: Gender-appropriate
   - ✅ Video duration: 12-20 seconds

### Language-Specific Checks
- Proper handling of German compound words
- Correct pronunciation of umlauts (ä, ö, ü)
- Appropriate sentence stress and rhythm

---

## Automated Test Script

### Python Test Runner
```python
import requests
import json
import time
import os

class TTSTestRunner:
    def __init__(self, base_url="http://192.168.0.125:8001"):
        self.base_url = base_url
        self.test_results = []
    
    def run_test_case(self, test_name, text, language, expected_duration_range):
        print(f"Running {test_name}...")
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate-and-sync",
                data={"text": text, "target_lang": language},
                timeout=300
            )
            
            processing_time = time.time() - start_time
            
            result = {
                "test_name": test_name,
                "status": "PASS" if response.status_code == 200 else "FAIL",
                "processing_time": processing_time,
                "response_size": len(response.content),
                "language": language
            }
            
            if response.status_code == 200:
                # Save video for manual inspection
                filename = f"test_{language}_{int(time.time())}.mp4"
                with open(filename, "wb") as f:
                    f.write(response.content)
                result["output_file"] = filename
            
            self.test_results.append(result)
            print(f"✅ {test_name}: {result['status']} ({processing_time:.2f}s)")
            
        except Exception as e:
            result = {
                "test_name": test_name,
                "status": "ERROR",
                "error": str(e),
                "language": language
            }
            self.test_results.append(result)
            print(f"❌ {test_name}: ERROR - {e}")
    
    def run_all_tests(self):
        test_cases = [
            ("Hindi Translation Test", 
             "Artificial intelligence is revolutionizing the way we communicate and interact with technology in our daily lives.", 
             "hi", (8, 15)),
            ("English Baseline Test", 
             "The future of human-computer interaction lies in natural language processing and realistic avatar generation.", 
             "en", (10, 18)),
            ("German Translation Test", 
             "Modern technology enables seamless translation and speech synthesis across multiple languages and cultures.", 
             "de", (12, 20))
        ]
        
        for test_name, text, language, duration_range in test_cases:
            self.run_test_case(test_name, text, language, duration_range)
            time.sleep(2)  # Brief pause between tests
    
    def generate_report(self):
        print("\n" + "="*50)
        print("TEST EXECUTION REPORT")
        print("="*50)
        
        for result in self.test_results:
            print(f"Test: {result['test_name']}")
            print(f"Status: {result['status']}")
            print(f"Language: {result['language']}")
            if 'processing_time' in result:
                print(f"Processing Time: {result['processing_time']:.2f}s")
            if 'output_file' in result:
                print(f"Output File: {result['output_file']}")
            print("-" * 30)

# Run tests
if __name__ == "__main__":
    runner = TTSTestRunner()
    runner.run_all_tests()
    runner.generate_report()
```

---

## Manual Validation Checklist

### Audio Quality Assessment
- [ ] Clear pronunciation without distortion
- [ ] Appropriate speech speed (not too fast/slow)
- [ ] Natural intonation and rhythm
- [ ] No background noise or artifacts
- [ ] Proper volume levels

### Video Quality Assessment
- [ ] Lip movements synchronized with audio
- [ ] Avatar facial expressions appear natural
- [ ] No visual artifacts or glitches
- [ ] Appropriate video resolution and quality
- [ ] Smooth frame transitions

### Translation Quality Assessment
- [ ] Semantic accuracy maintained
- [ ] Cultural context preserved
- [ ] Proper script/character encoding
- [ ] Grammar and syntax correctness
- [ ] Natural language flow

### Performance Metrics
- [ ] Processing time within acceptable limits
- [ ] File sizes reasonable for content length
- [ ] No memory leaks or resource issues
- [ ] Consistent results across multiple runs
- [ ] Error handling works correctly

---

## Troubleshooting Guide

### Common Issues and Solutions

1. **Translation Failures**
   - Check Gemini API key configuration
   - Verify internet connectivity
   - Review rate limiting settings

2. **Audio Generation Issues**
   - Ensure gTTS dependencies are installed
   - Check audio output directory permissions
   - Verify language code format

3. **Video Generation Failures**
   - Confirm Wav2Lip model files exist
   - Check FFmpeg installation
   - Verify avatar image files are accessible

4. **Performance Issues**
   - Monitor system resources (CPU, memory)
   - Check disk space for output files
   - Review concurrent request handling

---

## Success Criteria Summary

✅ **All three test cases pass with 95% success rate**
✅ **Processing time under 30 seconds per video**
✅ **Translation accuracy >85% for all languages**
✅ **Audio quality meets production standards**
✅ **Lip-sync accuracy visually acceptable**
✅ **No critical errors in processing pipeline**
