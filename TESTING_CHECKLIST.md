# Testing Checklist for PDF Download Fixes

## Pre-Testing Setup

- [ ] Backend server is running
- [ ] Frontend is connected to backend
- [ ] Font system is initialized
- [ ] Test PDF files are ready

## Test 1: Basic PDF Download

**Steps:**

1. Upload a simple PDF with 1-2 paragraphs
2. Select target language (e.g., Hindi)
3. Click "Extract & Translate"
4. Wait for translation to complete
5. Click "Download PDF"

**Expected Results:**

- [ ] PDF downloads successfully (no "Failed to fetch" error)
- [ ] PDF opens without errors
- [ ] Text is in paragraphs, not line-by-line
- [ ] Font renders correctly (no black boxes)

## Test 2: Multi-Paragraph Document

**Steps:**

1. Upload PDF with 5+ paragraphs
2. Translate to Hindi/Urdu/Sinhala
3. Download PDF

**Expected Results:**

- [ ] All paragraphs are separated
- [ ] Paragraph spacing is consistent
- [ ] Text wraps within margins
- [ ] No text overflow

## Test 3: Long Paragraphs

**Steps:**

1. Upload PDF with very long paragraphs (200+ words)
2. Translate and download

**Expected Results:**

- [ ] Text wraps correctly across multiple lines
- [ ] No text cut off at page edges
- [ ] Line spacing is consistent
- [ ] Page breaks work correctly

## Test 4: Mixed Content

**Steps:**

1. Upload PDF with titles, headings, and body text
2. Translate and download

**Expected Results:**

- [ ] Different text sizes preserved
- [ ] Paragraph structure maintained
- [ ] Spacing between sections appropriate

## Test 5: Font Rendering

**Steps:**

1. Test with Hindi (Devanagari script)
2. Test with Urdu (Arabic script)
3. Test with Sinhala script

**Expected Results:**

- [ ] All characters render correctly
- [ ] No black boxes or missing glyphs
- [ ] Font is consistent throughout document
- [ ] Special characters (vowel marks, etc.) display properly

## Test 6: Edge Cases

**Steps:**

1. Upload very small PDF (1 sentence)
2. Upload very large PDF (10+ pages)
3. Upload PDF with special characters

**Expected Results:**

- [ ] Small PDFs work correctly
- [ ] Large PDFs process without timeout
- [ ] Special characters preserved

## Test 7: Error Handling

**Steps:**

1. Try downloading before translation completes
2. Try with corrupted PDF
3. Try with unsupported language

**Expected Results:**

- [ ] Appropriate error messages shown
- [ ] No crashes or freezes
- [ ] User can retry

## Test 8: Performance

**Steps:**

1. Measure translation time for 2-page PDF
2. Measure PDF generation time
3. Check memory usage

**Expected Results:**

- [ ] Translation completes in reasonable time (<30s)
- [ ] PDF generates quickly (<5s)
- [ ] No memory leaks
- [ ] GPU memory cleared properly

## Test 9: API Response Validation

**Steps:**

1. Check API response structure
2. Verify layout_data serialization
3. Check translated_text_chunks format

**Expected Results:**

- [ ] Response contains all required fields
- [ ] layout_data is properly serialized JSON
- [ ] translated_text_chunks is array of strings
- [ ] No serialization errors in logs

## Test 10: Comparison with Live Preview

**Steps:**

1. Translate same PDF in both modes
2. Compare output quality
3. Compare formatting

**Expected Results:**

- [ ] Translation quality is similar
- [ ] Extract & Translate has better formatting
- [ ] Both modes work without errors

## Backend Logs to Check

- [ ] No "AttributeError: 'PdfLayoutData' object has no attribute 'dict'" errors
- [ ] "Using paragraph-based translation" message appears
- [ ] "Grouped X elements into Y paragraphs" message appears
- [ ] "Successfully created PDF with paragraph-based rendering" message appears
- [ ] No font registration errors

## Frontend Console to Check

- [ ] No fetch errors
- [ ] Download triggers correctly
- [ ] File saves with correct name
- [ ] No JavaScript errors

## Known Issues to Watch For

1. **Black boxes:** If characters show as black boxes, check font initialization
2. **Line-by-line:** If still line-by-line, check paragraph grouping thresholds
3. **Download fails:** Check serialization in backend logs
4. **Slow performance:** Check batch sizes and memory clearing

## Success Criteria

All tests must pass with:

- ✅ No errors in backend logs
- ✅ No errors in frontend console
- ✅ PDF downloads successfully
- ✅ Text is in paragraph format
- ✅ Fonts render correctly
- ✅ Performance is acceptable

## Regression Testing

After fixes, verify these still work:

- [ ] Live Preview mode
- [ ] Simple translation mode
- [ ] Word document download
- [ ] Batch translation
- [ ] Font status endpoints

## If Tests Fail

### Black Boxes Issue

1. Check font system initialization: `GET /font-health`
2. Verify font files exist in `Pyback1/fonts/`
3. Check font registration logs
4. Test specific language: `GET /font-test/hin_Deva`

### Line-by-Line Issue

1. Check paragraph grouping logs
2. Adjust `y_threshold` and `paragraph_threshold`
3. Verify layout_data is being used
4. Check if falling back to simple rendering

### Download Fails

1. Check backend logs for serialization errors
2. Verify `asdict()` is being used
3. Check deserialization in download endpoint
4. Test with smaller PDF first

### Performance Issues

1. Reduce batch_size
2. Reduce max_pages
3. Check GPU memory usage
4. Clear cache between tests
