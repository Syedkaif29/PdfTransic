# PDF Download Fix for Extract & Translate Mode

## Problem
The "Download PDF" button in Extract & Translate mode was failing with the error: "PDF download failed: Failed to fetch"

## Root Cause
The backend was using `.dict()` method on dataclass objects, which is a Pydantic-specific method. The `PdfLayoutData` and `PdfTextElement` classes are Python dataclasses, not Pydantic models, so they don't have a `.dict()` method.

When the backend tried to serialize these objects to JSON for the API response, it failed because:
1. Dataclasses don't have a `.dict()` method
2. The layout_data was being returned without proper serialization
3. The download endpoint wasn't properly deserializing the layout_data back to objects

## Solution
Changed from using `.dict()` (Pydantic method) to `asdict()` (dataclasses method) for serialization.

### Files Modified

#### 1. `Pyback1/main.py`
- Added `asdict` import from dataclasses
- Changed `[p.dict() for p in layout_data]` to `[asdict(p) for p in layout_data]` (2 occurrences)

#### 2. `Pyback1/pdf_translation_api.py`
- Added `asdict` import from dataclasses
- Added `PdfLayoutData` import from translation_core
- Changed `layout_data: layout_data` to `layout_data: [asdict(p) for p in layout_data] if layout_data else []`
- Added proper deserialization in `/download-translated-pdf` endpoint to convert JSON dicts back to PdfLayoutData objects

#### 3. `Pyback1/translation_core.py`
- Added `asdict` import from dataclasses (for consistency)

## Changes in Detail

### Serialization (Backend → Frontend)
**Before:**
```python
"layout_data": [p.dict() for p in layout_data]  # ❌ AttributeError: 'PdfLayoutData' object has no attribute 'dict'
```

**After:**
```python
"layout_data": [asdict(p) for p in layout_data] if layout_data else []  # ✅ Works correctly
```

### Deserialization (Frontend → Backend)
**Before:**
```python
layout_data = json.loads(layout_data_json)  # Returns list of dicts
pdf_buffer = create_pdf_from_text(..., layout_data=layout_data)  # ❌ Expects PdfLayoutData objects
```

**After:**
```python
layout_data_dicts = json.loads(layout_data_json)
layout_data = [PdfLayoutData(**item) for item in layout_data_dicts]  # ✅ Converts dicts to objects
pdf_buffer = create_pdf_from_text(..., layout_data=layout_data)
```

## Testing
Created `test_dataclass_serialization.py` to verify:
1. ✅ asdict() properly serializes dataclasses to dicts
2. ✅ JSON serialization works correctly
3. ✅ Deserialization reconstructs objects properly
4. ✅ List serialization/deserialization works
5. ✅ Confirmed .dict() doesn't exist on dataclasses

## Impact
- ✅ PDF download now works in Extract & Translate mode
- ✅ Layout data is properly preserved through the serialization cycle
- ✅ No breaking changes to the API contract
- ✅ Frontend code requires no changes

## Related Endpoints
The fix applies to these endpoints:
- `/translate-pdf-enhanced` - Returns layout_data in response
- `/download-translated-pdf` - Receives and processes layout_data
- `/translate-pdf` - Returns layout_data in response (in main.py)

## Notes
- The frontend was already correctly passing the data
- The issue was entirely on the backend serialization/deserialization
- This is a common mistake when mixing Pydantic models and Python dataclasses
