# GPT-5 Compatibility Fix

## Issue
Application crashed with error when using GPT-5 model:
```
Error code: 400 - {'error': {'message': "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.", 'type': 'invalid_request_error', 'param': 'max_tokens', 'code': 'unsupported_parameter'}}
```

## Root Cause
OpenAI changed the API parameter name for newer models (GPT-5, o1, o3):
- **Old models** (GPT-3.5, GPT-4): Use `max_tokens`
- **New models** (GPT-5, o1, o3): Use `max_completion_tokens`

## Solution
Updated the code to detect the model version and use the appropriate parameter.

### Changes Made

#### 1. Updated `generate_with_gpt()` function (events.py:129-153)
Added conditional logic to use the correct parameter based on model name:

```python
def generate_with_gpt(prompt: str, temperature: float = 0.0, max_tokens: int = 2060) -> str:
    client = _create_openai_client()
    model_name = get_resolved_gpt_model()

    # GPT-5 and newer models use max_completion_tokens instead of max_tokens
    if model_name.startswith('gpt-5') or model_name.startswith('o1') or model_name.startswith('o3'):
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_completion_tokens=max_tokens,  # ✅ New parameter
        )
    else:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,  # ✅ Old parameter
        )
```

#### 2. Updated `get_resolved_gpt_model()` function (events.py:91-125)
Added the same conditional logic for model testing:

```python
# Try to use the model with a simple test
if candidate.startswith('gpt-5') or candidate.startswith('o1') or candidate.startswith('o3'):
    client.chat.completions.create(
        model=candidate,
        messages=[{"role": "user", "content": "test"}],
        max_completion_tokens=1
    )
else:
    client.chat.completions.create(
        model=candidate,
        messages=[{"role": "user", "content": "test"}],
        max_tokens=1
    )
```

#### 3. Updated `_is_model_unavailable_error()` function (events.py:86-92)
Added detection for `unsupported_parameter` errors so the model fallback works correctly:

```python
def _is_model_unavailable_error(error: Exception) -> bool:
    message = str(error).lower()
    return ("not found" in message or
            "does not exist" in message or
            "not supported" in message or
            "unsupported_parameter" in message)  # ✅ Added
```

## Supported Models

### New Parameter (`max_completion_tokens`)
- GPT-5 series: `gpt-5`, `gpt-5-turbo`
- o1 series: `o1-preview`, `o1-mini`
- o3 series: `o3-mini`

### Old Parameter (`max_tokens`)
- GPT-4 series: `gpt-4`, `gpt-4o`, `gpt-4-turbo`
- GPT-3.5 series: `gpt-3.5-turbo`

## Benefits

1. **Backwards compatible** - Still works with GPT-4 and GPT-3.5
2. **Forward compatible** - Now works with GPT-5 and newer models
3. **Automatic fallback** - If GPT-5 fails, falls back to GPT-4
4. **Future-proof** - Ready for o1 and o3 models

## Testing

The fix can be verified by:
1. Setting `GPT_MODEL=gpt-5` in environment or secrets
2. Running the app and scraping events
3. Generating essays with GPT-5

## Files Modified
- `events.py` - Updated 3 functions for GPT-5 compatibility
- `docs/GPT5_COMPATIBILITY_FIX.md` - This documentation (NEW)
