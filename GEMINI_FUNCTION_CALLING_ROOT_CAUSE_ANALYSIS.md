# Gemini Function Calling - Root Cause Analysis

## Date: 2025-11-13

## Problem Statement

Function calling works perfectly with OpenAI Realtime API but fails completely with Gemini Live API. When a user provides information that should trigger a function call (e.g., ticket reference number "2278"), nothing happens - the model doesn't invoke any functions.

## Executive Summary

**Root Cause**: Gemini Live API has **two fundamentally separate input paths** for audio, and the current implementation uses the wrong path that doesn't support function calling evaluation.

**Solution**: Disable automatic VAD and manually bridge audio transcriptions from the fast path to the structured path that supports function calling.

---

## Architectural Comparison

### OpenAI Realtime API - Unified Architecture

OpenAI has a **single unified path** for processing audio:

```
User Audio
    ↓
[send audio to OpenAI]
    ↓
[OpenAI Internal Processing]
    ├─ Transcribe audio
    ├─ Evaluate for function calling
    └─ Decide: Generate response OR Call function
    ↓
Response: {text/audio} OR {function_call}
```

**Key Characteristics:**
- Single audio input method
- Function calling evaluation happens automatically
- Seamless integration of VAD, transcription, and function calling
- Developer just sends audio and handles responses

### Gemini Live API - Dual-Path Architecture

Gemini has **TWO SEPARATE PATHS** for processing audio:

#### Path A: `send_realtime_input()` - Fast VAD Path

```
User Audio
    ↓
[session.send_realtime_input(audio)]
    ↓
[Gemini VAD Processing]
    ├─ Voice Activity Detection
    ├─ Transcription (if enabled)
    └─ AUTOMATIC fast response generation
    ↓
Response: {audio} (NO function calling evaluation)
```

**Characteristics:**
- Optimized for ultra-low latency
- Automatic VAD-based responses
- **DOES NOT evaluate function calls**
- From Gemini docs: *"optimized for responsiveness at the expense of deterministic ordering"*

#### Path B: `send_client_content()` - Structured Path

```
User Input (text or audio)
    ↓
[session.send_client_content(turns=...)]
    ↓
[Gemini Structured Processing]
    ├─ Transcribe (if audio)
    ├─ EVALUATE FOR FUNCTION CALLING
    └─ Decide: Generate response OR Call function
    ↓
Response: {audio/text} OR {function_call}
```

**Characteristics:**
- Higher latency (waits for complete input)
- Deterministic ordering
- **EVALUATES function calls**
- Manual control over when responses are triggered

---

## Why OpenAI Works and Gemini Doesn't

### OpenAI Implementation

```python
# Single method - everything works
await openai_client.send_audio(audio_chunk)

# OpenAI automatically:
# 1. Buffers audio
# 2. Detects speech end via VAD
# 3. Transcribes
# 4. Evaluates for function calls
# 5. Returns response OR function call
```

**Result:** Function calling works seamlessly ✅

### Current Gemini Implementation (Broken)

```python
# Using Path A - fast VAD path
await session.send_realtime_input(
    audio=types.Blob(data=pcm16_16k, mime_type="audio/pcm;rate=16000")
)

# What happens:
# 1. Audio sent to Gemini
# 2. VAD detects speech
# 3. Transcription generated (we see this in logs)
# 4. Automatic response triggered immediately
# 5. Function calling is NEVER evaluated ❌
```

**Configuration:**
```python
realtime_input_config = types.RealtimeInputConfig(
    automatic_activity_detection=types.AutomaticActivityDetection(
        disabled=False,  # ← AUTOMATIC VAD ENABLED
        # ...
    )
)
```

**Result:** Function calling never happens ❌

---

## Evidence from Logs

### Successful Conversation Flow (No Function Call Needed)

```
0:43 [Gemini] Input: 'I want to change the date of my ticket.'
0:45 [Gemini] Output: 'I can help with that. Can you please provide the ticket reference number?'
```
✅ Model responds correctly with a question

### Failed Function Call (When Function SHOULD Be Called)

```
User says: "2278"
[Gemini transcribes: ' 2278']
... SILENCE - nothing happens ...
```
❌ Model should call `genesys_data_action_custom_37be4e2a...` to retrieve ticket details
❌ Instead, nothing happens at all

### Why This Happens

1. Audio "2278" is sent via `send_realtime_input()`
2. Gemini transcribes it: `' 2278'`
3. Automatic VAD generates a response (or tries to)
4. **Function calling is NEVER evaluated** because we're using Path A
5. The transcription is logged but never sent through Path B where function calls are evaluated

---

## Official Documentation Evidence

### From Gemini Docs on `send_realtime_input()`

> "With `send_realtime_input`, the API will respond to audio automatically based on VAD... **`send_realtime_input` is optimized for responsiveness at the expense of deterministic ordering**."

### From Gemini Docs on Function Calling

**EVERY single function calling example uses `send_client_content()`, not `send_realtime_input()`:**

```python
# From official Gemini function calling example
async with client.aio.live.connect(model=model, config=config) as session:
    prompt = "Turn on the lights please"
    await session.send_client_content(turns={"parts": [{"text": prompt}]})  # ← Uses send_client_content

    async for response in session.receive():
        if response.tool_call:
            # Handle function call
```

### From Gemini Docs on Disabling Automatic VAD

> "Alternatively, the automatic VAD can be disabled by setting `realtimeInputConfig.automaticActivityDetection.disabled` to `true` in the setup message. **In this configuration the client is responsible for detecting user speech** and sending `activityStart` and `activityEnd` messages at the appropriate times."

---

## Why the Previous Fix Attempt Failed

### What Was Tried

The previous fix attempted to:
1. Use `send_realtime_input()` to get transcriptions quickly
2. Accumulate transcriptions as they arrive
3. When turn complete, send transcription via `send_client_content()`
4. Hope that function calling would be triggered

### Why It Failed

**Race Condition**: By the time we accumulate the transcription and try to send it via `send_client_content()`, the automatic VAD response has already been triggered and completed!

```
Timeline:
t=0: Audio "2278" sent via send_realtime_input()
t=1: Gemini starts transcribing
t=2: Automatic VAD detects end of speech
t=3: Automatic response generation starts (bypasses function calling)
t=4: Response completes
t=5: ❌ TOO LATE! We try to send transcription via send_client_content()
```

The two paths are **asynchronous and independent** - you can't override or cancel the automatic response that was already triggered.

---

## The Real Solution

### Disable Automatic VAD

```python
realtime_input_config = types.RealtimeInputConfig(
    automatic_activity_detection=types.AutomaticActivityDetection(
        disabled=True,  # ← DISABLE automatic responses
        # ...
    )
)
```

### Implement Manual Flow Control

```
1. When audio starts → send activity_start (once)
2. Stream audio via send_realtime_input()
3. Accumulate input transcriptions as they arrive
4. Detect when user stops speaking (timing heuristic)
5. Send activity_end
6. Send accumulated transcription via send_client_content() with turn_complete=True
7. Model evaluates for function calling
8. Receive response (audio OR function_call)
```

### Benefits

- ✅ Still use `send_realtime_input()` for low-latency audio streaming
- ✅ Still get input transcriptions for free
- ✅ But control WHEN responses are generated
- ✅ Bridge transcriptions to `send_client_content()` for function calling evaluation
- ✅ Function calling works!

### Trade-offs

- Slightly higher latency (need to wait for speech end detection)
- Need to implement client-side speech end detection (timing-based)
- More complex state management

But these are acceptable trade-offs to make function calling work!

---

## Implementation Details

### Changes Required

1. **Config Change**: Set `automatic_activity_detection.disabled = True`

2. **Add State Tracking**:
   - Track whether activity has started
   - Accumulate input transcriptions
   - Detect speech end via timing

3. **Add Activity Management**:
   - Send `activity_start` when first audio arrives
   - Send `activity_end` when speech stops
   - Clear transcriptions on interruptions

4. **Add Transcription Bridging**:
   - Accumulate transcriptions as they arrive
   - When speech ends, send via `send_client_content()`
   - Let Gemini evaluate for function calling

### Speech End Detection

Since we disable automatic VAD, we need to detect when user stops speaking:

**Option A: Time-based** (simpler)
- If no new transcription updates for 1 second → speech ended
- Send `activity_end` and bridge transcription

**Option B: Audio analysis** (more complex)
- Track audio amplitude
- If silence detected for 1 second → speech ended
- Send `activity_end` and bridge transcription

**Option C: Hybrid** (recommended)
- Track both transcription updates AND audio
- Speech ends when BOTH are silent for 1 second
- Most robust approach

---

## Expected Behavior After Fix

### Before (Broken)
```
User: "2278"
[Gemini transcribes: ' 2278']
... SILENCE ...
```

### After (Fixed)
```
User: "2278"
[Gemini transcribes: ' 2278']
[Detect speech end after 1s]
[Send activity_end]
[Send transcription via send_client_content()]
[Function call triggered: genesys_data_action_custom_37be4e2a...]
[Execute function]
[Agent responds: "I found your ticket for Berlin to Munich..."]
```

---

## Conclusion

The root cause is **architectural**: Gemini Live API requires using `send_client_content()` for function calling, but our implementation uses `send_realtime_input()` with automatic VAD, which bypasses function calling evaluation.

The solution is to **disable automatic VAD** and manually bridge audio transcriptions from the fast path to the structured path that supports function calling.

This is not a bug in our code or in Gemini - it's a fundamental design difference between OpenAI's unified architecture and Gemini's dual-path architecture. We need to adapt our implementation to work with Gemini's design.

---

## References

- [Gemini Live API Documentation](https://ai.google.dev/gemini-api/docs/live)
- [Gemini Function Calling](https://ai.google.dev/gemini-api/docs/live-tools)
- [Gemini VAD Configuration](https://ai.google.dev/gemini-api/docs/live-guide#voice-activity-detection-vad)
- [OpenAI Realtime API Documentation](https://platform.openai.com/docs/guides/realtime)
