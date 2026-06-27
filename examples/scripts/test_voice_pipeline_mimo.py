"""
Voice Evaluation Scenario: Full Pipeline Test
==============================================
Tests the complete voice AI pipeline using Mimo speech models:
  1. TTS (mimo-v2.5-tts): Text → Audio
  2. ASR (mimo-v2.5-asr): Audio → Text (verifies transcription quality)
  3. LLM (mimo-v2.5-pro): Text → Agent Response
  4. Score: Agent response vs reference answer

This tests a real voice agent flow: customer speaks → ASR → agent responds.

Usage:
    python voice_eval_mimo.py
"""

import sys
import asyncio
import base64
import json
import os
import tempfile
from pathlib import Path

sys.path.insert(0, ".")

from openai import OpenAI, AsyncOpenAI

# ── Config ────────────────────────────────────────────────────
API_KEY = "tp-sjgd0v2ckg08hbi46e8uvkfmonq4ywqvxv4dsa27qz6xk5zg"
BASE_URL = "https://token-plan-sgp.xiaomimimo.com/v1"  # OpenAI-compatible endpoint
TTS_MODEL = "mimo-v2.5-tts"
ASR_MODEL = "mimo-v2.5-asr"
LLM_MODEL = "mimo-v2.5-pro"
VOICE = "Chloe"

# ── Test Dataset ──────────────────────────────────────────────
VOICE_SCENARIOS = [
    {
        "id": "v001",
        "query": "What is 12 multiplied by 15?",
        "reference_answer": "180",
        "category": "math",
        "style": "Professional and clear.",
    },
    {
        "id": "v002",
        "query": "A customer ordered 3 items at $25 each. What is the total?",
        "reference_answer": "75",
        "category": "billing",
        "style": "Friendly customer service tone.",
    },
    {
        "id": "v003",
        "query": "What is 15 percent of 200 dollars?",
        "reference_answer": "30",
        "category": "math",
        "style": "Calm and helpful.",
    },
    {
        "id": "v004",
        "query": "A store has 24 apples. They sell 8 in the morning and 6 in the afternoon. How many are left?",
        "reference_answer": "10",
        "category": "word_problem",
        "style": "Patient and clear.",
    },
    {
        "id": "v005",
        "query": "If a train travels at 60 miles per hour for 2 and a half hours, how far does it go?",
        "reference_answer": "150",
        "category": "word_problem",
        "style": "Confident and professional.",
    },
]


# ── TTS: Text → Audio ────────────────────────────────────────
async def text_to_speech(text: str, style: str = "Professional and clear.") -> bytes:
    """Convert text to speech using Mimo TTS."""
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

    completion = await client.chat.completions.create(
        model=TTS_MODEL,
        messages=[
            {"role": "user", "content": style},
            {"role": "assistant", "content": text},
        ],
        extra_body={"audio": {"format": "wav", "voice": VOICE}},
    )

    message = completion.choices[0].message
    if hasattr(message, "audio") and message.audio:
        # message.audio is ChatCompletionAudio with .data attribute
        if hasattr(message.audio, "data"):
            return base64.b64decode(message.audio.data)
        # Fallback: dict format
        elif isinstance(message.audio, dict):
            return base64.b64decode(message.audio["data"])
    raise ValueError("No audio in TTS response")


# ── ASR: Audio → Text ────────────────────────────────────────
async def speech_to_text(audio_bytes: bytes) -> str:
    """Transcribe audio using Mimo ASR."""
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    completion = await client.chat.completions.create(
        model=ASR_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": f"data:audio/wav;base64,{audio_b64}"},
                    }
                ],
            }
        ],
        extra_body={"asr_options": {"language": "auto"}},
    )

    return completion.choices[0].message.content or ""


# ── LLM: Agent Response ──────────────────────────────────────
async def get_agent_response(query: str, system_prompt: str) -> str:
    """Get agent response using Mimo LLM."""
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

    # Mimo uses Anthropic-style API at this URL
    # Switch to anthropic client for the LLM
    import anthropic
    aclient = anthropic.AsyncAnthropic(
        api_key=API_KEY,
        base_url="https://token-plan-sgp.xiaomimimo.com/anthropic",
    )

    response = await aclient.messages.create(
        model=LLM_MODEL,
        max_tokens=500,
        system=system_prompt,
        messages=[{"role": "user", "content": query}],
    )

    # Handle ThinkingBlock
    for block in response.content:
        if hasattr(block, "text"):
            return block.text
    return ""


# ── Scoring ───────────────────────────────────────────────────
def score_response(response: str, reference: str) -> dict:
    """Score agent response against reference answer."""
    response_clean = response.strip().lower()
    reference_clean = reference.strip().lower()

    # Check exact match
    exact_match = reference_clean in response_clean

    # Check numeric match
    import re
    numbers = re.findall(r"-?\d+\.?\d*", response)
    numeric_match = any(abs(float(n) - float(reference)) < 0.01 for n in numbers if n)

    passed = exact_match or numeric_match

    return {
        "passed": passed,
        "exact_match": exact_match,
        "numeric_match": numeric_match,
        "score": 1.0 if passed else 0.0,
    }


# ── Main Pipeline ─────────────────────────────────────────────
async def run_voice_eval():
    print("=" * 70)
    print("TensorEval Voice Evaluation — Full Pipeline")
    print("TTS (mimo-v2.5-tts) -> ASR (mimo-v2.5-asr) -> LLM (mimo-v2.5-pro)")
    print("=" * 70)
    print()

    results = []
    tmpdir = tempfile.mkdtemp(prefix="tensoreval-voice-")

    for i, scenario in enumerate(VOICE_SCENARIOS):
        query_id = scenario["id"]
        query = scenario["query"]
        reference = scenario["reference_answer"]
        style = scenario.get("style", "Professional and clear.")

        print(f"[{i+1}/{len(VOICE_SCENARIOS)}] {query_id}: {query[:50]}...")

        # Step 1: TTS — generate audio from query text
        try:
            audio_bytes = await text_to_speech(query, style)
            audio_path = os.path.join(tmpdir, f"{query_id}.wav")
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)
            print(f"  TTS: {len(audio_bytes)} bytes saved to {audio_path}")
        except Exception as e:
            print(f"  TTS FAILED: {e}")
            results.append({"id": query_id, "query": query, "error": f"TTS: {e}"})
            continue

        # Step 2: ASR — transcribe audio back to text
        try:
            transcribed = await speech_to_text(audio_bytes)
            print(f"  ASR: '{transcribed[:60]}'")
        except Exception as e:
            print(f"  ASR FAILED: {e}")
            results.append({"id": query_id, "query": query, "error": f"ASR: {e}"})
            continue

        # Step 3: LLM — get agent response
        try:
            agent_response = await get_agent_response(
                transcribed,
                "You are a helpful assistant. Answer the question concisely with just the number or short answer."
            )
            print(f"  LLM: '{agent_response[:60]}'")
        except Exception as e:
            print(f"  LLM FAILED: {e}")
            results.append({"id": query_id, "query": query, "error": f"LLM: {e}"})
            continue

        # Step 4: Score
        score = score_response(agent_response, reference)
        status = "PASS" if score["passed"] else "FAIL"
        print(f"  Score: {score['score']:.1f} [{status}] (exact={score['exact_match']}, numeric={score['numeric_match']})")
        print()

        results.append({
            "id": query_id,
            "query": query,
            "reference": reference,
            "transcribed": transcribed,
            "agent_response": agent_response,
            "score": score["score"],
            "passed": score["passed"],
            "audio_path": audio_path,
        })

    # ── Summary ───────────────────────────────────────────────
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    total = len(results)
    passed = sum(1 for r in results if r.get("passed"))
    failed = total - passed
    avg_score = sum(r.get("score", 0) for r in results) / total if total else 0

    print(f"  Total:      {total}")
    print(f"  Passed:     {passed}")
    print(f"  Failed:     {failed}")
    print(f"  Avg Score:  {avg_score:.2f}")
    print(f"  Pass Rate:  {passed/total:.0%}" if total else "")
    print()

    for r in results:
        status = "PASS" if r.get("passed") else "FAIL"
        print(f"  [{status}] {r['id']}: {r['query'][:40]}...")
        if r.get("transcribed"):
            print(f"        ASR: {r['transcribed'][:40]}")
        if r.get("agent_response"):
            print(f"        LLM: {r['agent_response'][:40]}")
        if r.get("reference"):
            print(f"        Ref: {r['reference']}")

    # Save results
    results_path = os.path.join(tmpdir, "voice_eval_results.json")
    with open(results_path, "w") as f:
        json.dump({"summary": {"total": total, "passed": passed, "avg_score": avg_score}, "results": results}, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return results


if __name__ == "__main__":
    asyncio.run(run_voice_eval())
