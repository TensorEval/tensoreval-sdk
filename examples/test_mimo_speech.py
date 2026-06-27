"""Quick test: Mimo TTS and ASR API connectivity."""

import sys, asyncio, base64
sys.path.insert(0, ".")
from openai import AsyncOpenAI

API_KEY = "tp-sjgd0v2ckg08hbi46e8uvkfmonq4ywqvxv4dsa27qz6xk5zg"

async def test_tts():
    print("[TTS] Testing mimo-v2.5-tts...")
    client = AsyncOpenAI(api_key=API_KEY, base_url="https://token-plan-sgp.xiaomimimo.com/v1")
    try:
        completion = await client.chat.completions.create(
            model="mimo-v2.5-tts",
            messages=[
                {"role": "user", "content": "Professional and clear."},
                {"role": "assistant", "content": "What is 12 multiplied by 15?"},
            ],
            extra_body={"audio": {"format": "wav", "voice": "Chloe"}},
        )
        msg = completion.choices[0].message
        if hasattr(msg, "audio") and msg.audio:
            audio_data = msg.audio["data"] if isinstance(msg.audio, dict) else msg.audio
            audio_bytes = base64.b64decode(audio_data)
            print(f"  OK: {len(audio_bytes)} bytes of audio")
            return audio_bytes
        else:
            print(f"  No audio in response. Message attrs: {dir(msg)}")
            print(f"  Message: {msg}")
            return None
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        return None

async def test_asr(audio_bytes: bytes):
    print()
    print("[ASR] Testing mimo-v2.5-asr...")
    client = AsyncOpenAI(api_key=API_KEY, base_url="https://token-plan-sgp.xiaomimimo.com/v1")
    try:
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        completion = await client.chat.completions.create(
            model="mimo-v2.5-asr",
            messages=[{
                "role": "user",
                "content": [{
                    "type": "input_audio",
                    "input_audio": {"data": f"data:audio/wav;base64,{audio_b64}"}
                }]
            }],
            extra_body={"asr_options": {"language": "auto"}},
        )
        text = completion.choices[0].message.content
        print(f"  OK: '{text}'")
        return text
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        return None

async def test_llm():
    print()
    print("[LLM] Testing mimo-v2.5-pro...")
    import anthropic
    client = anthropic.AsyncAnthropic(
        api_key=API_KEY,
        base_url="https://token-plan-sgp.xiaomimimo.com/anthropic",
    )
    try:
        response = await client.messages.create(
            model="mimo-v2.5-pro",
            max_tokens=200,
            messages=[{"role": "user", "content": "What is 2+2? Answer with just the number."}],
        )
        for block in response.content:
            if hasattr(block, "text"):
                print(f"  OK: '{block.text}'")
                return block.text
        print(f"  No text block found")
        return None
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        return None

async def main():
    print("=" * 50)
    print("Mimo Speech Model API Test")
    print("=" * 50)

    tts = await test_tts()
    if tts:
        asr = await test_asr(tts)
    else:
        asr = None

    llm = await test_llm()

    print()
    print("=" * 50)
    print(f"TTS: {'OK' if tts else 'FAIL'}")
    print(f"ASR: {'OK' if asr else 'FAIL'}")
    print(f"LLM: {'OK' if llm else 'FAIL'}")
    print("=" * 50)

asyncio.run(main())
