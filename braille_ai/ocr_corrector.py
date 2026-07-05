"""
ocr_corrector.py
----------------
Groq-powered OCR correction for Braille text output.
Uses llama-3.1-8b-instant to fix OCR errors and reconstruct readable English.
"""

import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

_client = None
_client_failed = False


def _valid_api_key() -> Optional[str]:
    key = (os.getenv("GROQ_API_KEY") or "").strip()
    if not key:
        return None
    if key.startswith("your_") or "your_groq" in key.lower():
        return None
    return key


def _get_client():
    global _client, _client_failed
    if _client_failed:
        return None
    if _client is not None:
        return _client
    api_key = _valid_api_key()
    if not api_key:
        return None
    try:
        from groq import Groq
        _client = Groq(api_key=api_key)
        return _client
    except Exception as e:
        _client_failed = True
        print(f"⚠️  Groq client unavailable: {e}")
        return None


def correct_with_groq(raw_text: str) -> str:
    """
    Use Groq llama-3.1-8b-instant to correct OCR errors in braille-decoded text.

    Args:
        raw_text: Raw text from the Braille OCR pipeline (may have errors)

    Returns:
        Corrected English text, or the original if correction fails/unavailable.
    """
    if not raw_text or not raw_text.strip():
        return raw_text

    client = _get_client()
    if client is None:
        return raw_text

    prompt = (
        "You are correcting text decoded from hand-drawn Braille dots using computer vision.\n"
        "The dots were drawn by hand on paper (pen/pencil), so the OCR is imperfect.\n\n"
        "Common errors to fix:\n"
        "- '?' means the dot pattern was unrecognised — replace with the most likely letter based on context\n"
        "- Missing or extra letters due to dot detection errors\n"
        "- Wrong letters from misaligned dot rows (e.g. 'b' instead of 'a', 'k' instead of 'a')\n"
        "- Spacing errors between words\n\n"
        "Rules:\n"
        "1. Use English dictionary words — correct spelling based on context\n"
        "2. If the text looks like a word (e.g. 'hel?o' → 'hello'), fix it\n"
        "3. Preserve the original meaning and word count as much as possible\n"
        "4. Return ONLY the corrected text — no explanations, no quotes\n"
        "5. If the text is already correct English, return it unchanged\n\n"
        f"Raw OCR text: {raw_text}\n"
        "Corrected text:"
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=256,
        )
        corrected = response.choices[0].message.content.strip()
        if len(corrected) > len(raw_text) * 3:
            return raw_text
        return corrected if corrected else raw_text
    except Exception as e:
        print(f"⚠️  Groq correction failed: {e}")
        return raw_text


def is_groq_available() -> bool:
    """Return True if Groq API is configured and the client can be created."""
    return _get_client() is not None


if __name__ == "__main__":
    test_cases = [
        "HE1LO W0RLD",
        "the qu1ck br0wn f0x",
        "brail1e is a tact1le wr1ting system",
    ]
    print("Testing Groq OCR corrector...\n")
    for raw in test_cases:
        corrected = correct_with_groq(raw)
        print(f"  Raw:       {raw}")
        print(f"  Corrected: {corrected}\n")
