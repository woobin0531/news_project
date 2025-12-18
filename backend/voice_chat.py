# voice_chat.py
# - OpenAI TTS/STT 헬퍼
# - TTS: tts-1 (voice=alloy)로 mp3 생성하여 경로 반환
# - STT: gpt-4o-mini-transcribe 우선, 실패 시 whisper-1 폴백
# - BytesIO/file 양쪽 지원, 빈/짧은 오디오 가드

import os
import io
import tempfile
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

# =========================
# 환경 변수 및 클라이언트
# =========================
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY is not defined in environment")

client = OpenAI(api_key=API_KEY)

# =========================
# TTS
# =========================
def generate_audio_file(text: str, *, model: str = "tts-1", voice: str = "alloy", suffix: str = ".mp3") -> Optional[str]:
    """
    입력 텍스트를 음성으로 합성하여 임시 파일로 저장하고, 파일 경로를 반환합니다.
    - 반환: 생성된 파일의 절대 경로(str). 실패 시 None
    - main.py의 ensure_bytes_from_generate_audio가 경로/바이트 모두 처리하므로 경로 반환이 안전합니다.
    """
    if not text or not text.strip():
        return None

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    path = Path(tmp.name)
    tmp.close()

    
    resp = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text,
        format="mp3",  
    )

    resp.stream_to_file(path)
    return str(path)


# =========================
# STT
# =========================
def _transcribe_filelike(fh, *, prefer_model: str = "gpt-4o-mini-transcribe", fallback_model: str = "whisper-1") -> str:
    """
    파일 핸들을 받아 STT 수행. prefer -> 실패 시 fallback.
    fh: file-like opened in binary mode (RB). 반드시 .name 속성이 있으면 포맷 추정에 도움됨.
    """
    try:
        result = client.audio.transcriptions.create(
            model=prefer_model,
            file=fh,
        )
        return getattr(result, "text", "") or ""
    except Exception:
        fh.seek(0)
        result = client.audio.transcriptions.create(
            model=fallback_model,
            file=fh,
        )
        return getattr(result, "text", "") or ""


def transcribe_audio_bytes(audio_bytes: bytes, *, filename: str = "rec.webm",
                           prefer_model: str = "gpt-4o-mini-transcribe",
                           fallback_model: str = "whisper-1",
                           min_size: int = 2048) -> str:
    """
    바이트 배열을 받아 STT 수행.
    - filename: BytesIO.name으로 설정하여 포맷 추정 도움
    - min_size: 최소 바이트 가드(빈/매우 짧은 오디오 방지)
    """
    if not audio_bytes or len(audio_bytes) < min_size:
        return ""

    bio = io.BytesIO(audio_bytes)
    bio.name = filename

    return _transcribe_filelike(
        bio,
        prefer_model=prefer_model,
        fallback_model=fallback_model,
    )


def transcribe_file(path: str,
                    *,
                    prefer_model: str = "gpt-4o-mini-transcribe",
                    fallback_model: str = "whisper-1",
                    min_size: int = 2048) -> str:
    """
    파일 경로를 받아 STT 수행. 주로 내부 테스트/유틸용.
    """
    p = Path(path)
    if not p.exists() or not p.is_file():
        return ""
    if p.stat().st_size < min_size:
        return ""

    with open(p, "rb") as fh:
        return _transcribe_filelike(
            fh,
            prefer_model=prefer_model,
            fallback_model=fallback_model,
        )
