import re

def clean_text(text, language):
    if language == 'KR':
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\uAC00-\uD7A3a-zA-Z0-9.,!?~“”‘’\"\'\s]", "", text)
        text = re.sub(r"[▶▲◆■]", "", text)
        return text.strip()
    else:
        raise ValueError(f"This version is for Korean (KR) only. Input language='{language}' detected.")
