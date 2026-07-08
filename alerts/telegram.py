"""
Telegram Alert System.

Handles:
  - Sending formatted messages to Telegram
  - Message splitting for 4096 char limit
  - Error handling with retry
"""

import sys
from typing import List

import requests

from config.settings import Settings


def send_telegram(msg: str, parse_mode: str = "Markdown") -> bool:
    """
    Send a message to Telegram. Auto-splits if too long.
    Returns True if sent successfully.
    """
    cfg = Settings.get()
    if not cfg.telegram_token or not cfg.telegram_chat_id:
        print("[TG] Skipped — no credentials.")
        return False

    max_len = cfg.telegram_message_max_len
    chunks = _split_message(msg, max_len)
    success = True

    for chunk in chunks:
        ok = _send_single(chunk, cfg.telegram_token,
                          cfg.telegram_chat_id, parse_mode)
        if not ok:
            success = False

    return success


def _send_single(msg: str, token: str, chat_id: str,
                 parse_mode: str) -> bool:
    """Send a single Telegram message with retry."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": msg,
        "parse_mode": parse_mode,
    }

    for attempt in range(3):
        try:
            r = requests.post(url, json=payload, timeout=10)
            if r.status_code == 200:
                print("[TG] ✅ Sent")
                return True
            elif r.status_code == 429:
                # Rate limited
                import time
                time.sleep(2 * (attempt + 1))
                continue
            else:
                print(f"[TG] ❌ Status {r.status_code}: {r.text[:200]}",
                      file=sys.stderr)
                # If markdown parsing fails, retry without markdown
                if attempt == 0 and parse_mode == "Markdown":
                    r2 = requests.post(url, json={
                        "chat_id": chat_id,
                        "text": msg,
                    }, timeout=10)
                    if r2.status_code == 200:
                        print("[TG] ✅ Sent (plain text fallback)")
                        return True
                return False
        except Exception as e:
            print(f"[TG] Error: {e}", file=sys.stderr)
            if attempt < 2:
                import time
                time.sleep(1)

    return False


def _split_message(msg: str, max_len: int) -> List[str]:
    """Split a long message into chunks at line boundaries."""
    if len(msg) <= max_len:
        return [msg]

    chunks = []
    current = ""

    for line in msg.split("\n"):
        if len(current) + len(line) + 1 > max_len:
            if current:
                chunks.append(current)
            current = line
        else:
            current = current + "\n" + line if current else line

    if current:
        chunks.append(current)

    return chunks if chunks else [msg[:max_len]]
