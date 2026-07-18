"""
WhatsApp Cloud API channel helpers.

The webhook can receive text and image messages. Sending replies and downloading
media are enabled only when WhatsApp Cloud API environment variables are set.
"""
from __future__ import annotations

import json
import os
from typing import Any


GRAPH_VERSION = os.getenv("WHATSAPP_GRAPH_VERSION", "v20.0")


def parse_whatsapp_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for entry in payload.get("entry", []) or []:
        for change in entry.get("changes", []) or []:
            value = change.get("value", {}) or {}
            phone_number_id = value.get("metadata", {}).get("phone_number_id")
            for message in value.get("messages", []) or []:
                msg_type = message.get("type", "")
                parsed = {
                    "message_id": message.get("id"),
                    "from": message.get("from"),
                    "type": msg_type,
                    "phone_number_id": phone_number_id,
                    "text": "",
                    "media_id": None,
                }
                if msg_type == "text":
                    parsed["text"] = message.get("text", {}).get("body", "")
                elif msg_type == "image":
                    parsed["media_id"] = message.get("image", {}).get("id")
                    parsed["text"] = message.get("image", {}).get("caption", "")
                messages.append(parsed)
    return messages


def format_whatsapp_reply(advice: dict[str, Any]) -> str:
    lines = [
        "AgriBloom crop advice",
        "",
        f"Crop: {advice.get('crop', 'Unknown')}",
        f"Problem: {advice.get('problem', 'Unknown')}",
        f"Risk: {str(advice.get('risk_level', 'unknown')).title()}",
        "",
        "What to do today:",
    ]
    for index, item in enumerate(advice.get("what_to_do_today", [])[:4], 1):
        lines.append(f"{index}. {item}")

    treatment = advice.get("treatment_guidance")
    if treatment:
        lines.extend(["", f"Treatment: {treatment}"])

    expert = advice.get("when_to_call_expert")
    if expert:
        lines.extend(["", f"Expert help: {expert}"])

    helpline = advice.get("helpline", {}) or {}
    if helpline.get("number"):
        lines.append(f"Helpline: {helpline['number']}")

    return "\n".join(str(line) for line in lines)[:3500]


def _whatsapp_token() -> str:
    return os.getenv("WHATSAPP_ACCESS_TOKEN", "").strip()


def _phone_number_id(default: str | None = None) -> str:
    return os.getenv("WHATSAPP_PHONE_NUMBER_ID", "").strip() or (default or "")


def send_whatsapp_text(to: str, text: str, phone_number_id: str | None = None) -> bool:
    token = _whatsapp_token()
    sender_id = _phone_number_id(phone_number_id)
    if not token or not sender_id or not to:
        return False

    try:
        import urllib.request

        url = f"https://graph.facebook.com/{GRAPH_VERSION}/{sender_id}/messages"
        payload = json.dumps({
            "messaging_product": "whatsapp",
            "to": to,
            "type": "text",
            "text": {"preview_url": False, "body": text},
        }).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


def download_whatsapp_media(media_id: str) -> bytes | None:
    token = _whatsapp_token()
    if not token or not media_id:
        return None

    try:
        import urllib.request

        meta_req = urllib.request.Request(
            f"https://graph.facebook.com/{GRAPH_VERSION}/{media_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(meta_req, timeout=15) as response:
            media_url = json.loads(response.read().decode("utf-8")).get("url")
        if not media_url:
            return None

        media_req = urllib.request.Request(
            media_url,
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(media_req, timeout=30) as response:
            return response.read()
    except Exception:
        return None


__all__ = [
    "download_whatsapp_media",
    "format_whatsapp_reply",
    "parse_whatsapp_messages",
    "send_whatsapp_text",
]

