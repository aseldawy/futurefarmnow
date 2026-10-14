"""
Flask blueprint: ai_assistant

Endpoint:
  POST /ai/chat  — Multipart chat with optional image upload.

Security:
  - Prompt injection pre-scan (regex)
  - Input size limits
  - Magic-byte image validation
  - GEMINI_API_KEY never exposed in responses
"""

import re
import json
import logging

from flask import Blueprint, request, jsonify

logger = logging.getLogger(__name__)

ai_assistant_bp = Blueprint("ai_assistant", __name__, url_prefix="/ai")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_MESSAGE_LENGTH = 2000
MAX_HISTORY_TURNS = 20
MAX_IMAGE_BYTES = 4 * 1024 * 1024      # 4 MB
ALLOWED_IMAGE_MIMES = {"image/jpeg", "image/png", "image/webp"}

_VALID_ROLES = {"user", "model"}
_HISTORY_CHAR_BUDGET = 12_000          # max total chars across all history turns

# Magic bytes for each allowed MIME type.
# We verify the actual file content, not just the client-supplied Content-Type.
_IMAGE_MAGIC: dict[str, list[bytes]] = {
    "image/jpeg": [b"\xff\xd8\xff"],
    "image/png":  [b"\x89PNG\r\n\x1a\n"],
    "image/webp": [b"RIFF"],   # RIFF....WEBP — checked further below
}

# Prompt injection patterns (case-insensitive)
# Covers the most common jailbreak / instruction-override attempts.
# This is a defense-in-depth layer — the system-prompt guardrail is the primary defense.
_INJECTION_PATTERNS = re.compile(
    # Classic instruction overrides
    r"ignore\s+(previous|above|all)\s+instruction"
    r"|disregard\s+(all|your|previous|the\s+(above|following))"
    r"|forget\s+(your|all|previous|the\s+above)\s+(instruction|rule|constraint|prompt|guideline)"
    r"|override\s+(your|all|previous|the)\s+(instruction|rule|constraint|prompt)"
    r"|new\s+instruction"
    # System prompt extraction / leaking
    r"|system\s*prompt"
    r"|reveal\s+(your|the)\s+(prompt|instruction|system)"
    r"|repeat\s+(everything|the)\s+(above|before|prior|system)"
    r"|print\s+(your|the)\s+(prompt|instruction|system)"
    r"|show\s+me\s+(your|the)\s+(prompt|instruction)"
    # Persona switching
    r"|you\s+are\s+now\s+(?!a\s+farmer|an?\s+agronomist)"
    r"|pretend\s+(you\s+are|to\s+be)"
    r"|act\s+as\s+(?!a\s+farmer|an?\s+agronomist)"
    r"|roleplay\s+as"
    r"|simulate\s+(being|an?\s+AI|a\s+model)"
    r"|your\s+true\s+self"
    r"|no\s+longer\s+bound"
    r"|without\s+(any\s+)?(restriction|limitation|filter|constraint)"
    r"|unrestricted\s+mode"
    # Named jailbreak modes
    r"|jailbreak"
    r"|DAN\s+mode"
    r"|developer\s+mode"
    r"|god\s+mode"
    r"|sudo\s+mode"
    r"|training\s+mode"
    r"|maintenance\s+mode",
    re.IGNORECASE,
)

# Language-force prefixes (stripped before storing in history)
_LANG_FORCE_RE = re.compile(r"^\[(Respond in (Spanish|English))\]\s*", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@ai_assistant_bp.route("/chat", methods=["POST"])
def chat():
    """
    POST /ai/chat

    Accepts multipart/form-data:
      message  string  required
      context  string  JSON — frontend map state
      history  string  JSON array [{role, content}]
      image    file    optional field/crop photo

    Returns JSON:
      {reply, tool_calls, sources, detected_language}
    """
    # ---- Parse message ----
    message = request.form.get("message", "").strip()
    if not message:
        return jsonify({"error": "invalid_input", "message": "Message is required."}), 400
    if len(message) > MAX_MESSAGE_LENGTH:
        return jsonify({
            "error": "invalid_input",
            "message": f"Message too long (max {MAX_MESSAGE_LENGTH} characters).",
        }), 400

    # Strip language-force prefix before injection scan (allow legitimate use)
    clean_message = _LANG_FORCE_RE.sub("", message).strip()

    # ---- Prompt injection scan ----
    if _INJECTION_PATTERNS.search(clean_message):
        logger.warning("Prompt injection attempt: %.100s", clean_message)
        return jsonify({
            "error": "invalid_input",
            "message": "That input isn't something I can process. Please ask a farming question.",
        }), 400

    # ---- Parse context ----
    context_raw = request.form.get("context", "{}")
    try:
        context = json.loads(context_raw)
        if not isinstance(context, dict):
            context = {}
    except (json.JSONDecodeError, ValueError):
        context = {}

    # ---- Parse history ----
    history_raw = request.form.get("history", "[]")
    try:
        history = json.loads(history_raw)
        if not isinstance(history, list):
            history = []
    except (json.JSONDecodeError, ValueError):
        history = []

    # Sanitise: whitelist roles, cap per-turn content length
    history = [
        {
            "role": t.get("role") if t.get("role") in _VALID_ROLES else "user",
            "content": str(t.get("content", ""))[:4000],
        }
        for t in history
        if isinstance(t, dict)
    ]
    # Cap to recent N turns
    history = history[-MAX_HISTORY_TURNS:]
    # Trim by total character budget (keep most recent turns)
    total = 0
    trimmed = []
    for turn in reversed(history):
        total += len(turn["content"])
        if total > _HISTORY_CHAR_BUDGET:
            break
        trimmed.append(turn)
    history = list(reversed(trimmed))

    # ---- Parse optional image ----
    image_bytes: bytes | None = None
    image_mime = "image/jpeg"
    image_file = request.files.get("image")
    if image_file:
        # 1. Check client-supplied MIME type against whitelist
        mime = (image_file.mimetype or "").lower()
        if mime not in ALLOWED_IMAGE_MIMES:
            return jsonify({
                "error": "invalid_input",
                "message": "Unsupported image format. Please use JPEG, PNG, or WebP.",
            }), 400

        # 2. Read (size-capped)
        raw = image_file.read(MAX_IMAGE_BYTES + 1)
        if len(raw) > MAX_IMAGE_BYTES:
            return jsonify({
                "error": "invalid_input",
                "message": "Image too large (max 4 MB).",
            }), 400

        # 3. Validate magic bytes — reject files whose content doesn't match the declared type
        magic_ok = False
        for sig in _IMAGE_MAGIC.get(mime, []):
            if raw.startswith(sig):
                # Extra check for WebP: bytes 8-12 must be "WEBP"
                if mime == "image/webp":
                    magic_ok = len(raw) >= 12 and raw[8:12] == b"WEBP"
                else:
                    magic_ok = True
                break
        if not magic_ok:
            logger.warning("Image magic-byte mismatch (declared %s)", mime)
            return jsonify({
                "error": "invalid_input",
                "message": "Image file content does not match the declared format.",
            }), 400

        image_bytes = raw
        image_mime = mime

    # ---- Call Gemini ----
    try:
        from .gemini_client import chat as gemini_chat
        result = gemini_chat(
            message=message,          # pass original (with language prefix if any)
            context=context,
            history=history,
            image_bytes=image_bytes,
            image_mime=image_mime,
        )
    except RuntimeError as e:
        # Missing API key or SDK init failure
        logger.error("Gemini client error: %s", e)
        return jsonify({
            "error": "gemini_error",
            "message": "AI service is not configured. Please contact the administrator.",
        }), 503
    except Exception as e:
        logger.exception("Unexpected error in Gemini chat: %s", e)
        return jsonify({
            "error": "gemini_error",
            "message": "AI service encountered an error. Please try again.",
        }), 500

    return jsonify({
        "reply": result.get("reply", ""),
        "tool_calls": result.get("tool_calls", []),
        "sources": result.get("sources", []),
        "detected_language": result.get("detected_language", "en"),
    })
