"""
Package: app.ai_assistant

Provides the FarmBot AI assistant blueprint — a bilingual (EN/ES) farming
assistant powered by Gemini 2.5 Flash. All Gemini API calls are server-side;
the API key never reaches the frontend.

API Endpoints (all under /ai):
- POST /ai/chat  — Multimodal chat with tool use, image analysis, search grounding

Security:
- API key loaded from GEMINI_API_KEY env var only
- Per-IP rate limiting (20 req/min)
- Prompt injection pre-scan
- Domain guardrails enforced via system prompt

Usage:
    from app.ai_assistant import ai_assistant_bp
    app.register_blueprint(ai_assistant_bp)
"""

from .routes import ai_assistant_bp

__all__ = ['ai_assistant_bp']
