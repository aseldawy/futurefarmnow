'use client'

import { useRef, useState, useEffect, KeyboardEvent } from 'react'
import { Send, Paperclip, X, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'

const MAX_CHARS = 2000
const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp']

interface Props {
  isLoading: boolean
  detectedLanguage: 'en' | 'es'
  pendingImagePreview: string | null
  initialMessage?: string | null
  onSend: (message: string) => void
  onImageSelect: (file: File, preview: string) => void
  onImageRemove: () => void
  onForceLanguage: (lang: 'en' | 'es') => void
  onInitialMessageConsumed?: () => void
}

export function ChatInput({
  isLoading,
  detectedLanguage,
  pendingImagePreview,
  initialMessage,
  onSend,
  onImageSelect,
  onImageRemove,
  onForceLanguage,
  onInitialMessageConsumed,
}: Props) {
  const [text, setText] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Pre-populate textarea when a pending message is injected (e.g. from "Analyze with AI")
  useEffect(() => {
    if (initialMessage) {
      setText(initialMessage)
      onInitialMessageConsumed?.()
      // Focus textarea so user can edit/send immediately
      setTimeout(() => textareaRef.current?.focus(), 50)
    }
  }, [initialMessage, onInitialMessageConsumed])

  const canSend = text.trim().length > 0 && !isLoading

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      if (canSend) submit()
    }
  }

  function submit() {
    const msg = text.trim()
    if (!msg) return
    onSend(msg)
    setText('')
    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  function handleTextChange(e: React.ChangeEvent<HTMLTextAreaElement>) {
    const val = e.target.value
    if (val.length > MAX_CHARS) return
    setText(val)
    // Auto-resize
    const ta = e.target
    ta.style.height = 'auto'
    ta.style.height = `${Math.min(ta.scrollHeight, 120)}px`
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    if (!ALLOWED_TYPES.includes(file.type)) return
    const reader = new FileReader()
    reader.onload = (ev) => {
      onImageSelect(file, ev.target?.result as string)
    }
    reader.readAsDataURL(file)
    // Reset input so same file can be re-selected
    e.target.value = ''
  }

  const otherLang: 'en' | 'es' = detectedLanguage === 'en' ? 'es' : 'en'
  const langLabel = detectedLanguage === 'en' ? 'EN' : 'ES'
  const langTitle =
    detectedLanguage === 'en'
      ? 'Switch to Spanish'
      : 'Switch to English'

  return (
    <div id="chat-input-container" className="border-t border-border bg-background px-3 pt-2 pb-3 space-y-2">
      {/* Image preview strip */}
      {pendingImagePreview && (
        <div id="chat-image-preview" className="flex items-center gap-2 p-2 rounded-lg bg-muted/50 border border-border">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={pendingImagePreview}
            alt="Selected photo"
            className="h-10 w-10 rounded object-cover border border-border"
          />
          <span className="text-xs text-muted-foreground flex-1 truncate">Photo attached</span>
          <button
            id="chat-remove-image"
            onClick={onImageRemove}
            className="text-muted-foreground hover:text-foreground transition-colors"
            aria-label="Remove image"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      )}

      {/* Input row */}
      <div id="chat-input-row" className="flex items-end gap-2">
        {/* Image upload */}
        <div>
          <input
            id="chat-file-input"
            ref={fileInputRef}
            type="file"
            accept="image/jpeg,image/png,image/webp"
            className="hidden"
            onChange={handleFileChange}
          />
          <Button
            id="chat-attach-btn"
            type="button"
            variant="ghost"
            size="icon"
            className="h-8 w-8 shrink-0 text-muted-foreground hover:text-foreground"
            onClick={() => fileInputRef.current?.click()}
            title="Attach a field photo"
            disabled={isLoading}
          >
            <Paperclip className="h-4 w-4" />
          </Button>
        </div>

        {/* Textarea */}
        <div className="relative flex-1">
          <textarea
            id="chat-textarea"
            ref={textareaRef}
            value={text}
            onChange={handleTextChange}
            onKeyDown={handleKeyDown}
            rows={1}
            placeholder={detectedLanguage === 'es' ? 'Pregunta sobre tu campo…' : 'Ask about your field…'}
            disabled={isLoading}
            className="w-full resize-none rounded-xl border border-input bg-background px-3 py-2 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring disabled:opacity-50 leading-relaxed"
            style={{ minHeight: '36px', maxHeight: '120px' }}
          />
          {/* Char counter — only show when nearing limit */}
          {text.length > MAX_CHARS * 0.8 && (
            <span
              id="chat-char-counter"
              className={`absolute bottom-1.5 right-2 text-[10px] tabular-nums ${
                text.length >= MAX_CHARS ? 'text-destructive' : 'text-muted-foreground'
              }`}
            >
              {MAX_CHARS - text.length}
            </span>
          )}
        </div>

        {/* Language badge */}
        <Button
          id="chat-lang-btn"
          type="button"
          variant="outline"
          size="sm"
          className="h-8 px-2 text-[11px] font-bold shrink-0"
          onClick={() => onForceLanguage(otherLang)}
          title={langTitle}
        >
          {langLabel}
        </Button>

        {/* Send */}
        <Button
          id="chat-send-btn"
          type="button"
          size="icon"
          className="h-8 w-8 shrink-0"
          disabled={!canSend}
          onClick={submit}
          aria-label="Send message"
        >
          {isLoading
            ? <Loader2 className="h-4 w-4 animate-spin" />
            : <Send className="h-4 w-4" />
          }
        </Button>
      </div>

      <p id="chat-hint" className="text-[10px] text-muted-foreground text-center">
        {detectedLanguage === 'es'
          ? 'Respuestas de orientación — confirme decisiones importantes con su asesor agrícola'
          : 'Responses are a guide — verify important decisions with your farm advisor'
        }
      </p>
    </div>
  )
}
