'use client'

import ReactMarkdown from 'react-markdown'
import { Bot, User, AlertCircle, ExternalLink } from 'lucide-react'
import { ChatMessage as ChatMessageType } from '@/lib/types/chat'
import { ToolCallDisplay } from './ToolCallDisplay'

interface Props {
  message: ChatMessageType
  onToggleTool: (messageId: string, toolIndex: number) => void
}

// Only render model-returned source links that are well-formed https URLs
function isSafeSourceUrl(url: unknown): url is string {
  if (typeof url !== 'string') return false
  try {
    return new URL(url).protocol === 'https:'
  } catch {
    return false
  }
}

export function ChatMessage({ message, onToggleTool }: Props) {
  const isUser = message.role === 'user'
  const isError = !!message.error
  const safeSources = (message.sources ?? []).filter(isSafeSourceUrl)

  return (
    <div
      id={`chat-msg-${message.id}`}
      className={`flex gap-2 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}
    >
      {/* Avatar */}
      <div
        id={`chat-avatar-${message.id}`}
        className={`shrink-0 h-7 w-7 rounded-full flex items-center justify-center mt-0.5 ${
          isUser
            ? 'bg-primary text-primary-foreground'
            : 'bg-muted border border-border'
        }`}
      >
        {isUser
          ? <User className="h-4 w-4" />
          : <Bot className="h-4 w-4 text-primary" />
        }
      </div>

      {/* Bubble */}
      <div
        id={`chat-bubble-${message.id}`}
        className={`max-w-[82%] space-y-1.5 ${isUser ? 'items-end' : 'items-start'} flex flex-col`}
      >
        {/* Image preview (user uploads) */}
        {message.imagePreviewUrl && (
          <div id={`chat-img-preview-${message.id}`} className="rounded-lg overflow-hidden border border-border max-w-[200px]">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={message.imagePreviewUrl}
              alt="Uploaded field photo"
              className="object-cover w-full"
            />
          </div>
        )}

        {/* Text content */}
        <div
          id={`chat-content-${message.id}`}
          className={`rounded-2xl px-3 py-2 text-sm leading-relaxed ${
            isUser
              ? 'bg-primary text-primary-foreground rounded-tr-sm'
              : isError
              ? 'bg-destructive/10 border border-destructive/30 text-foreground rounded-tl-sm'
              : 'bg-card border border-border text-foreground rounded-tl-sm'
          }`}
        >
          {isError && (
            <div className="flex items-center gap-1.5 mb-1 text-destructive text-xs font-medium">
              <AlertCircle className="h-3.5 w-3.5" />
              <span>Could not process request</span>
            </div>
          )}

          {isUser ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : (
            <div id={`chat-markdown-${message.id}`} className="prose prose-sm dark:prose-invert max-w-none prose-p:my-0.5 prose-ul:my-0.5 prose-li:my-0 prose-headings:mt-2 prose-headings:mb-0.5">
              <ReactMarkdown>{message.content || (message.error ? 'Sorry, something went wrong. Please try again.' : '')}</ReactMarkdown>
            </div>
          )}
        </div>

        {/* Tool calls (assistant only) */}
        {!isUser && message.toolCalls && message.toolCalls.length > 0 && (
          <div id={`chat-tools-${message.id}`} className="w-full">
            <ToolCallDisplay
              messageId={message.id}
              toolCalls={message.toolCalls}
              onToggle={(toolIndex) => onToggleTool(message.id, toolIndex)}
            />
          </div>
        )}

        {/* Web sources */}
        {!isUser && safeSources.length > 0 && (
          <div id={`chat-sources-${message.id}`} className="text-[11px] text-muted-foreground space-y-0.5">
            <p className="font-medium">Sources</p>
            {safeSources.slice(0, 3).map((url, i) => (
              <a
                key={i}
                href={url}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1 hover:text-foreground transition-colors truncate"
              >
                <ExternalLink className="h-3 w-3 shrink-0" />
                <span className="truncate">{url}</span>
              </a>
            ))}
          </div>
        )}

        {/* Timestamp */}
        <time
          id={`chat-time-${message.id}`}
          className="text-[10px] text-muted-foreground px-1"
          dateTime={new Date(message.timestamp).toISOString()}
        >
          {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </time>
      </div>
    </div>
  )
}
