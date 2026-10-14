'use client'

import { useEffect, useState } from 'react'
import { Bot } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useChatStore } from '@/lib/stores/chatStore'
import dynamic from 'next/dynamic'

// Lazy-load the heavy panel (includes react-markdown)
const ChatPanel = dynamic(
  () => import('./ChatPanel').then((m) => ({ default: m.ChatPanel })),
  { ssr: false }
)

export function ChatWidget() {
  const { isOpen, toggleChat, messages } = useChatStore()

  // Track unread count (messages since panel was last opened)
  const [lastSeenCount, setLastSeenCount] = useState(messages.length)
  const [unreadCount, setUnreadCount] = useState(0)

  useEffect(() => {
    if (isOpen) {
      setLastSeenCount(messages.length)
      setUnreadCount(0)
    } else {
      const newAssistantMessages = messages
        .slice(lastSeenCount)
        .filter((m) => m.role === 'assistant' && m.content).length
      if (newAssistantMessages > 0) {
        setUnreadCount((c) => c + newAssistantMessages)
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [messages.length, isOpen])

  return (
    <>
      {/* Floating action button */}
      <div
        id="chat-widget-fab"
        className={`fixed bottom-6 right-6 transition-opacity duration-200 ${isOpen ? 'opacity-0 pointer-events-none' : 'opacity-100'}`}
        style={{ zIndex: 9990 }}
      >
        <Button
          id="chat-toggle-btn"
          onClick={toggleChat}
          size="icon"
          className="h-12 w-12 rounded-full shadow-lg"
          aria-label={isOpen ? 'Close FarmBot AI' : 'Open FarmBot AI'}
        >
          <Bot className="h-5 w-5" />
          {!isOpen && unreadCount > 0 && (
            <span
              id="chat-unread-badge"
              className="absolute -top-1 -right-1 h-4 w-4 rounded-full bg-destructive text-destructive-foreground text-[10px] font-bold flex items-center justify-center"
            >
              {unreadCount > 9 ? '9+' : unreadCount}
            </span>
          )}
        </Button>
      </div>

      {/* Slide-in chat panel */}
      <div
        id="chat-panel-container"
        className={`fixed inset-y-0 right-0 w-[400px] shadow-xl transition-transform duration-300 ease-in-out ${
          isOpen ? 'translate-x-0' : 'translate-x-full'
        }`}
        style={{ zIndex: 9989 }}
        aria-hidden={!isOpen}
      >
        {/* Only mount panel content when open for performance */}
        {isOpen && <ChatPanel />}
      </div>
    </>
  )
}
