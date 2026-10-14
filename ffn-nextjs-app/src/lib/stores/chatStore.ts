import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { ChatMessage, ToolCallRecord } from '@/lib/types/chat'

const MAX_PERSISTED_MESSAGES = 20

interface ChatState {
  isOpen: boolean
  messages: ChatMessage[]
  isLoading: boolean
  detectedLanguage: 'en' | 'es'
  pendingImageFile: File | null
  pendingImagePreview: string | null
  pendingMessage: string | null

  // Actions
  openChat: () => void
  closeChat: () => void
  toggleChat: () => void
  addMessage: (msg: ChatMessage) => void
  updateLastAssistantMessage: (partial: Partial<ChatMessage>) => void
  toggleToolCall: (messageId: string, toolIndex: number) => void
  setIsLoading: (v: boolean) => void
  setDetectedLanguage: (lang: 'en' | 'es') => void
  setPendingImage: (file: File | null, preview: string | null) => void
  setPendingMessage: (msg: string | null) => void
  clearHistory: () => void
}

export const useChatStore = create<ChatState>()(
  persist(
    (set) => ({
      isOpen: false,
      messages: [],
      isLoading: false,
      detectedLanguage: 'en',
      pendingImageFile: null,
      pendingImagePreview: null,
      pendingMessage: null,

      openChat: () => set({ isOpen: true }),
      closeChat: () => set({ isOpen: false }),
      toggleChat: () => set((s) => ({ isOpen: !s.isOpen })),

      addMessage: (msg) =>
        set((s) => {
          const messages = [...s.messages, msg]
          // Trim to max persisted length
          return { messages: messages.slice(-MAX_PERSISTED_MESSAGES) }
        }),

      updateLastAssistantMessage: (partial) =>
        set((s) => {
          const messages = [...s.messages]
          // Find the last assistant message
          for (let i = messages.length - 1; i >= 0; i--) {
            if (messages[i].role === 'assistant') {
              messages[i] = { ...messages[i], ...partial }
              break
            }
          }
          return { messages }
        }),

      toggleToolCall: (messageId, toolIndex) =>
        set((s) => {
          const messages = s.messages.map((msg) => {
            if (msg.id !== messageId || !msg.toolCalls) return msg
            const toolCalls: ToolCallRecord[] = msg.toolCalls.map((tc, i) =>
              i === toolIndex ? { ...tc, isExpanded: !tc.isExpanded } : tc
            )
            return { ...msg, toolCalls }
          })
          return { messages }
        }),

      setIsLoading: (v) => set({ isLoading: v }),
      setDetectedLanguage: (lang) => set({ detectedLanguage: lang }),

      setPendingImage: (file, preview) =>
        set({ pendingImageFile: file, pendingImagePreview: preview }),

      setPendingMessage: (msg) => set({ pendingMessage: msg }),

      clearHistory: () =>
        set({ messages: [] }),

    }),
    {
      name: 'ffn-chat',
      // Only persist messages and language — not UI state or file refs
      partialize: (s) => ({
        messages: s.messages,
        detectedLanguage: s.detectedLanguage,
      }),
    }
  )
)
