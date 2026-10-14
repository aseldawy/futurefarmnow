import { ChatContext, ChatAPIResponse, ChatMessage } from '@/lib/types/chat'

const AI_BASE_URL =
  process.env.NEXT_PUBLIC_AI_BASE_URL ||
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace('/futurefarmnow-backend-0.3-RC1', '') ||
  'http://localhost:5200'

export async function sendChatMessage(params: {
  message: string
  context: ChatContext
  history: ChatMessage[]
  imageFile?: File | null
}): Promise<ChatAPIResponse> {
  const formData = new FormData()
  formData.append('message', params.message)
  formData.append('context', JSON.stringify(params.context))

  // Send history as simplified role/content pairs
  const historyPayload = params.history.map((m) => ({
    role: m.role === 'assistant' ? 'model' : 'user',
    content: m.content,
  }))
  formData.append('history', JSON.stringify(historyPayload))

  if (params.imageFile) {
    formData.append('image', params.imageFile)
  }

  const response = await fetch(`${AI_BASE_URL}/ai/chat`, {
    method: 'POST',
    body: formData,
  })

  const data = await response.json()

  if (!response.ok) {
    throw new Error(data.message || `Request failed with status ${response.status}`)
  }

  return data as ChatAPIResponse
}
