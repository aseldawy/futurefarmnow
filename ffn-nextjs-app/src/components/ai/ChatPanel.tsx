'use client'

import { useEffect, useRef } from 'react'
import { Trash2, Bot, Info, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useMapStore } from '@/lib/stores/mapStore'
import { useFarmStore } from '@/lib/stores/farmStore'
import { useChatStore } from '@/lib/stores/chatStore'
import { sendChatMessage } from '@/lib/api/aiClient'
import { ChatContext, ChatMessage as ChatMessageType, WeatherContext } from '@/lib/types/chat'
import { GeoJSONGeometry } from '@/lib/types/api'
import { processForecastData } from '@/lib/utils/weather'

function buildWeatherContext(farmState: ReturnType<typeof useFarmStore.getState>): WeatherContext | null {
  const { farms, selectedFarmId, forecastData, weatherEndpoints } = farmState
  const selectedFarm = farms.find(f => f.id === selectedFarmId) ?? null
  if (!selectedFarm || !forecastData?.forecast || !forecastData?.hourly) return null
  const endpoints = selectedFarmId ? weatherEndpoints[selectedFarmId] : null
  const days = processForecastData(forecastData.forecast, forecastData.hourly)
  return {
    farmName: selectedFarm.name,
    location: endpoints ? `${endpoints.city}, ${endpoints.state}` : '',
    days: days.map(d => ({
      dayName: d.dayName,
      date: d.date,
      highTemp: d.highTemp,
      precipPercent: d.precipPercent,
      goNoGo: d.goNoGo,
      sprayGoodHours: d.sprayGoodHours,
      sprayMarginalHours: d.sprayMarginalHours,
      harvestGoodHours: d.harvestGoodHours,
      harvestMarginalHours: d.harvestMarginalHours,
      shortForecast: d.shortForecast,
      windSummary: d.windSummary,
    })),
  }
}

function polygonCentroid(geojson: GeoJSONGeometry): [number, number] | null {
  if (!geojson || geojson.type !== 'Polygon') return null
  const coords = geojson.coordinates[0] as [number, number][]
  const lat = coords.reduce((s, c) => s + c[1], 0) / coords.length
  const lng = coords.reduce((s, c) => s + c[0], 0) / coords.length
  return [lat, lng]
}
import { ChatMessage } from './ChatMessage'
import { ChatInput } from './ChatInput'
import { ContextIndicator } from './ContextIndicator'
import { toast } from '@/lib/utils/toast'

function buildContext(): ChatContext {
  // Read directly from stores (not hooks — this is called outside render)
  // We'll use Zustand's getState for this
  const mapState = useMapStore.getState()
  const farmState = useFarmStore.getState()

  return {
    hasPolygon: !!mapState.drawnPolygon,
    polygonGeojson: mapState.drawnPolygon,
    selectedSoilLayer: mapState.selectedSoilLayer ?? 'ph',
    selectedSoilDepth: mapState.selectedSoilDepth ?? '0-5',
    selectedDateRange: mapState.selectedDateRange ?? { from: '', to: '' },
    activeTab: mapState.activeTab ?? 'soil',
    hasSoilResults: !!mapState.soilImageUrl,
    hasNDVIResults: !!mapState.ndviImageUrl,
    soilAnalysisResult: mapState.soilAnalysisResult ?? null,
    ndviAnalysisResult: mapState.ndviAnalysisResult ?? null,
    farmCount: farmState.farms?.length ?? 0,
    currentZoom: mapState.currentZoom ?? 10,
    mapBounds: mapState.currentBounds ?? null,
    mapCenter: mapState.drawnPolygon
      ? polygonCentroid(mapState.drawnPolygon)
      : (mapState.currentCenter ?? null),
    selectedDataset: mapState.selectedDataset ?? null,
    hasETMapResult: !!mapState.etmapImageUrl,
    etmapRequestId: mapState.etmapRequestId ?? null,
    etmapAnalysisResult: mapState.etmapAnalysisResult ?? null,
    weatherContext: buildWeatherContext(farmState),
  }
}

function makeId(): string {
  return `msg-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`
}

// Lang-force prefix stripped before history storage
const LANG_FORCE_RE = /^\[(Respond in (Spanish|English))\]\s*/i

export function ChatPanel() {
  const {
    messages,
    isLoading,
    detectedLanguage,
    pendingImageFile,
    pendingImagePreview,
    pendingMessage,
    addMessage,
    toggleToolCall,
    setIsLoading,
    setDetectedLanguage,
    setPendingImage,
    setPendingMessage,
    clearHistory,
  } = useChatStore()

  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages.length])

  // Build current context for ContextIndicator display
  const mapState = useMapStore()
  const farmState = useFarmStore()
  const liveContext: ChatContext = {
    hasPolygon: !!mapState.drawnPolygon,
    polygonGeojson: mapState.drawnPolygon,
    selectedSoilLayer: mapState.selectedSoilLayer ?? 'ph',
    selectedSoilDepth: mapState.selectedSoilDepth ?? '0-5',
    selectedDateRange: mapState.selectedDateRange ?? { from: '', to: '' },
    activeTab: mapState.activeTab ?? 'soil',
    hasSoilResults: !!mapState.soilImageUrl,
    hasNDVIResults: !!mapState.ndviImageUrl,
    soilAnalysisResult: mapState.soilAnalysisResult ?? null,
    ndviAnalysisResult: mapState.ndviAnalysisResult ?? null,
    farmCount: farmState.farms?.length ?? 0,
    currentZoom: mapState.currentZoom ?? 10,
    mapBounds: mapState.currentBounds ?? null,
    mapCenter: mapState.drawnPolygon
      ? polygonCentroid(mapState.drawnPolygon)
      : (mapState.currentCenter ?? null),
    selectedDataset: mapState.selectedDataset ?? null,
    hasETMapResult: !!mapState.etmapImageUrl,
    etmapRequestId: mapState.etmapRequestId ?? null,
    etmapAnalysisResult: mapState.etmapAnalysisResult ?? null,
    weatherContext: buildWeatherContext(farmState),
  }

  async function handleSend(rawMessage: string) {
    if (isLoading) return

    // Handle language-force prefix
    let message = rawMessage
    let langPrefix = ''
    const langMatch = LANG_FORCE_RE.exec(rawMessage)
    if (langMatch) {
      langPrefix = `[${langMatch[1]}] `
      // Keep the prefix in the sent message so the model sees it,
      // but strip it for display and history storage
      message = rawMessage.replace(LANG_FORCE_RE, '').trim()
    }

    const imagePreviewUrl = pendingImagePreview ?? undefined
    const imageFile = pendingImageFile

    // Add user message immediately
    const userMsg: ChatMessageType = {
      id: makeId(),
      role: 'user',
      content: message,
      imagePreviewUrl,
      timestamp: Date.now(),
    }
    addMessage(userMsg)
    setPendingImage(null, null)
    setIsLoading(true)

    try {
      const context = buildContext()
      const history = messages.filter((m) => m.content)

      const response = await sendChatMessage({
        message: langPrefix + message,
        context,
        history,
        imageFile,
      })

      setDetectedLanguage(response.detected_language)

      addMessage({
        id: makeId(),
        role: 'assistant',
        content: response.reply,
        toolCalls: response.tool_calls?.map((tc) => ({
          ...tc,
          isExpanded: false,
        })),
        sources: response.sources,
        detectedLanguage: response.detected_language,
        timestamp: Date.now(),
      })
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : 'Something went wrong.'
      addMessage({
        id: makeId(),
        role: 'assistant',
        content: '',
        error: errMsg,
        timestamp: Date.now(),
      })
      toast.error('AI assistant error: ' + errMsg)
    } finally {
      setIsLoading(false)
    }
  }

  function handleForceLanguage(lang: 'en' | 'es') {
    const prefix = lang === 'es' ? '[Respond in Spanish] ' : '[Respond in English] '
    // Send a follow-up language switch message silently
    handleSend(prefix + (lang === 'es' ? 'Hola' : 'Hello'))
    setDetectedLanguage(lang)
  }

  return (
    <div id="chat-panel" className="flex flex-col h-full bg-background">
      {/* Header */}
      <div id="chat-panel-header" className="flex items-center gap-2 px-3 py-3 border-b border-border bg-background">
        <div className="h-7 w-7 rounded-full bg-primary/10 border border-primary/30 flex items-center justify-center">
          <Bot className="h-4 w-4 text-primary" />
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-semibold leading-none">FarmBot AI</p>
          <p className="text-[10px] text-muted-foreground mt-0.5">
            {isLoading ? 'Thinking…' : 'Ready • gemini-2.5-flash'}
          </p>
        </div>
        <Button
          id="chat-clear-btn"
          variant="ghost"
          size="icon"
          className="h-7 w-7 text-muted-foreground hover:text-foreground"
          onClick={() => {
            clearHistory()
            toast.success('Conversation cleared')
          }}
          title="Clear conversation"
        >
          <Trash2 className="h-3.5 w-3.5" />
        </Button>
        <Button
          id="chat-close-btn"
          variant="ghost"
          size="icon"
          className="h-7 w-7 text-muted-foreground hover:text-foreground"
          onClick={useChatStore.getState().closeChat}
          title="Close"
        >
          <X className="h-4 w-4" />
        </Button>
      </div>

      {/* Context indicator */}
      <ContextIndicator context={liveContext} />

      {/* Messages */}
      <div
        id="chat-messages"
        className="flex-1 overflow-y-auto px-3 py-3 space-y-4"
      >
        {messages.length === 0 && (
          <div id="chat-empty-state" className="flex flex-col items-center justify-center h-full text-center space-y-3 py-8">
            <div className="h-12 w-12 rounded-full bg-primary/10 border border-primary/20 flex items-center justify-center">
              <Bot className="h-6 w-6 text-primary" />
            </div>
            <div>
              <p className="text-sm font-medium">Hi! I&apos;m FarmBot</p>
              <p className="text-xs text-muted-foreground mt-1 max-w-[220px]">
                Ask me about soil, crops, irrigation (CIMIS), pest risk, market prices, or upload a field photo.
                I speak English and Spanish.
              </p>
            </div>
            <div id="chat-suggestions" className="space-y-1.5 w-full max-w-[260px]">
              {[
                'How much should I irrigate this week?',
                'What pests should I watch for right now?',
                "What's the market price for tomatoes?",
                'Is my soil pH good for almonds?',
                'Why is my NDVI low this month?',
                '¿Cuándo debo regar este campo?',
              ].map((suggestion) => (
                <button
                  key={suggestion}
                  id={`suggestion-${suggestion.slice(0, 20).replace(/\s+/g, '-').toLowerCase()}`}
                  onClick={() => handleSend(suggestion)}
                  className="w-full text-left text-xs px-3 py-2 rounded-lg border border-border hover:bg-muted/50 transition-colors text-muted-foreground hover:text-foreground"
                >
                  {suggestion}
                </button>
              ))}
            </div>

            {/* Disclaimer */}
            <div
              id="chat-disclaimer"
              className="flex gap-2 rounded-lg border border-amber-200 bg-amber-50 dark:border-amber-900/50 dark:bg-amber-950/30 px-3 py-2.5 text-left max-w-[280px]"
            >
              <Info className="h-3.5 w-3.5 text-amber-600 dark:text-amber-400 shrink-0 mt-0.5" />
              <p className="text-[11px] text-amber-800 dark:text-amber-300 leading-relaxed">
                FarmBot is a helpful guide, not a licensed agronomist. Always confirm
                important decisions with your farm advisor, county extension office,
                or certified crop consultant.
              </p>
            </div>
          </div>
        )}

        {messages.map((msg) => (
          <ChatMessage
            key={msg.id}
            message={msg}
            onToggleTool={toggleToolCall}
          />
        ))}

        {/* Loading dots while waiting for assistant response */}
        {isLoading && (
          <div id="chat-thinking" className="flex gap-2">
            <div className="h-7 w-7 rounded-full bg-muted border border-border flex items-center justify-center mt-0.5">
              <Bot className="h-4 w-4 text-primary" />
            </div>
            <div className="bg-card border border-border rounded-2xl rounded-tl-sm px-4 py-3">
              <div className="flex gap-1">
                {[0, 1, 2].map((i) => (
                  <div
                    key={i}
                    className="h-1.5 w-1.5 rounded-full bg-muted-foreground animate-bounce"
                    style={{ animationDelay: `${i * 150}ms` }}
                  />
                ))}
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <ChatInput
        isLoading={isLoading}
        detectedLanguage={detectedLanguage}
        pendingImagePreview={pendingImagePreview}
        initialMessage={pendingMessage}
        onSend={handleSend}
        onImageSelect={(file, preview) => setPendingImage(file, preview)}
        onImageRemove={() => setPendingImage(null, null)}
        onForceLanguage={handleForceLanguage}
        onInitialMessageConsumed={() => setPendingMessage(null)}
      />
    </div>
  )
}
