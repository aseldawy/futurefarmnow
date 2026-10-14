import { GeoJSONGeometry, BoundingBox, ETMapStatistics } from './api'

export interface ToolCallRecord {
  name: string
  args: Record<string, unknown>
  result: Record<string, unknown>
  isExpanded: boolean
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  imagePreviewUrl?: string
  toolCalls?: ToolCallRecord[]
  sources?: string[]
  detectedLanguage?: 'en' | 'es'
  timestamp: number
  error?: string
}

export interface SoilResultSnapshot {
  layer: string
  depth: string
  mean: number | null
  min: number | null
  max: number | null
  median: number | null
  stddev: number | null
  count: number | null
}

export interface NdviResultSnapshot {
  fromDate: string
  toDate: string
  averageNdvi: number | null
  maxNdvi: number | null
  minNdvi: number | null
  dataPoints: number
}

export interface ChatContext {
  hasPolygon: boolean
  polygonGeojson: GeoJSONGeometry | null
  selectedSoilLayer: string
  selectedSoilDepth: string
  selectedDateRange: { from: string; to: string }
  activeTab: string
  hasSoilResults: boolean
  hasNDVIResults: boolean
  soilAnalysisResult: SoilResultSnapshot | null
  ndviAnalysisResult: NdviResultSnapshot | null
  farmCount: number
  currentZoom: number
  mapBounds: BoundingBox | null
  mapCenter: [number, number] | null
  selectedDataset: string | null
  hasETMapResult: boolean
  etmapRequestId: string | null
  etmapAnalysisResult: ETMapStatistics | null
  weatherContext: WeatherContext | null
}

export interface WeatherDaySummary {
  dayName: string
  date: string
  highTemp: number
  precipPercent: number
  goNoGo: 'go' | 'caution' | 'no-go'
  sprayGoodHours: number
  sprayMarginalHours: number
  harvestGoodHours: number
  harvestMarginalHours: number
  shortForecast: string
  windSummary: string
}

export interface WeatherContext {
  farmName: string
  location: string
  days: WeatherDaySummary[]
}

export interface ChatAPIResponse {
  reply: string
  tool_calls: Array<{
    name: string
    args: Record<string, unknown>
    result: Record<string, unknown>
  }>
  sources: string[]
  detected_language: 'en' | 'es'
  error?: string
  message?: string
}
