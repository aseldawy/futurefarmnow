'use client'

import { Layers, TrendingUp, MapPin, AlertCircle, Droplets, Cloud } from 'lucide-react'
import { ChatContext } from '@/lib/types/chat'

interface Props {
  context: ChatContext
}

interface Pill {
  id: string
  icon: React.ReactNode
  label: string
  active: boolean
  title: string
}

export function ContextIndicator({ context }: Props) {
  const pills: Pill[] = [
    {
      id: 'ctx-polygon',
      icon: <MapPin className="h-3 w-3" />,
      label: 'Field drawn',
      active: context.hasPolygon,
      title: context.hasPolygon
        ? 'AI can query your drawn polygon for live data'
        : 'Draw a field on the map for field-specific answers',
    },
    {
      id: 'ctx-soil',
      icon: <Layers className="h-3 w-3" />,
      label: `Soil (${context.selectedSoilLayer})`,
      active: context.hasSoilResults,
      title: context.hasSoilResults
        ? `Soil ${context.selectedSoilLayer} analysis is available`
        : 'Run soil analysis to let AI explain your soil data',
    },
    {
      id: 'ctx-ndvi',
      icon: <TrendingUp className="h-3 w-3" />,
      label: 'NDVI',
      active: context.hasNDVIResults,
      title: context.hasNDVIResults
        ? 'Vegetation (NDVI) analysis is available'
        : 'Run NDVI analysis to let AI explain crop health',
    },
    {
      id: 'ctx-etmap',
      icon: <Droplets className="h-3 w-3" />,
      label: 'ET Map',
      active: context.hasETMapResult,
      title: context.hasETMapResult
        ? 'Evapotranspiration map is available for AI analysis'
        : 'Run ET Map analysis to let AI explain crop water use',
    },
    {
      id: 'ctx-cimis',
      icon: <Cloud className="h-3 w-3" />,
      label: 'CIMIS',
      active: !!context.mapCenter,
      title: context.mapCenter
        ? 'Real-time CIMIS weather & ET₀ data available — ask about irrigation, pests, or market prices'
        : 'Navigate to a location to unlock CIMIS weather data',
    },
  ]

  const anyActive = pills.some((p) => p.active)

  return (
    <div id="chat-context-indicator" className="px-3 py-2 border-b border-border/60 bg-muted/20">
      <div className="flex items-center gap-1.5 flex-wrap">
        <span className="text-[10px] text-muted-foreground font-medium uppercase tracking-wide">
          AI context
        </span>
        {!anyActive && (
          <span
            id="ctx-no-data"
            className="flex items-center gap-1 text-[10px] text-muted-foreground"
            title="Draw a field or run analysis to unlock field-specific answers"
          >
            <AlertCircle className="h-3 w-3" />
            No field selected
          </span>
        )}
        {pills.map((pill) => (
          <span
            key={pill.id}
            id={pill.id}
            title={pill.title}
            className={`flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-medium transition-colors ${
              pill.active
                ? 'bg-primary/15 text-primary border border-primary/30'
                : 'bg-muted text-muted-foreground border border-transparent opacity-50'
            }`}
          >
            {pill.icon}
            {pill.label}
          </span>
        ))}
      </div>
    </div>
  )
}
