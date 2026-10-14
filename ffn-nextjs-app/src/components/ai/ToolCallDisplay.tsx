'use client'

import { ChevronDown, ChevronRight, Layers, TrendingUp, MapPin, Cloud, ShieldAlert, DollarSign, BookOpen } from 'lucide-react'
import { ToolCallRecord } from '@/lib/types/chat'

const TOOL_META: Record<string, { label: string; icon: React.ReactNode }> = {
  get_soil_stats: {
    label: 'Soil Analysis',
    icon: <Layers className="h-3.5 w-3.5" />,
  },
  get_ndvi_timeseries: {
    label: 'Vegetation (NDVI)',
    icon: <TrendingUp className="h-3.5 w-3.5" />,
  },
  get_soil_sample_points: {
    label: 'Sample Points',
    icon: <MapPin className="h-3.5 w-3.5" />,
  },
  get_cimis_weather: {
    label: 'CIMIS Weather',
    icon: <Cloud className="h-3.5 w-3.5" />,
  },
  get_pest_disease_risk: {
    label: 'Pest & Disease Risk',
    icon: <ShieldAlert className="h-3.5 w-3.5" />,
  },
  get_market_prices: {
    label: 'Market Prices',
    icon: <DollarSign className="h-3.5 w-3.5" />,
  },
  get_uc_ipm_guidance: {
    label: 'UC IPM Guidance',
    icon: <BookOpen className="h-3.5 w-3.5" />,
  },
}

interface Props {
  id: 'tool-call-display'
  messageId: string
  toolCalls: ToolCallRecord[]
  onToggle: (toolIndex: number) => void
}

export function ToolCallDisplay({ messageId, toolCalls, onToggle }: Omit<Props, 'id'>) {
  if (!toolCalls || toolCalls.length === 0) return null

  return (
    <div id={`tool-calls-${messageId}`} className="mt-2 space-y-1.5">
      {toolCalls.map((tc, i) => {
        const meta = TOOL_META[tc.name] ?? { label: tc.name, icon: <Layers className="h-3.5 w-3.5" /> }
        const hasError = 'error' in tc.result

        return (
          <div
            key={i}
            id={`tool-call-${messageId}-${i}`}
            className="rounded-md border border-border/60 bg-muted/30 text-xs overflow-hidden"
          >
            <button
              onClick={() => onToggle(i)}
              className="flex w-full items-center gap-2 px-3 py-2 text-left hover:bg-muted/50 transition-colors"
            >
              <span className={hasError ? 'text-destructive' : 'text-primary'}>
                {meta.icon}
              </span>
              <span className="font-medium text-foreground/80 flex-1">
                {meta.label}
                {tc.args && Object.keys(tc.args).length > 0 && (
                  <span className="ml-1.5 text-muted-foreground font-normal">
                    — {formatArgs(tc.name, tc.args)}
                  </span>
                )}
              </span>
              {tc.isExpanded
                ? <ChevronDown className="h-3 w-3 text-muted-foreground shrink-0" />
                : <ChevronRight className="h-3 w-3 text-muted-foreground shrink-0" />
              }
            </button>

            {tc.isExpanded && (
              <div id={`tool-result-${messageId}-${i}`} className="border-t border-border/40 px-3 py-2 space-y-1">
                {hasError
                  ? <p className="text-destructive">{String((tc.result as Record<string,unknown>).message || 'Error retrieving data')}</p>
                  : renderResult(tc.name, tc.result)
                }
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

function formatArgs(toolName: string, args: Record<string, unknown>): string {
  if (toolName === 'get_soil_stats') {
    return `${args.layer ?? ''} at ${args.soildepth ?? ''}cm`
  }
  if (toolName === 'get_ndvi_timeseries') {
    return `${args.from_date ?? ''} – ${args.to_date ?? ''}`
  }
  if (toolName === 'get_soil_sample_points') {
    return `${args.num_points ?? ''} points`
  }
  if (toolName === 'get_cimis_weather') {
    return `${args.start_date ?? ''} – ${args.end_date ?? ''}`
  }
  if (toolName === 'get_pest_disease_risk') {
    return args.crop ? `${args.crop}` : 'all crops'
  }
  if (toolName === 'get_market_prices') {
    return `${args.commodity ?? ''}`
  }
  if (toolName === 'get_uc_ipm_guidance') {
    return args.pest ? `${args.crop} / ${args.pest}` : `${args.crop ?? ''}`
  }
  return ''
}

function renderResult(toolName: string, result: Record<string, unknown>): React.ReactNode {
  if (toolName === 'get_soil_stats') {
    const rows = [
      { label: 'Mean', value: result.mean },
      { label: 'Median', value: result.median },
      { label: 'Min', value: result.min },
      { label: 'Max', value: result.max },
      { label: 'Std Dev', value: result.stddev },
    ]
    return (
      <dl id="soil-stats-result" className="grid grid-cols-2 gap-x-4 gap-y-0.5">
        {rows.map(({ label, value }) =>
          value != null ? (
            <div key={label} className="flex justify-between col-span-1">
              <dt className="text-muted-foreground">{label}</dt>
              <dd className="font-medium tabular-nums">{Number(value).toFixed(3)}</dd>
            </div>
          ) : null
        )}
      </dl>
    )
  }

  if (toolName === 'get_ndvi_timeseries') {
    return (
      <dl id="ndvi-result" className="grid grid-cols-2 gap-x-4 gap-y-0.5">
        <div className="flex justify-between col-span-2">
          <dt className="text-muted-foreground">Data points</dt>
          <dd className="font-medium">{String(result.data_points ?? '—')}</dd>
        </div>
        {result.average_ndvi != null && (
          <div className="flex justify-between col-span-1">
            <dt className="text-muted-foreground">Average</dt>
            <dd className="font-medium tabular-nums">{Number(result.average_ndvi).toFixed(3)}</dd>
          </div>
        )}
        {result.max_ndvi != null && (
          <div className="flex justify-between col-span-1">
            <dt className="text-muted-foreground">Peak</dt>
            <dd className="font-medium tabular-nums">{Number(result.max_ndvi).toFixed(3)}</dd>
          </div>
        )}
      </dl>
    )
  }

  if (toolName === 'get_soil_sample_points') {
    const locations = result.sample_locations as Array<{id: number, lat: number, lon: number}> | undefined
    return (
      <div id="sample-points-result">
        <p className="text-muted-foreground mb-1">{result.num_points as number} sampling locations generated</p>
        {locations && locations.slice(0, 5).map((loc) => (
          <div key={loc.id} className="flex justify-between tabular-nums">
            <span className="text-muted-foreground">#{loc.id}</span>
            <span>{Number(loc.lat).toFixed(5)}, {Number(loc.lon).toFixed(5)}</span>
          </div>
        ))}
        {locations && locations.length > 5 && (
          <p className="text-muted-foreground mt-0.5">…and {locations.length - 5} more</p>
        )}
      </div>
    )
  }

  if (toolName === 'get_cimis_weather') {
    const daily = result.daily as Array<Record<string, unknown>> | undefined
    return (
      <div id="cimis-weather-result" className="space-y-1.5">
        <p className="text-[10px] text-muted-foreground">{String(result.station ?? '')} · {String(result.station_dist_km ?? '')} km away</p>
        <dl className="grid grid-cols-2 gap-x-4 gap-y-0.5">
          {result.total_eto_in != null && (
            <div className="flex justify-between col-span-1">
              <dt className="text-muted-foreground">Total ETo</dt>
              <dd className="font-medium tabular-nums">{Number(result.total_eto_in).toFixed(2)} in</dd>
            </div>
          )}
          {result.total_precip_in != null && (
            <div className="flex justify-between col-span-1">
              <dt className="text-muted-foreground">Total Precip</dt>
              <dd className="font-medium tabular-nums">{Number(result.total_precip_in).toFixed(2)} in</dd>
            </div>
          )}
          {result.avg_tmax_f != null && (
            <div className="flex justify-between col-span-1">
              <dt className="text-muted-foreground">Avg High</dt>
              <dd className="font-medium tabular-nums">{Number(result.avg_tmax_f).toFixed(1)}°F</dd>
            </div>
          )}
          {result.avg_tmin_f != null && (
            <div className="flex justify-between col-span-1">
              <dt className="text-muted-foreground">Avg Low</dt>
              <dd className="font-medium tabular-nums">{Number(result.avg_tmin_f).toFixed(1)}°F</dd>
            </div>
          )}
        </dl>
        {daily && daily.length > 0 && (
          <div className="mt-1 space-y-0.5 max-h-32 overflow-y-auto">
            {daily.slice(-7).map((d, i) => (
              <div key={i} className="flex justify-between text-[10px]">
                <span className="text-muted-foreground">{String(d.date ?? '')}</span>
                <span className="tabular-nums">ETo {d.eto_in != null ? Number(d.eto_in).toFixed(2) : '—'}" · {d.tavg_f != null ? `${Number(d.tavg_f).toFixed(0)}°F` : '—'}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    )
  }

  if (toolName === 'get_pest_disease_risk') {
    const pests = result.pest_risks as Record<string, Record<string, unknown>> | undefined
    const diseases = result.disease_risks as Record<string, Record<string, unknown>> | undefined
    const riskColor = (level: string) =>
      level === 'high' ? 'text-destructive' : level === 'moderate' ? 'text-amber-600 dark:text-amber-400' : 'text-emerald-600 dark:text-emerald-400'
    return (
      <div id="pest-risk-result" className="space-y-2">
        <p className="text-[10px] text-muted-foreground">{String(result.station ?? '')} · {String(result.days_analyzed ?? '')} days · {result.crop_filter ? `${result.crop_filter}` : 'all crops'}</p>
        {pests && Object.keys(pests).length > 0 && (
          <div>
            <p className="text-[10px] font-medium mb-0.5">Insect Pests (GDD)</p>
            <div className="space-y-0.5">
              {Object.entries(pests).map(([key, p]) => (
                <div key={key} className="flex justify-between text-[10px]">
                  <span className="text-muted-foreground">{String(p.label ?? key)}</span>
                  <span className={`font-medium ${riskColor(String(p.risk_level))}`}>
                    {String(p.risk_level).toUpperCase()} · {Number(p.gdd_accumulated).toFixed(0)} GDD
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
        {diseases && Object.keys(diseases).length > 0 && (
          <div>
            <p className="text-[10px] font-medium mb-0.5">Disease Risk</p>
            <div className="space-y-0.5">
              {Object.entries(diseases).map(([key, d]) => (
                <div key={key} className="flex justify-between text-[10px]">
                  <span className="text-muted-foreground">{String(d.label ?? key)}</span>
                  <span className={`font-medium ${riskColor(String(d.risk_level))}`}>
                    {String(d.risk_level).toUpperCase()} · {Number(d.favorable_days)} days
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    )
  }

  if (toolName === 'get_market_prices') {
    const prices = result.prices as Array<Record<string, unknown>> | undefined
    return (
      <div id="market-prices-result" className="space-y-1">
        <p className="text-[10px] text-muted-foreground">{String(result.report_label ?? '')} · {String(result.report_date ?? 'latest')}</p>
        {prices && prices.length > 0 ? (
          <div className="space-y-0.5">
            {prices.slice(0, 6).map((p, i) => (
              <div key={i} className="flex justify-between text-[10px]">
                <span className="text-muted-foreground truncate max-w-[140px]">
                  {String(p.commodity ?? '')}
                  {p.variety ? ` (${p.variety})` : ''}
                  {p.organic ? ' 🌿' : ''}
                </span>
                <span className="font-medium tabular-nums shrink-0 ml-2">
                  ${Number(p.low_price).toFixed(2)}–${Number(p.high_price).toFixed(2)}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-[10px] text-muted-foreground">{String(result.message ?? 'No price data available.')}</p>
        )}
      </div>
    )
  }

  if (toolName === 'get_uc_ipm_guidance') {
    return (
      <div id="uc-ipm-result" className="space-y-1.5">
        <p className="text-[10px] font-medium">{String(result.crop ?? '')} {result.pest ? `— ${result.pest}` : ''}</p>
        {!!result.treatment_threshold && (
          <div className="rounded bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-900/50 px-2 py-1">
            <p className="text-[10px] text-amber-800 dark:text-amber-300 font-medium">Treatment threshold</p>
            <p className="text-[10px] text-amber-700 dark:text-amber-400">{String(result.treatment_threshold)}</p>
          </div>
        )}
        {!!result.monitoring_tips && (
          <p className="text-[10px] text-muted-foreground">{String(result.monitoring_tips)}</p>
        )}
        {!!result.source_url && (
          <a
            href={String(result.source_url)}
            target="_blank"
            rel="noopener noreferrer"
            className="text-[10px] text-primary underline underline-offset-2"
          >
            UC IPM Guidelines ↗
          </a>
        )}
      </div>
    )
  }

  // Generic fallback
  return (
    <pre id="tool-raw-result" className="text-muted-foreground text-[10px] whitespace-pre-wrap break-all">
      {JSON.stringify(result, null, 2)}
    </pre>
  )
}
