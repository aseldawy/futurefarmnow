'use client';

import React, { useMemo, useState } from 'react';
import { HourlyScoringResult, ScoreLevel } from '@/lib/types/weather';

interface HourlyTimelineProps {
  scores: HourlyScoringResult[];
}

const SCORE_COLORS: Record<ScoreLevel, string> = {
  good: 'bg-green-500 dark:bg-green-600',
  marginal: 'bg-yellow-400 dark:bg-yellow-500',
  bad: 'bg-red-400 dark:bg-red-500',
};

interface SprayWindow {
  start: string;
  end: string;
  hours: number;
}

function formatHour(isoTime: string): string {
  const d = new Date(isoTime);
  const h = d.getHours();
  if (h === 0) return '12a';
  if (h === 12) return '12p';
  return h < 12 ? `${h}a` : `${h - 12}p`;
}

function findGoodWindows(scores: HourlyScoringResult[], key: 'sprayScore' | 'harvestScore'): SprayWindow[] {
  const windows: SprayWindow[] = [];
  let start: number | null = null;

  for (let i = 0; i <= scores.length; i++) {
    const isGood = i < scores.length && scores[i][key] === 'good';
    if (isGood && start === null) {
      start = i;
    } else if (!isGood && start !== null) {
      windows.push({
        start: formatHour(scores[start].hour),
        end: formatHour(scores[i - 1].hour),
        hours: i - start,
      });
      start = null;
    }
  }
  return windows;
}

export function HourlyTimeline({ scores }: HourlyTimelineProps) {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  const sprayWindows = useMemo(() => findGoodWindows(scores, 'sprayScore'), [scores]);
  const harvestWindows = useMemo(() => findGoodWindows(scores, 'harvestScore'), [scores]);

  return (
    <div id="hourly-timeline" className="space-y-2">
      {/* Window summaries */}
      {sprayWindows.length > 0 && (
        <p className="text-[10px] text-green-700 dark:text-green-400">
          Spray windows:{' '}
          {sprayWindows.map((w, i) => (
            <span key={i}>
              {i > 0 && ', '}
              {w.start}-{w.end} ({w.hours}h)
            </span>
          ))}
        </p>
      )}
      {harvestWindows.length > 0 && (
        <p className="text-[10px] text-blue-700 dark:text-blue-400">
          Harvest windows:{' '}
          {harvestWindows.map((w, i) => (
            <span key={i}>
              {i > 0 && ', '}
              {w.start}-{w.end} ({w.hours}h)
            </span>
          ))}
        </p>
      )}

      {/* Hourly bars */}
      <div className="relative">
        {/* Hour labels */}
        <div className="flex gap-px mb-0.5">
          {scores.map((s, i) => (
            <div
              key={`label-${i}`}
              className="flex-1 text-center text-[8px] text-muted-foreground"
            >
              {i % 3 === 0 ? formatHour(s.hour) : ''}
            </div>
          ))}
        </div>

        {/* Spray row */}
        <div className="flex items-center gap-0.5 mb-0.5">
          <span className="text-[8px] text-muted-foreground w-8 flex-shrink-0">Spray</span>
          <div className="flex gap-px flex-1">
            {scores.map((s, i) => (
              <div
                key={`spray-${i}`}
                className={`flex-1 h-3 rounded-sm cursor-pointer transition-all ${SCORE_COLORS[s.sprayScore]} ${
                  hoveredIdx === i ? 'ring-1 ring-foreground scale-y-125' : ''
                }`}
                onMouseEnter={() => setHoveredIdx(i)}
                onMouseLeave={() => setHoveredIdx(null)}
              />
            ))}
          </div>
        </div>

        {/* Harvest row */}
        <div className="flex items-center gap-0.5">
          <span className="text-[8px] text-muted-foreground w-8 flex-shrink-0">Harv</span>
          <div className="flex gap-px flex-1">
            {scores.map((s, i) => (
              <div
                key={`harvest-${i}`}
                className={`flex-1 h-3 rounded-sm cursor-pointer transition-all ${SCORE_COLORS[s.harvestScore]} ${
                  hoveredIdx === i ? 'ring-1 ring-foreground scale-y-125' : ''
                }`}
                onMouseEnter={() => setHoveredIdx(i)}
                onMouseLeave={() => setHoveredIdx(null)}
              />
            ))}
          </div>
        </div>

        {/* Tooltip */}
        {hoveredIdx !== null && scores[hoveredIdx] && (
          <div className="absolute bottom-full mb-1 left-1/2 -translate-x-1/2 bg-popover text-popover-foreground border border-border rounded-md shadow-md p-2 text-[10px] whitespace-nowrap z-10 pointer-events-none weather-fade-in">
            <p className="font-medium">{formatHour(scores[hoveredIdx].hour)}</p>
            <p>Temp: {scores[hoveredIdx].temperature}°F</p>
            <p>Wind: {scores[hoveredIdx].windSpeed} mph</p>
            <p>Precip: {scores[hoveredIdx].precipPercent}%</p>
            <p>RH: {scores[hoveredIdx].humidity}%</p>
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="flex gap-3 text-[9px] text-muted-foreground">
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-sm bg-green-500" /> Good
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-sm bg-yellow-400" /> Marginal
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-sm bg-red-400" /> Bad
        </span>
      </div>
    </div>
  );
}
