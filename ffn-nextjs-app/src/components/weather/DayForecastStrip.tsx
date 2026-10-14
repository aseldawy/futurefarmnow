'use client';

import React, { useState } from 'react';
import { Sun, Cloud, CloudRain, CloudSnow, CloudDrizzle, CloudLightning, CloudFog, Wind } from 'lucide-react';
import { DayScoringResult, GoNoGoLevel } from '@/lib/types/weather';
import { GoNoGoBadge } from './GoNoGoBadge';
import { DayDetailCard } from './DayDetailCard';
import { HourlyTimeline } from './HourlyTimeline';

interface DayForecastStripProps {
  days: DayScoringResult[];
}

const GO_DOT: Record<GoNoGoLevel, string> = {
  go: 'bg-green-500',
  caution: 'bg-yellow-400',
  'no-go': 'bg-red-500',
};

function WeatherIcon({ forecast, className = 'h-4 w-4' }: { forecast: string; className?: string }) {
  const f = forecast.toLowerCase();
  if (f.includes('thunder') || f.includes('storm')) return <CloudLightning className={className} />;
  if (f.includes('snow') || f.includes('sleet') || f.includes('ice')) return <CloudSnow className={className} />;
  if (f.includes('rain') || f.includes('shower')) return <CloudRain className={className} />;
  if (f.includes('drizzle')) return <CloudDrizzle className={className} />;
  if (f.includes('fog') || f.includes('haze') || f.includes('mist')) return <CloudFog className={className} />;
  if (f.includes('wind')) return <Wind className={className} />;
  if (f.includes('cloud') || f.includes('overcast')) return <Cloud className={className} />;
  if (f.includes('partly')) return <Cloud className={className} />;
  return <Sun className={className} />;
}

export { WeatherIcon };

export function DayForecastStrip({ days }: DayForecastStripProps) {
  const [selectedDayIdx, setSelectedDayIdx] = useState<number | null>(null);

  return (
    <div id="day-forecast-strip" className="space-y-2">
      {/* Compact day strip */}
      <div className="flex gap-1 overflow-x-auto pb-0.5 scrollbar-thin">
        {days.map((day, idx) => {
          const isSelected = selectedDayIdx === idx;
          const isToday = idx === 0;

          return (
            <button
              key={day.date}
              onClick={() => setSelectedDayIdx(isSelected ? null : idx)}
              className={`flex-shrink-0 flex flex-col items-center gap-0.5 px-2 py-1.5 rounded-lg border transition-colors min-w-[52px] ${
                isSelected
                  ? 'border-primary bg-primary/5'
                  : 'border-transparent hover:border-border hover:bg-muted/50'
              }`}
            >
              <span className="text-[9px] font-medium text-muted-foreground">
                {isToday ? 'Today' : day.dayName}
              </span>
              <WeatherIcon forecast={day.shortForecast} className="h-4 w-4" />
              <span className="text-[11px] font-semibold">{day.highTemp}°</span>
              <span className={`w-1.5 h-1.5 rounded-full ${GO_DOT[day.goNoGo]}`} />
            </button>
          );
        })}
      </div>

      {/* Expanded day detail */}
      {selectedDayIdx !== null && days[selectedDayIdx] && (
        <div className="space-y-2 weather-slide-in">
          <DayDetailCard day={days[selectedDayIdx]} />
          {days[selectedDayIdx].hourlyScores.length > 0 && (
            <HourlyTimeline scores={days[selectedDayIdx].hourlyScores} />
          )}
        </div>
      )}
    </div>
  );
}
