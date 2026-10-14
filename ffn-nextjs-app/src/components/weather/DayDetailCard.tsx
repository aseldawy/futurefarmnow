'use client';

import React, { useState } from 'react';
import { ChevronDown, ChevronUp, Wind, Droplets } from 'lucide-react';
import { DayScoringResult } from '@/lib/types/weather';
import { GoNoGoBadge } from './GoNoGoBadge';

interface DayDetailCardProps {
  day: DayScoringResult;
}

export function DayDetailCard({ day }: DayDetailCardProps) {
  const [showDetails, setShowDetails] = useState(false);

  return (
    <div id={`day-detail-${day.date}`} className="rounded-lg border border-border bg-card p-2.5 space-y-1.5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3 text-xs text-muted-foreground">
          <span className="flex items-center gap-1">
            <Wind className="h-3 w-3" />
            {day.windSummary}
          </span>
          <span className="flex items-center gap-1">
            <Droplets className="h-3 w-3" />
            {day.precipPercent}%
          </span>
        </div>
        <GoNoGoBadge level={day.goNoGo} />
      </div>

      <div className="flex gap-4 text-[10px]">
        <span className="text-green-700 dark:text-green-400">
          Spray {day.sprayGoodHours}h good
        </span>
        <span className="text-blue-700 dark:text-blue-400">
          Harvest {day.harvestGoodHours}h good
        </span>
      </div>

      {day.detailedForecast && (
        <>
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="flex items-center gap-1 text-[10px] text-primary hover:underline"
          >
            {showDetails ? (
              <><ChevronUp className="h-3 w-3" /> Less</>
            ) : (
              <><ChevronDown className="h-3 w-3" /> Details</>
            )}
          </button>
          {showDetails && (
            <p className="text-[11px] text-muted-foreground leading-relaxed">
              {day.detailedForecast}
            </p>
          )}
        </>
      )}
    </div>
  );
}
