'use client';

import React, { useMemo } from 'react';
import { CloudOff, RefreshCw } from 'lucide-react';
import { useFarmStore } from '@/lib/stores/farmStore';
import { processForecastData } from '@/lib/utils/weather';
import { WeatherHeader } from './WeatherHeader';
import { DayForecastStrip } from './DayForecastStrip';
import { GoNoGoBadge } from './GoNoGoBadge';

export function WeatherOverlay() {
  const selectedFarmId = useFarmStore((s) => s.selectedFarmId);
  const forecastData = useFarmStore((s) => s.forecastData);
  const isLoadingWeather = useFarmStore((s) => s.isLoadingWeather);
  const weatherError = useFarmStore((s) => s.weatherError);
  const isWeatherPopupOpen = useFarmStore((s) => s.isWeatherPopupOpen);
  const isWeatherPopupExpanded = useFarmStore((s) => s.isWeatherPopupExpanded);
  const selectFarm = useFarmStore((s) => s.selectFarm);

  const dayScores = useMemo(() => {
    if (!forecastData) return [];
    return processForecastData(forecastData.forecast, forecastData.hourly);
  }, [forecastData]);

  if (!selectedFarmId || !isWeatherPopupOpen) return null;

  const todayScore = dayScores[0];

  return (
    <div
      id="weather-overlay"
      className="absolute top-3 right-14 z-[800] w-72 max-w-[calc(100vw-6rem)] weather-slide-in"
    >
      <div className="rounded-xl border border-border bg-card/95 backdrop-blur-sm shadow-lg overflow-hidden">
        <div className="px-3 pt-2.5 pb-1.5">
          <WeatherHeader />
        </div>

        {/* Loading state */}
        {isLoadingWeather && !forecastData && (
          <div id="weather-loading" className="px-3 pb-2.5 space-y-1.5">
            <div className="h-3 bg-muted animate-pulse rounded w-3/4" />
            <div className="h-3 bg-muted animate-pulse rounded w-1/2" />
          </div>
        )}

        {/* Error state */}
        {weatherError && !isLoadingWeather && (
          <div id="weather-error" className="px-3 pb-2.5">
            <div className="flex items-center gap-2 p-1.5 rounded-lg bg-destructive/10 text-destructive text-[11px]">
              <CloudOff className="h-3.5 w-3.5 flex-shrink-0" />
              <span className="flex-1">Unavailable</span>
              <button
                onClick={() => {
                  const id = selectedFarmId;
                  selectFarm(null);
                  setTimeout(() => selectFarm(id), 100);
                }}
                className="flex items-center gap-1 text-primary hover:underline"
              >
                <RefreshCw className="h-3 w-3" /> Retry
              </button>
            </div>
          </div>
        )}

        {/* Today's summary */}
        {todayScore && !isLoadingWeather && (
          <div id="weather-today-summary" className="px-3 pb-2">
            <div className="flex items-center justify-between">
              <div>
                <span className="text-base font-bold">{todayScore.highTemp}°F</span>
                {todayScore.lowTemp !== null && (
                  <span className="text-xs text-muted-foreground ml-1">/ {todayScore.lowTemp}°</span>
                )}
                <p className="text-[10px] text-muted-foreground leading-tight">
                  {todayScore.shortForecast}
                </p>
              </div>
              <div className="text-right">
                <GoNoGoBadge level={todayScore.goNoGo} size="md" />
                <p className="text-[9px] text-muted-foreground mt-0.5">
                  {todayScore.sprayGoodHours}h spray · {todayScore.harvestGoodHours}h harvest
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Expanded content */}
        {isWeatherPopupExpanded && dayScores.length > 0 && !isLoadingWeather && (
          <div id="weather-expanded" className="px-3 pb-2.5 border-t border-border pt-2 weather-slide-in">
            <DayForecastStrip days={dayScores} />
          </div>
        )}
      </div>
    </div>
  );
}
