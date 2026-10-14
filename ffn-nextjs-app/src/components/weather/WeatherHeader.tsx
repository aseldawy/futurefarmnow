'use client';

import React from 'react';
import { ChevronUp, ChevronDown, X, Cloud, RefreshCw } from 'lucide-react';
import { useFarmStore } from '@/lib/stores/farmStore';

export function WeatherHeader() {
  const selectedFarmId = useFarmStore((s) => s.selectedFarmId);
  const farms = useFarmStore((s) => s.farms);
  const weatherEndpoints = useFarmStore((s) => s.weatherEndpoints);
  const forecastData = useFarmStore((s) => s.forecastData);
  const isWeatherPopupExpanded = useFarmStore((s) => s.isWeatherPopupExpanded);
  const setIsWeatherPopupExpanded = useFarmStore((s) => s.setIsWeatherPopupExpanded);
  const setIsWeatherPopupOpen = useFarmStore((s) => s.setIsWeatherPopupOpen);
  const isLoadingWeather = useFarmStore((s) => s.isLoadingWeather);

  const farm = farms.find((f) => f.id === selectedFarmId);
  const endpoints = selectedFarmId ? weatherEndpoints[selectedFarmId] : null;

  const lastUpdated = forecastData
    ? Math.round((Date.now() - forecastData.cachedAt) / 60000)
    : null;

  return (
    <div id="weather-header" className="flex items-center justify-between">
      <div className="flex items-center gap-2 min-w-0">
        <Cloud className="h-4 w-4 text-primary flex-shrink-0" />
        <div className="min-w-0">
          <p className="text-sm font-semibold truncate">{farm?.name ?? 'Weather'}</p>
          {endpoints && (
            <p className="text-[10px] text-muted-foreground truncate">
              {endpoints.city}, {endpoints.state}
              {lastUpdated !== null && ` · ${lastUpdated}m ago`}
            </p>
          )}
        </div>
      </div>

      <div className="flex items-center gap-1 flex-shrink-0">
        {isLoadingWeather && (
          <RefreshCw className="h-3.5 w-3.5 text-muted-foreground animate-spin" />
        )}
        <button
          onClick={() => setIsWeatherPopupExpanded(!isWeatherPopupExpanded)}
          className="p-1 rounded hover:bg-muted transition-colors"
          title={isWeatherPopupExpanded ? 'Collapse' : 'Expand'}
        >
          {isWeatherPopupExpanded ? (
            <ChevronUp className="h-4 w-4" />
          ) : (
            <ChevronDown className="h-4 w-4" />
          )}
        </button>
        <button
          onClick={() => setIsWeatherPopupOpen(false)}
          className="p-1 rounded hover:bg-muted transition-colors"
          title="Close weather"
        >
          <X className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}
