'use client';

import { useEffect, useRef } from 'react';
import { useFarmStore } from '@/lib/stores/farmStore';
import {
  fetchWeatherPoints,
  fetchForecast,
  fetchHourlyForecast,
  isWeatherStale,
} from '@/lib/utils/weather';
import toast from 'react-hot-toast';

export function useWeather() {
  const abortRef = useRef<AbortController | null>(null);

  const selectedFarmId = useFarmStore((s) => s.selectedFarmId);
  const farms = useFarmStore((s) => s.farms);
  const weatherEndpoints = useFarmStore((s) => s.weatherEndpoints);
  const forecastData = useFarmStore((s) => s.forecastData);

  const setWeatherEndpoints = useFarmStore((s) => s.setWeatherEndpoints);
  const setForecastData = useFarmStore((s) => s.setForecastData);
  const setIsLoadingWeather = useFarmStore((s) => s.setIsLoadingWeather);
  const setWeatherError = useFarmStore((s) => s.setWeatherError);

  useEffect(() => {
    if (!selectedFarmId) return;

    const farm = farms.find((f) => f.id === selectedFarmId);
    if (!farm) return;

    // Check if we already have fresh data for this farm
    if (forecastData && !isWeatherStale(forecastData)) return;

    // Cancel any in-flight requests
    if (abortRef.current) {
      abortRef.current.abort();
    }
    const controller = new AbortController();
    abortRef.current = controller;

    const fetchWeather = async () => {
      setIsLoadingWeather(true);
      setWeatherError(null);

      try {
        // Step 1: Get endpoints (use cache if available)
        let endpoints = weatherEndpoints[selectedFarmId];
        if (!endpoints) {
          endpoints = await fetchWeatherPoints(
            farm.center[0],
            farm.center[1],
            controller.signal
          );
          setWeatherEndpoints(selectedFarmId, endpoints);
        }

        // Step 2: Parallel fetch forecast + hourly
        const [forecast, hourly] = await Promise.all([
          fetchForecast(endpoints.forecastUrl, controller.signal),
          fetchHourlyForecast(endpoints.forecastHourlyUrl, controller.signal),
        ]);

        // Step 3: Store the data
        setForecastData({
          forecast,
          hourly,
          generatedAt: forecast.properties.generatedAt,
          cachedAt: Date.now(),
        });
      } catch (err: unknown) {
        if (err instanceof Error && err.name === 'AbortError') return;
        if (err instanceof Error && err.name === 'CanceledError') return;

        const message =
          err instanceof Error ? err.message : 'Failed to fetch weather data';
        setWeatherError(message);
        toast.error('Weather data unavailable. Try again later.');
      } finally {
        setIsLoadingWeather(false);
      }
    };

    fetchWeather();

    return () => {
      controller.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedFarmId]);
}
