import axios from 'axios';
import {
  WeatherPointsResponse,
  ForecastResponse,
  HourlyForecastResponse,
  CachedForecastData,
  CachedWeatherEndpoints,
  ScoreLevel,
  GoNoGoLevel,
  HourlyScoringResult,
  DayScoringResult,
  ForecastPeriod,
  HourlyPeriod,
} from '../types/weather';

const WEATHER_API_BASE = 'https://api.weather.gov';
const CACHE_TTL_MS = 60 * 60 * 1000; // 1 hour

const weatherAxios = axios.create({
  baseURL: WEATHER_API_BASE,
  headers: {
    'User-Agent': 'FutureFarmNow (futurefarmnow@ucr.edu)',
    Accept: 'application/geo+json',
  },
  timeout: 15000,
});

export async function fetchWeatherPoints(
  lat: number,
  lng: number,
  signal?: AbortSignal
): Promise<CachedWeatherEndpoints> {
  // Weather.gov wants max 4 decimal places
  const roundedLat = Math.round(lat * 10000) / 10000;
  const roundedLng = Math.round(lng * 10000) / 10000;

  const { data } = await weatherAxios.get<WeatherPointsResponse>(
    `/points/${roundedLat},${roundedLng}`,
    { signal }
  );

  return {
    forecastUrl: data.properties.forecast,
    forecastHourlyUrl: data.properties.forecastHourly,
    city: data.properties.relativeLocation.properties.city,
    state: data.properties.relativeLocation.properties.state,
  };
}

export async function fetchForecast(
  url: string,
  signal?: AbortSignal
): Promise<ForecastResponse> {
  const { data } = await weatherAxios.get<ForecastResponse>(url, { signal });
  return data;
}

export async function fetchHourlyForecast(
  url: string,
  signal?: AbortSignal
): Promise<HourlyForecastResponse> {
  const { data } = await weatherAxios.get<HourlyForecastResponse>(url, { signal });
  return data;
}

export function isWeatherStale(cachedData: CachedForecastData | null): boolean {
  if (!cachedData) return true;
  return Date.now() - cachedData.cachedAt > CACHE_TTL_MS;
}

/**
 * Parse wind speed string like "5 to 15 mph" or "10 mph" and return the max value.
 */
export function parseWindSpeed(windSpeed: string): number {
  const matches = windSpeed.match(/\d+/g);
  if (!matches) return 0;
  return Math.max(...matches.map(Number));
}

/**
 * Score an hour for spray suitability.
 * Good: precip <= 15%, wind <= 10 mph, RH 40-85%, temp 45-90F
 * Marginal: precip <= 30%, wind <= 15 mph, RH 30-90%, temp 40-95F
 * Bad: anything worse
 */
export function scoreSprayHour(period: HourlyPeriod): ScoreLevel {
  const precip = period.probabilityOfPrecipitation?.value ?? 0;
  const wind = parseWindSpeed(period.windSpeed);
  const rh = period.relativeHumidity?.value ?? 50;
  const temp = period.temperature;

  if (precip <= 15 && wind <= 10 && rh >= 40 && rh <= 85 && temp >= 45 && temp <= 90) {
    return 'good';
  }
  if (precip <= 30 && wind <= 15 && rh >= 30 && rh <= 90 && temp >= 40 && temp <= 95) {
    return 'marginal';
  }
  return 'bad';
}

/**
 * Score an hour for harvest suitability.
 * Good: precip <= 20%, wind <= 20 mph, RH <= 90%
 * Marginal: precip <= 40%, wind <= 25 mph, RH <= 95%
 * Bad: anything worse
 */
export function scoreHarvestHour(period: HourlyPeriod): ScoreLevel {
  const precip = period.probabilityOfPrecipitation?.value ?? 0;
  const wind = parseWindSpeed(period.windSpeed);
  const rh = period.relativeHumidity?.value ?? 50;

  if (precip <= 20 && wind <= 20 && rh <= 90) {
    return 'good';
  }
  if (precip <= 40 && wind <= 25 && rh <= 95) {
    return 'marginal';
  }
  return 'bad';
}

/**
 * Score a day for Go/No-Go based on precipitation probability.
 * @deprecated Prefer scoreDayGoNoGoFromHours — this ignores wind/RH/temp.
 */
export function scoreDayGoNoGo(precipPercent: number): GoNoGoLevel {
  if (precipPercent >= 75) return 'no-go';
  if (precipPercent >= 40) return 'caution';
  return 'go';
}

/**
 * Derive the day-level Go/No-Go from actual computed hourly windows.
 * Uses spray hours as the primary indicator (more restrictive than harvest).
 * - go:      ≥4 good spray hours, OR ≥6 good harvest hours
 * - caution: ≥1 good spray hour,  OR ≥2 good harvest hours
 * - no-go:   no usable operational windows
 */
export function scoreDayGoNoGoFromHours(
  sprayGoodHours: number,
  harvestGoodHours: number
): GoNoGoLevel {
  if (sprayGoodHours >= 4 || harvestGoodHours >= 6) return 'go';
  if (sprayGoodHours >= 1 || harvestGoodHours >= 2) return 'caution';
  return 'no-go';
}

/**
 * Compute a full day scoring result from a daily forecast period
 * and the matching hourly periods.
 */
export function computeDayScoring(
  dayPeriod: ForecastPeriod,
  nightPeriod: ForecastPeriod | null,
  hourlyPeriods: HourlyPeriod[]
): DayScoringResult {
  const precipPercent = dayPeriod.probabilityOfPrecipitation?.value ?? 0;
  const wind = parseWindSpeed(dayPeriod.windSpeed);

  const hourlyScores: HourlyScoringResult[] = hourlyPeriods.map((hp) => ({
    hour: hp.startTime,
    temperature: hp.temperature,
    windSpeed: parseWindSpeed(hp.windSpeed),
    precipPercent: hp.probabilityOfPrecipitation?.value ?? 0,
    humidity: hp.relativeHumidity?.value ?? 0,
    sprayScore: scoreSprayHour(hp),
    harvestScore: scoreHarvestHour(hp),
  }));

  const sprayGoodHours = hourlyScores.filter((h) => h.sprayScore === 'good').length;
  const sprayMarginalHours = hourlyScores.filter((h) => h.sprayScore === 'marginal').length;
  const harvestGoodHours = hourlyScores.filter((h) => h.harvestScore === 'good').length;
  const harvestMarginalHours = hourlyScores.filter((h) => h.harvestScore === 'marginal').length;

  const startDate = new Date(dayPeriod.startTime);
  const dayName = startDate.toLocaleDateString('en-US', { weekday: 'short' });
  const date = startDate.toISOString().split('T')[0];

  let temperatureTrend = '';
  if (dayPeriod.temperatureTrend) {
    temperatureTrend = dayPeriod.temperatureTrend;
  } else if (nightPeriod) {
    temperatureTrend = `High ${dayPeriod.temperature}°F, Low ${nightPeriod.temperature}°F`;
  }

  return {
    dayName,
    date,
    highTemp: dayPeriod.temperature,
    lowTemp: nightPeriod?.temperature ?? null,
    temperatureTrend,
    windSummary: `${dayPeriod.windDirection} ${dayPeriod.windSpeed}`,
    precipPercent,
    goNoGo: scoreDayGoNoGoFromHours(sprayGoodHours, harvestGoodHours),
    icon: dayPeriod.icon,
    shortForecast: dayPeriod.shortForecast,
    detailedForecast: dayPeriod.detailedForecast,
    sprayGoodHours,
    sprayMarginalHours,
    harvestGoodHours,
    harvestMarginalHours,
    hourlyScores,
  };
}

/**
 * Process full forecast data into an array of day scoring results.
 */
export function processForecastData(
  forecast: ForecastResponse,
  hourly: HourlyForecastResponse
): DayScoringResult[] {
  const periods = forecast.properties.periods;
  const hourlyPeriods = hourly.properties.periods;

  const results: DayScoringResult[] = [];

  for (let i = 0; i < periods.length; i++) {
    const period = periods[i];
    if (!period.isDaytime) continue;

    const nightPeriod = periods[i + 1]?.isDaytime === false ? periods[i + 1] : null;
    const dayDate = new Date(period.startTime).toISOString().split('T')[0];

    // Find hourly periods matching this day
    const matchingHourly = hourlyPeriods.filter((hp) => {
      const hpDate = new Date(hp.startTime).toISOString().split('T')[0];
      return hpDate === dayDate;
    });

    results.push(computeDayScoring(period, nightPeriod, matchingHourly));
  }

  return results;
}
