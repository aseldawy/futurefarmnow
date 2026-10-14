export interface WeatherPointsResponse {
  properties: {
    forecast: string;
    forecastHourly: string;
    relativeLocation: {
      properties: {
        city: string;
        state: string;
      };
    };
  };
}

export interface ForecastPeriod {
  number: number;
  name: string;
  startTime: string;
  endTime: string;
  isDaytime: boolean;
  temperature: number;
  temperatureUnit: string;
  temperatureTrend: string | null;
  probabilityOfPrecipitation: { value: number | null };
  relativeHumidity: { value: number | null };
  windSpeed: string;
  windDirection: string;
  icon: string;
  shortForecast: string;
  detailedForecast: string;
}

export interface HourlyPeriod {
  number: number;
  startTime: string;
  endTime: string;
  isDaytime: boolean;
  temperature: number;
  temperatureUnit: string;
  probabilityOfPrecipitation: { value: number | null };
  relativeHumidity: { value: number | null };
  windSpeed: string;
  windDirection: string;
  icon: string;
  shortForecast: string;
}

export interface ForecastResponse {
  properties: {
    generatedAt: string;
    updateTime: string;
    periods: ForecastPeriod[];
  };
}

export interface HourlyForecastResponse {
  properties: {
    generatedAt: string;
    updateTime: string;
    periods: HourlyPeriod[];
  };
}

export interface CachedWeatherEndpoints {
  forecastUrl: string;
  forecastHourlyUrl: string;
  city: string;
  state: string;
}

export interface CachedForecastData {
  forecast: ForecastResponse;
  hourly: HourlyForecastResponse;
  generatedAt: string;
  cachedAt: number; // Date.now()
}

export type ScoreLevel = 'good' | 'marginal' | 'bad';
export type GoNoGoLevel = 'go' | 'caution' | 'no-go';

export interface HourlyScoringResult {
  hour: string; // ISO time
  temperature: number;
  windSpeed: number;
  precipPercent: number;
  humidity: number;
  sprayScore: ScoreLevel;
  harvestScore: ScoreLevel;
}

export interface DayScoringResult {
  dayName: string;
  date: string;
  highTemp: number;
  lowTemp: number | null;
  temperatureTrend: string;
  windSummary: string;
  precipPercent: number;
  goNoGo: GoNoGoLevel;
  icon: string;
  shortForecast: string;
  detailedForecast: string;
  sprayGoodHours: number;
  sprayMarginalHours: number;
  harvestGoodHours: number;
  harvestMarginalHours: number;
  hourlyScores: HourlyScoringResult[];
}
