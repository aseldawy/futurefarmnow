import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { Farm } from '../types/farm';
import { CachedWeatherEndpoints, CachedForecastData } from '../types/weather';

interface FarmState {
  // Persisted
  farms: Farm[];
  selectedFarmId: string | null;
  weatherEndpoints: Record<string, CachedWeatherEndpoints>;

  // Transient
  forecastData: CachedForecastData | null;
  isLoadingWeather: boolean;
  weatherError: string | null;
  isWeatherPopupOpen: boolean;
  isWeatherPopupExpanded: boolean;
  isAddingFarm: boolean;
  pendingFarmName: string;

  // Actions
  addFarm: (farm: Farm) => void;
  removeFarm: (id: string) => void;
  selectFarm: (id: string | null) => void;
  updateFarm: (id: string, updates: Partial<Farm>) => void;
  setIsAddingFarm: (v: boolean) => void;
  setPendingFarmName: (v: string) => void;
  getMostRecentFarm: () => Farm | null;
  setWeatherEndpoints: (farmId: string, endpoints: CachedWeatherEndpoints) => void;
  setForecastData: (data: CachedForecastData | null) => void;
  setIsLoadingWeather: (v: boolean) => void;
  setWeatherError: (v: string | null) => void;
  setIsWeatherPopupOpen: (v: boolean) => void;
  setIsWeatherPopupExpanded: (v: boolean) => void;
}

export const useFarmStore = create<FarmState>()(
  persist(
    (set, get) => ({
      // Persisted defaults
      farms: [],
      selectedFarmId: null,
      weatherEndpoints: {},

      // Transient defaults
      forecastData: null,
      isLoadingWeather: false,
      weatherError: null,
      isWeatherPopupOpen: true,
      isWeatherPopupExpanded: false,
      isAddingFarm: false,
      pendingFarmName: '',

      addFarm: (farm) =>
        set((s) => ({
          farms: [...s.farms, farm],
        })),

      removeFarm: (id) =>
        set((s) => ({
          farms: s.farms.filter((f) => f.id !== id),
          selectedFarmId: s.selectedFarmId === id ? null : s.selectedFarmId,
          forecastData: s.selectedFarmId === id ? null : s.forecastData,
        })),

      selectFarm: (id) =>
        set({
          selectedFarmId: id,
          forecastData: null,
          weatherError: null,
          isWeatherPopupOpen: id !== null,
        }),

      updateFarm: (id, updates) =>
        set((s) => ({
          farms: s.farms.map((f) =>
            f.id === id ? { ...f, ...updates, updatedAt: new Date().toISOString() } : f
          ),
        })),

      setIsAddingFarm: (v) => set({ isAddingFarm: v, pendingFarmName: '' }),
      setPendingFarmName: (v) => set({ pendingFarmName: v }),

      getMostRecentFarm: () => {
        const { farms } = get();
        if (farms.length === 0) return null;
        return farms.reduce((latest, f) =>
          new Date(f.updatedAt) > new Date(latest.updatedAt) ? f : latest
        );
      },

      setWeatherEndpoints: (farmId, endpoints) =>
        set((s) => ({
          weatherEndpoints: { ...s.weatherEndpoints, [farmId]: endpoints },
        })),

      setForecastData: (data) => set({ forecastData: data }),
      setIsLoadingWeather: (v) => set({ isLoadingWeather: v }),
      setWeatherError: (v) => set({ weatherError: v }),
      setIsWeatherPopupOpen: (v) => set({ isWeatherPopupOpen: v }),
      setIsWeatherPopupExpanded: (v) => set({ isWeatherPopupExpanded: v }),
    }),
    {
      name: 'ffn-farms',
      partialize: (state) => ({
        farms: state.farms,
        selectedFarmId: state.selectedFarmId,
        weatherEndpoints: state.weatherEndpoints,
      }),
    }
  )
);
