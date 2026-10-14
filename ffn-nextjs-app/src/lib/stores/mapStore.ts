import { create } from 'zustand'
import { GeoJSONGeometry, BoundingBox, SoilLayerEnum, ETMapStatistics } from '@/lib/types/api'
import L from 'leaflet'

interface MapState {
  selectedDataset: string | null
  currentBounds: BoundingBox | null
  drawnPolygon: GeoJSONGeometry | null
  drawnPolygons: GeoJSONGeometry[] // Array to store multiple polygons
  currentZoom: number
  currentCenter: [number, number] | null
  mapInstance: L.Map | null
  drawnItems: L.FeatureGroup | null
  soilImageUrl: string | null
  soilImageBounds: [[number, number], [number, number]] | null
  farmlandGeoJSON: any | null
  ndviImageUrl: string | null
  ndviImageBounds: [[number, number], [number, number]] | null
  etmapImageUrl: string | null
  etmapImageBounds: [[number, number], [number, number]] | null
  etmapRequestId: string | null
  samplePoints: Array<{id: number, x: number, y: number}> | null
  soilAnalysisResult: { layer: string; depth: string; mean: number | null; min: number | null; max: number | null; median: number | null; stddev: number | null; count: number | null } | null
  ndviAnalysisResult: { fromDate: string; toDate: string; averageNdvi: number | null; maxNdvi: number | null; minNdvi: number | null; dataPoints: number } | null
  etmapAnalysisResult: ETMapStatistics | null
  selectedSoilLayer: SoilLayerEnum
  selectedSoilDepth: string
  selectedDateRange: {
    from: string
    to: string
  }

  // UI state
  activeTab: 'soil' | 'ndvi' | 'sample' | 'etmap'
  isAnalyzing: boolean
  clearTrigger: number // Incremented when clear is triggered

  // Actions
  setSelectedDataset: (dataset: string | null) => void
  setCurrentBounds: (bounds: BoundingBox | null) => void
  setDrawnPolygon: (polygon: GeoJSONGeometry | null) => void
  addDrawnPolygon: (polygon: GeoJSONGeometry) => void
  setDrawnPolygons: (polygons: GeoJSONGeometry[]) => void
  setCurrentZoom: (zoom: number) => void
  setCurrentCenter: (center: [number, number] | null) => void
  setMapInstance: (map: L.Map | null) => void
  setDrawnItems: (drawnItems: L.FeatureGroup | null) => void
  setSoilImageOverlay: (url: string | null, bounds: [[number, number], [number, number]] | null) => void
  setFarmlandGeoJSON: (geoJSON: any | null) => void
  setNDVIImageOverlay: (url: string | null, bounds: [[number, number], [number, number]] | null) => void
  setETMapImageOverlay: (url: string | null, bounds: [[number, number], [number, number]] | null) => void
  setETMapRequestId: (id: string | null) => void
  setSamplePoints: (points: Array<{id: number, x: number, y: number}> | null) => void
  setSoilAnalysisResult: (result: MapState['soilAnalysisResult']) => void
  setNdviAnalysisResult: (result: MapState['ndviAnalysisResult']) => void
  setEtmapAnalysisResult: (result: ETMapStatistics | null) => void
  clearArea: () => void
  clearAllAnalysis: () => void
  clearNDVICache: () => void
  setSelectedSoilLayer: (layer: SoilLayerEnum) => void
  setSelectedSoilDepth: (depth: string) => void
  setSelectedDateRange: (range: { from: string; to: string }) => void
  setActiveTab: (tab: 'soil' | 'ndvi' | 'sample' | 'etmap') => void
  setIsAnalyzing: (analyzing: boolean) => void
  reset: () => void
}

export const useMapStore = create<MapState>((set, get) => ({
  selectedDataset: 'farmland',
  currentBounds: null,
  drawnPolygon: null,
  drawnPolygons: [],
  currentZoom: 6,
  currentCenter: null,
  mapInstance: null,
  drawnItems: null,
  soilImageUrl: null,
  soilImageBounds: null,
  farmlandGeoJSON: null,
  ndviImageUrl: null,
  ndviImageBounds: null,
  etmapImageUrl: null,
  etmapImageBounds: null,
  etmapRequestId: null,
  samplePoints: null,
  soilAnalysisResult: null,
  ndviAnalysisResult: null,
  etmapAnalysisResult: null,
  selectedSoilLayer: 'ph',
  selectedSoilDepth: '0-5',
  selectedDateRange: {
    from: '2025-01-01',
    to: '2025-01-02',
  },
  activeTab: 'soil',
  isAnalyzing: false,
  clearTrigger: 0,

  setSelectedDataset: (dataset) => set({ selectedDataset: dataset }),
  setCurrentBounds: (bounds) => set({ currentBounds: bounds }),
  setDrawnPolygon: (polygon) => set({ drawnPolygon: polygon }),
  addDrawnPolygon: (polygon) => set((state) => ({ 
    drawnPolygons: [...state.drawnPolygons, polygon],
    drawnPolygon: polygon // Keep last polygon as current
  })),
  setDrawnPolygons: (polygons) => set({ drawnPolygons: polygons }),
  setCurrentZoom: (zoom) => set({ currentZoom: zoom }),
  setCurrentCenter: (center) => set({ currentCenter: center }),
  setMapInstance: (map) => set({ mapInstance: map }),
  setDrawnItems: (drawnItems) => set({ drawnItems }),
  setSoilImageOverlay: (url, bounds) => set({ soilImageUrl: url, soilImageBounds: bounds }),
  setFarmlandGeoJSON: (geoJSON) => set({ farmlandGeoJSON: geoJSON }),
  setNDVIImageOverlay: (url, bounds) => set({ ndviImageUrl: url, ndviImageBounds: bounds }),
  setETMapImageOverlay: (url, bounds) => set({ etmapImageUrl: url, etmapImageBounds: bounds }),
  setETMapRequestId: (id) => set({ etmapRequestId: id }),
  setSamplePoints: (points) => set({ samplePoints: points }),
  setSoilAnalysisResult: (result) => set({ soilAnalysisResult: result }),
  setNdviAnalysisResult: (result) => set({ ndviAnalysisResult: result }),
  setEtmapAnalysisResult: (result) => set({ etmapAnalysisResult: result }),
  clearArea: () => {
    const { drawnItems } = get()
    if (drawnItems) {
      drawnItems.clearLayers()
    }
    set({ drawnPolygon: null, drawnPolygons: [], soilImageUrl: null, soilImageBounds: null, farmlandGeoJSON: null, ndviImageUrl: null, ndviImageBounds: null, etmapImageUrl: null, etmapImageBounds: null, etmapRequestId: null, samplePoints: null, soilAnalysisResult: null, ndviAnalysisResult: null, etmapAnalysisResult: null })
  },
  clearAllAnalysis: () => {
    const { drawnItems, clearTrigger } = get()
    if (drawnItems) {
      drawnItems.clearLayers()
    }
    set({
      drawnPolygon: null,
      drawnPolygons: [],
      soilImageUrl: null,
      soilImageBounds: null,
      farmlandGeoJSON: null,
      ndviImageUrl: null,
      ndviImageBounds: null,
      etmapImageUrl: null,
      etmapImageBounds: null,
      etmapRequestId: null,
      samplePoints: null,
      soilAnalysisResult: null,
      ndviAnalysisResult: null,
      etmapAnalysisResult: null,
      isAnalyzing: false,
      clearTrigger: clearTrigger + 1
    })
  },
  clearNDVICache: () => {
    const { clearTrigger } = get()
    set({
      ndviImageUrl: null,
      ndviImageBounds: null,
      clearTrigger: clearTrigger + 1
    })
  },
  setSelectedSoilLayer: (layer) => set({ selectedSoilLayer: layer }),
  setSelectedSoilDepth: (depth) => set({ selectedSoilDepth: depth }),
  setSelectedDateRange: (range) => set({ selectedDateRange: range }),
  setActiveTab: (tab) => set({ activeTab: tab }),
  setIsAnalyzing: (analyzing) => set({ isAnalyzing: analyzing }),
  reset: () => set({
    selectedDataset: 'farmland',
    currentBounds: null,
    drawnPolygon: null,
    drawnPolygons: [],
    currentZoom: 6,
    currentCenter: null,
    mapInstance: null,
    drawnItems: null,
    soilImageUrl: null,
    soilImageBounds: null,
    farmlandGeoJSON: null,
    ndviImageUrl: null,
    ndviImageBounds: null,
    etmapImageUrl: null,
    etmapImageBounds: null,
    etmapRequestId: null,
    samplePoints: null,
    soilAnalysisResult: null,
    ndviAnalysisResult: null,
    etmapAnalysisResult: null,
    selectedSoilLayer: 'ph',
    selectedSoilDepth: '0-5',
    selectedDateRange: {
      from: '2025-01-01',
      to: '2025-02-28',
    },
    activeTab: 'soil',
    isAnalyzing: false,
    clearTrigger: 0,
  }),
}))