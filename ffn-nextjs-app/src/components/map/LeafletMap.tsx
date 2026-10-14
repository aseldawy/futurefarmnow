'use client'

import { useEffect, useRef, useState, memo } from 'react'
import L from 'leaflet'
import 'leaflet-draw'
import { useMapStore } from '@/lib/stores/mapStore'
import { useFarmStore } from '@/lib/stores/farmStore'
import { useTheme } from '@/lib/contexts/ThemeContext'
import { toast } from '@/lib/utils/toast'
import { valueToGrayscale, ndviToColor } from '@/lib/utils/color'
import { createSafePopupContent, formatNumber } from '@/lib/utils/sanitize'
import { WeatherOverlay } from '@/components/weather/WeatherOverlay'

// Fix for default markers in Leaflet
delete (L.Icon.Default.prototype as any)._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
})

function LeafletMapComponent() {
  const mapRef = useRef<HTMLDivElement>(null)
  const mapInstanceRef = useRef<L.Map | null>(null)
  const drawControlRef = useRef<L.Control.Draw | null>(null)
  const vectorTileLayerRef = useRef<L.TileLayer | null>(null)
  const soilImageOverlayRef = useRef<L.ImageOverlay | null>(null)
  const ndviImageOverlayRef = useRef<L.ImageOverlay | null>(null)
  const etmapImageOverlayRef = useRef<L.ImageOverlay | null>(null)
  const farmlandLayerRef = useRef<L.GeoJSON | null>(null)
  const samplePointsLayerRef = useRef<L.LayerGroup | null>(null)
  const drawnItemsRef = useRef<L.FeatureGroup | null>(null)
  const tileLayerRef = useRef<L.TileLayer | null>(null)
  const farmlandValueMapRef = useRef<Map<string, any> | null>(null)
  const farmPolygonsLayerRef = useRef<L.LayerGroup | null>(null)
  const isUpdatingFromURL = useRef(false)
  const hasCheckedURLCoordinates = useRef(false)
  const userMarkerTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const urlUpdateTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const isMapMovingRef = useRef(false)
  const lastAppliedState = useRef<{ center?: [number, number], zoom?: number }>({})

  const {
    selectedDataset,
    currentBounds,
    currentCenter,
    currentZoom,
    drawnPolygon,
    soilImageUrl,
    soilImageBounds,
    ndviImageUrl,
    ndviImageBounds,
    etmapImageUrl,
    etmapImageBounds,
    farmlandGeoJSON,
    samplePoints,
    setCurrentBounds,
    setDrawnPolygon,
    addDrawnPolygon,
    setDrawnPolygons,
    setCurrentZoom,
    setCurrentCenter,
    setMapInstance,
    setDrawnItems
  } = useMapStore()

  const { resolvedTheme } = useTheme()
  const [isInitialLoad, setIsInitialLoad] = useState(true)

  const createCustomPanes = (map: L.Map) => {
    if (!map.getPane('imagePane')) {
      const imagePane = map.createPane('imagePane')
      imagePane.style.zIndex = '250'
      // console.log('Created imagePane with z-index 250')
    }

    if (!map.getPane('vectorPane')) {
      const vectorPane = map.createPane('vectorPane')
      vectorPane.style.zIndex = '350'
      // console.log('Created vectorPane with z-index 350')
    }
  }

  const ensureDrawnItemsOnTop = () => {
    if (drawnItemsRef.current) {
      drawnItemsRef.current.eachLayer((layer: any) => {
        layer.setStyle({
          opacity: 1,
          fillOpacity: 0,
          color: '#3b82f6',
          weight: 2
        })
      })
      drawnItemsRef.current.bringToFront()
    }
  }

  // Function to pan to user's location
  const panToUserLocation = () => {
    if (!mapInstanceRef.current) return

    if (!navigator.geolocation) {
      toast.error('Geolocation is not supported by your browser')
      return
    }

    // Show loading toast
    const loadingToast = toast.loading('Getting your location...')

    navigator.geolocation.getCurrentPosition(
      (position) => {
        const { latitude, longitude } = position.coords

        // Pan and zoom to user's location
        mapInstanceRef.current?.setView([latitude, longitude], 15)

        // Add a temporary marker at user's location
        const userMarker = L.marker([latitude, longitude], {
          icon: L.divIcon({
            className: 'user-location-marker',
            html: `<div style="
              width: 20px;
              height: 20px;
              background-color: #3b82f6;
              border: 3px solid white;
              border-radius: 50%;
              box-shadow: 0 2px 4px rgba(0,0,0,0.3);
              position: relative;
            ">
              <div style="
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                width: 40px;
                height: 40px;
                border: 2px solid #3b82f6;
                border-radius: 50%;
                opacity: 0;
                animation: pulse 2s ease-out infinite;
              "></div>
            </div>`,
            iconSize: [20, 20],
            iconAnchor: [10, 10]
          })
        }).addTo(mapInstanceRef.current!)

        // Add pulse animation CSS if not already added
        if (!document.getElementById('location-pulse-style')) {
          const style = document.createElement('style')
          style.id = 'location-pulse-style'
          style.textContent = `
            @keyframes pulse {
              0% {
                opacity: 0.8;
                transform: translate(-50%, -50%) scale(1);
              }
              100% {
                opacity: 0;
                transform: translate(-50%, -50%) scale(2);
              }
            }
          `
          document.head.appendChild(style)
        }

        // Remove marker after 5 seconds
        userMarkerTimeoutRef.current = setTimeout(() => {
          if (mapInstanceRef.current) {
            mapInstanceRef.current.removeLayer(userMarker)
          }
        }, 5000)

        // Dismiss loading and show success
        toast.dismiss(loadingToast)
        toast.success('Moved to your location')
      },
      (error) => {
        toast.dismiss(loadingToast)

        switch (error.code) {
          case error.PERMISSION_DENIED:
            toast.error('Location access denied. Please enable location permissions.')
            break
          case error.POSITION_UNAVAILABLE:
            toast.error('Unable to retrieve your location')
            break
          case error.TIMEOUT:
            toast.error('Location request timed out')
            break
          default:
            toast.error('An error occurred while getting your location')
        }
      },
      {
        enableHighAccuracy: true,
        timeout: 10000,
        maximumAge: 0
      }
    )
  }

  useEffect(() => {
    if (!mapRef.current || mapInstanceRef.current) return

    const map = L.map(mapRef.current, {
      center: [36.7783, -119.4179], // California coordinates
      zoom: 6,
      minZoom: 3,
      maxZoom: 19,
      zoomControl: true,
      preferCanvas: true,
      attributionControl: false,
    })

    const tileLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© OpenStreetMap contributors',
      maxZoom: 19,
      keepBuffer: 4,
      updateWhenIdle: true,
      updateWhenZooming: false,
      zoomOffset: 0,
      tileSize: 256,
      crossOrigin: true,
    }).addTo(map)

    tileLayerRef.current = tileLayer

    const drawnItems = new L.FeatureGroup()
    map.addLayer(drawnItems)
    drawnItemsRef.current = drawnItems

    createCustomPanes(map)

    setMapInstance(map)
    setDrawnItems(drawnItems)

    const drawControl = new L.Control.Draw({
      position: 'topright',
      draw: {
        polygon: {
          showArea: true,
          showLength: true,
          shapeOptions: {
            color: '#3b82f6',
            weight: 2,
            opacity: 1,
            fillColor: '#3b82f6',
            fillOpacity: 0
          }
        },
        rectangle: {
          shapeOptions: {
            color: '#3b82f6',
            weight: 2,
            opacity: 1,
            fillColor: '#3b82f6',
            fillOpacity: 0
          }
        },
        circle: false,
        marker: false,
        polyline: false,
        circlemarker: false,
      },
      edit: {
        featureGroup: drawnItems,
        remove: true,
      },
    })
    map.addControl(drawControl)
    drawControlRef.current = drawControl

    map.on('draw:created', (event: any) => {
      const layer = event.layer

      layer.setStyle({
        color: '#3b82f6',
        weight: 2,
        opacity: 1,
        fillColor: '#3b82f6',
        fillOpacity: 0
      })

      drawnItems.addLayer(layer)

      const geoJSON = layer.toGeoJSON()
      console.log('Drawn polygon coordinates:', geoJSON.geometry.coordinates)
      console.log('Map center:', map.getCenter())
      console.log('Map bounds:', map.getBounds())
      console.log('Map CRS:', map.options.crs)
      addDrawnPolygon(geoJSON.geometry)

      toast.success('Polygon drawn successfully')
    })

    map.on('draw:deleted', (event: any) => {
      drawnItems.clearLayers()
      setDrawnPolygon(null)
      setDrawnPolygons([])
      toast.success('Area cleared! You can draw a new area now.')
    })

    map.on('draw:edited', (event: any) => {
      const layers = event.layers
      layers.eachLayer((layer: any) => {
        layer.setStyle({
          color: '#3b82f6',
          weight: 2,
          opacity: 1,
          fillColor: '#3b82f6',
          fillOpacity: 0
        })

        const geoJSON = layer.toGeoJSON()
        setDrawnPolygon(geoJSON.geometry)
      })
      toast.success('Area updated successfully!')
    })

    map.on('draw:deletestart', () => {
      toast.info('Click on the area you want to remove, then click Save to confirm.')
    })

    map.on('draw:editstart', () => {
      toast.info('Drag the corners to adjust your area, then click Save to confirm.')
    })

    // Optimized state update function - only called on moveend/zoomend
    const updateMapState = () => {
      if (isUpdatingFromURL.current) return

      const bounds = map.getBounds()
      const center = map.getCenter()
      const zoom = map.getZoom()

      const newBounds = {
        minx: bounds.getWest(),
        miny: bounds.getSouth(),
        maxx: bounds.getEast(),
        maxy: bounds.getNorth(),
      }

      // Round values to prevent micro-updates
      const newCenter: [number, number] = [
        Math.round(center.lat * 10000) / 10000,
        Math.round(center.lng * 10000) / 10000
      ]
      const newZoom = Math.round(zoom * 10) / 10

      // Only update if meaningfully different
      const needsUpdate = (
        !lastAppliedState.current.center ||
        Math.abs(lastAppliedState.current.center[0] - newCenter[0]) > 0.001 ||
        Math.abs(lastAppliedState.current.center[1] - newCenter[1]) > 0.001 ||
        Math.abs((lastAppliedState.current.zoom || 0) - newZoom) > 0.1
      )

      if (needsUpdate) {
        lastAppliedState.current = { center: newCenter, zoom: newZoom } // Update state in one batch
        setCurrentBounds(newBounds)
        setCurrentZoom(newZoom)
        setCurrentCenter(newCenter)
      }
    }

    // Track map movement state
    map.on('movestart zoomstart', () => {
      isMapMovingRef.current = true
    })

    // Only update state when movement ends (not during movement)
    map.on('moveend', () => {
      isMapMovingRef.current = false
      if (!isUpdatingFromURL.current) {
        // Use requestIdleCallback for better performance
        if (window.requestIdleCallback) {
          window.requestIdleCallback(() => updateMapState(), { timeout: 50 })
        } else {
          updateMapState()
        }
      }
    })

    map.on('zoomend', () => {
      isMapMovingRef.current = false
      if (!isUpdatingFromURL.current) {
        // Use requestIdleCallback for better performance
        if (window.requestIdleCallback) {
          window.requestIdleCallback(() => updateMapState(), { timeout: 50 })
        } else {
          updateMapState()
        }
      }
    })

    mapInstanceRef.current = map

    // Cleanup
    return () => {
      // Clear timeouts
      if (userMarkerTimeoutRef.current) {
        clearTimeout(userMarkerTimeoutRef.current)
        userMarkerTimeoutRef.current = null
      }

      if (urlUpdateTimeoutRef.current) {
        clearTimeout(urlUpdateTimeoutRef.current)
        urlUpdateTimeoutRef.current = null
      }

      // Remove style element if it exists
      const styleElement = document.getElementById('location-pulse-style')
      if (styleElement) {
        styleElement.remove()
      }

      if (mapInstanceRef.current) {
        // Remove all event listeners
        mapInstanceRef.current.off()

        // Clean up draw control
        if (drawControlRef.current) {
          mapInstanceRef.current.removeControl(drawControlRef.current)
          drawControlRef.current = null
        }

        // Clean up all layer references
        if (vectorTileLayerRef.current) {
          mapInstanceRef.current.removeLayer(vectorTileLayerRef.current)
          vectorTileLayerRef.current = null
        }

        if (tileLayerRef.current) {
          mapInstanceRef.current.removeLayer(tileLayerRef.current)
          tileLayerRef.current = null
        }

        if (soilImageOverlayRef.current) {
          mapInstanceRef.current.removeLayer(soilImageOverlayRef.current)
          soilImageOverlayRef.current = null
        }

        if (farmlandLayerRef.current) {
          mapInstanceRef.current.removeLayer(farmlandLayerRef.current)
          farmlandLayerRef.current = null
        }

        if (farmPolygonsLayerRef.current) {
          mapInstanceRef.current.removeLayer(farmPolygonsLayerRef.current)
          farmPolygonsLayerRef.current = null
        }

        if (drawnItemsRef.current) {
          drawnItemsRef.current.clearLayers()
          mapInstanceRef.current.removeLayer(drawnItemsRef.current)
          drawnItemsRef.current = null
        }

        // Clear store references
        setMapInstance(null)
        setDrawnItems(null)

        // Remove the map instance
        mapInstanceRef.current.remove()
        mapInstanceRef.current = null
      }
    }
  }, [setCurrentBounds, setCurrentZoom, setCurrentCenter, setDrawnPolygon, addDrawnPolygon, setDrawnPolygons, setMapInstance, setDrawnItems])

  // Handle initial URL state when map is ready
  useEffect(() => {
    if (!mapInstanceRef.current || hasCheckedURLCoordinates.current) return

    // Mark as checked
    hasCheckedURLCoordinates.current = true

    // Get initial state from URL via store (which was set by useURLSync on mount)
    const state = useMapStore.getState()
    if (state.currentCenter && state.currentZoom !== undefined && state.currentZoom !== null) {
      isUpdatingFromURL.current = true
      mapInstanceRef.current.setView(state.currentCenter, state.currentZoom)

      setTimeout(() => {
        isUpdatingFromURL.current = false
      }, 200)
    }
  }, []) // Empty dependency - run once on mount

  useEffect(() => {
    if (!mapInstanceRef.current || !tileLayerRef.current) return

    // Create new tile layer based on theme
    let newTileLayer: L.TileLayer

    if (resolvedTheme === 'dark') {
      // Use Esri Dark Gray Canvas tiles for better boundary contrast.
      newTileLayer = L.tileLayer('https://services.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Tiles © Esri',
        maxZoom: 16,
        keepBuffer: 4,
        updateWhenIdle: true,
        updateWhenZooming: false,
        crossOrigin: true,
      })
    } else {
      // Use standard OpenStreetMap for light theme
      newTileLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors',
        maxZoom: 19,
        keepBuffer: 4,
        updateWhenIdle: true,
        updateWhenZooming: false,
        crossOrigin: true,
      })
    }

    // Add new layer first, then remove old one to prevent flicker
    const oldTileLayer = tileLayerRef.current
    newTileLayer.addTo(mapInstanceRef.current)

    // Small delay to ensure new tiles start loading before removing old ones
    setTimeout(() => {
      if (mapInstanceRef.current && oldTileLayer) {
        mapInstanceRef.current.removeLayer(oldTileLayer)
      }
    }, 100)

    tileLayerRef.current = newTileLayer

    // Re-add vector tile layer if one is selected (preserve data layers)
    if (vectorTileLayerRef.current && selectedDataset) {
      // Ensure vector tiles are above base map
      vectorTileLayerRef.current.bringToFront()
    }

    // Cleanup function for this effect
    return () => {
      if (newTileLayer && mapInstanceRef.current) {
        try {
          mapInstanceRef.current.removeLayer(newTileLayer)
        } catch (e) {
          // Layer might already be removed, ignore error
        }
      }
    }
  }, [resolvedTheme, selectedDataset])

  useEffect(() => {
    if (!mapInstanceRef.current) return

    // Remove existing vector tile layer
    if (vectorTileLayerRef.current) {
      mapInstanceRef.current.removeLayer(vectorTileLayerRef.current)
      vectorTileLayerRef.current = null
    }

    // Add new vector tile layer if dataset is selected
    if (selectedDataset) {
      const baseURL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://raptor.cs.ucr.edu/futurefarmnow-backend-0.3-RC1'
      const tileUrl = `${baseURL}/vectors/${selectedDataset}/tile-{z}-{x}-{y}.png`

      const vectorTileLayer = L.tileLayer(tileUrl, {
        maxZoom: 19,
        minZoom: 1,
        opacity: 0.7,
        attribution: 'FutureFarmNow',
        pane: 'vectorPane',
        keepBuffer: 2,
        updateWhenIdle: true,
        updateWhenZooming: false,
        crossOrigin: true,
      })

      vectorTileLayer.addTo(mapInstanceRef.current)
      vectorTileLayerRef.current = vectorTileLayer

      // Vector tiles will automatically use vectorPane with z-index 350

      // Only show toast and center map if not initial load
      if (!isInitialLoad) {
        // Check if we should center map (only if no URL coordinates were ever provided)
        if (!hasCheckedURLCoordinates.current) {
          hasCheckedURLCoordinates.current = true
          // Check current store state for URL coordinates at time of execution
          const storeState = useMapStore.getState()
          if (!storeState.currentCenter && !storeState.currentZoom) {
            if (selectedDataset === 'farmland') {
              mapInstanceRef.current.setView([36.7783, -119.4179], 6)
            } else if (selectedDataset === 'AZ_Farmland') {
              mapInstanceRef.current.setView([34.0489, -111.0937], 6)
            }
          }
        }

        // Always show toast when dataset changes (but not on zoom changes)
        if (selectedDataset === 'farmland') {
          toast.success('Loaded California farmland dataset')
        } else if (selectedDataset === 'AZ_Farmland') {
          toast.success('Loaded Arizona farmland dataset')
        }
      } else {
        // On initial load, check URL coordinates once
        hasCheckedURLCoordinates.current = true
        const storeState = useMapStore.getState()
        if (!storeState.currentCenter && !storeState.currentZoom) {
          if (selectedDataset === 'farmland') {
            mapInstanceRef.current.setView([36.7783, -119.4179], 6)
          } else if (selectedDataset === 'AZ_Farmland') {
            mapInstanceRef.current.setView([34.0489, -111.0937], 6)
          }
        }
        setIsInitialLoad(false)
      }
    }

    // Cleanup function
    return () => {
      if (vectorTileLayerRef.current && mapInstanceRef.current) {
        try {
          mapInstanceRef.current.removeLayer(vectorTileLayerRef.current)
        } catch (e) {
          // Layer might already be removed, ignore error
        }
      }
    }
  }, [selectedDataset, isInitialLoad])

  useEffect(() => {
    if (!mapInstanceRef.current) return

    // Store reference to current overlay for cleanup
    const currentOverlay = soilImageOverlayRef.current

    // Remove existing overlay
    if (currentOverlay) {
      mapInstanceRef.current.removeLayer(currentOverlay)
      soilImageOverlayRef.current = null
    }

    ensureDrawnItemsOnTop()

    let newOverlay: L.ImageOverlay | null = null
    if (soilImageUrl && soilImageBounds) {
      newOverlay = L.imageOverlay(soilImageUrl, soilImageBounds, {
        opacity: 1.0,
        className: 'soil-image-overlay',
        pane: 'imagePane'
      })

      newOverlay.addTo(mapInstanceRef.current)
      soilImageOverlayRef.current = newOverlay

      // Vector tiles will stay on top automatically via vectorPane
    }

    // Cleanup function
    return () => {
      if (newOverlay && mapInstanceRef.current) {
        try {
          mapInstanceRef.current.removeLayer(newOverlay)
        } catch (e) {
          // Layer might already be removed, ignore error
        }
      }
    }
  }, [soilImageUrl, soilImageBounds])

  useEffect(() => {
    if (!mapInstanceRef.current) return

    // Store reference to current overlay for cleanup
    const currentOverlay = ndviImageOverlayRef.current

    // Remove existing overlay
    if (currentOverlay) {
      mapInstanceRef.current.removeLayer(currentOverlay)
      ndviImageOverlayRef.current = null
    }

    ensureDrawnItemsOnTop()

    let newOverlay: L.ImageOverlay | null = null
    if (ndviImageUrl && ndviImageBounds) {
      newOverlay = L.imageOverlay(ndviImageUrl, ndviImageBounds, {
        opacity: 1.0,
        className: 'ndvi-image-overlay',
        pane: 'imagePane'
      })

      newOverlay.addTo(mapInstanceRef.current)
      ndviImageOverlayRef.current = newOverlay

      // Vector tiles will stay on top automatically via vectorPane
    }

    // Cleanup function
    return () => {
      if (newOverlay && mapInstanceRef.current) {
        try {
          mapInstanceRef.current.removeLayer(newOverlay)
        } catch (e) {
          // Layer might already be removed, ignore error
        }
      }
    }
  }, [ndviImageUrl, ndviImageBounds, soilImageUrl])

  // ETMap image overlay effect
  useEffect(() => {
    if (!mapInstanceRef.current) return

    // Store reference to current overlay for cleanup
    const currentOverlay = etmapImageOverlayRef.current

    // Remove existing overlay
    if (currentOverlay) {
      mapInstanceRef.current.removeLayer(currentOverlay)
      etmapImageOverlayRef.current = null
    }

    ensureDrawnItemsOnTop()

    let newOverlay: L.ImageOverlay | null = null
    if (etmapImageUrl && etmapImageBounds) {
      newOverlay = L.imageOverlay(etmapImageUrl, etmapImageBounds, {
        opacity: 1.0,
        className: 'etmap-image-overlay',
        pane: 'imagePane'
      })

      newOverlay.addTo(mapInstanceRef.current)
      etmapImageOverlayRef.current = newOverlay
    }

    // Cleanup function
    return () => {
      if (newOverlay && mapInstanceRef.current) {
        try {
          mapInstanceRef.current.removeLayer(newOverlay)
        } catch (e) {
          // Layer might already be removed, ignore error
        }
      }
    }
  }, [etmapImageUrl, etmapImageBounds])

  useEffect(() => {
    if (!mapInstanceRef.current) return

    // Store reference to current layer for cleanup
    const currentFarmlandLayer = farmlandLayerRef.current

    // Remove existing farmland layer
    if (currentFarmlandLayer) {
      mapInstanceRef.current.removeLayer(currentFarmlandLayer)
      farmlandLayerRef.current = null
    }

    // Add new farmland layer if GeoJSON is available
    if (farmlandGeoJSON) {
      const geoJSONData = farmlandGeoJSON.geoJSON || farmlandGeoJSON
      const colorData = farmlandGeoJSON.colorData
      const ndviData = farmlandGeoJSON.ndviData

      // Check if we can reuse cached value map
      let farmlandValueMap = farmlandValueMapRef.current

      // Only recreate value map if data has changed (check data presence and size)
      const hasNdviData = ndviData && Array.isArray(ndviData) && ndviData.length > 0
      const hasColorData = colorData && colorData.farmlands && colorData.farmlands.length > 0
      const expectedMapSize = (hasNdviData ? ndviData.length * 3 : 0) + (hasColorData ? colorData.farmlands.length * 2 : 0)

      if (!farmlandValueMap || farmlandValueMap.size !== expectedMapSize) {
        farmlandValueMap = new Map()

        // Pre-compute NDVI averages for performance
        if (ndviData && Array.isArray(ndviData)) {
          ndviData.forEach((farmland: any) => {
            if (farmland.objectid && farmland.results) {
              // Pre-calculate mean NDVI from time series
              const validValues = farmland.results
                .map((point: any) => point.mean)
                .filter((val: any) => val !== undefined && val !== null && !isNaN(val))

              let precomputedAverage = null
              if (validValues.length > 0) {
                precomputedAverage = validValues.reduce((sum: number, val: number) => sum + val, 0) / validValues.length
              }

              // Store precomputed average for instant lookup
              farmlandValueMap!.set(`ndvi_avg_${farmland.objectid}`, precomputedAverage)
              // Store NDVI time series data
              farmlandValueMap!.set(`ndvi_${farmland.objectid}`, farmland.results)
              farmlandValueMap!.set(`stats_${farmland.objectid}`, {
                ...farmland,
                type: 'ndvi',
                dataCount: farmland.results.length,
                precomputedAverage
              })
            }
          })
        }

        // Process soil color data if available
        if (colorData && colorData.farmlands) {
          colorData.farmlands.forEach((farmland: any, index: number) => {
            // Try different ways to get the farmland ID - prioritize objectid since that's what we're seeing
            const possibleIds = [
              farmland.objectid,
              farmland.OBJECTID,
              farmland.id,
              farmland.fid,
              farmland.FID,
              farmland.ID,
              farmland.properties?.objectid,
              farmland.properties?.OBJECTID,
              farmland.properties?.id,
              farmland.properties?.fid,
              farmland.properties?.FID,
              farmland.properties?.ID,
              index // Use array index as fallback
            ].filter(id => id !== undefined && id !== null)

            const value = farmland.average ?? farmland.mean ?? farmland.results?.mean ?? farmland.results?.average

            if (possibleIds.length > 0 && value !== undefined) {
              // Map all possible IDs to the same value, and also store the full farmland data
              possibleIds.forEach(id => {
                farmlandValueMap!.set(id.toString(), value)
                // Also store the full farmland stats for popup display
                farmlandValueMap!.set(`stats_${id}`, farmland)
              })
            }
          })
        }

        // Cache the value map for reuse
        farmlandValueMapRef.current = farmlandValueMap
      }

      const farmlandLayer = L.geoJSON(geoJSONData, {
        style: (feature) => {
          // Default style
          let fillColor = '#3b82f6'
          let fillOpacity = 0.1

          if (feature?.properties) {
            // Try multiple ID fields - prioritize objectid since that's what we're seeing in data
            const possibleFeatureIds = [
              feature.properties.objectid,
              feature.properties.OBJECTID,
              feature.properties.id,
              feature.properties.fid,
              feature.properties.ID,
              feature.properties.FID,
              feature.id // GeoJSON feature ID
            ].filter(id => id !== undefined && id !== null)

            // Check for cached values first (optimized lookup)
            let value
            let isNDVIData = false

            // First try to find pre-computed NDVI average (O(1) lookup)
            if (ndviData && Array.isArray(ndviData)) {
              for (const id of possibleFeatureIds) {
                const precomputedAvg = farmlandValueMap!.get(`ndvi_avg_${id}`)
                if (precomputedAvg !== undefined && precomputedAvg !== null) {
                  value = precomputedAvg
                  isNDVIData = true
                  break
                }
              }
            }

            // If no NDVI data found, try soil color data
            if (!isNDVIData && colorData) {
              // Try to find a matching soil value using the same ID fields
              for (const id of possibleFeatureIds) {
                const soilValue = farmlandValueMap!.get(id.toString())
                if (soilValue !== undefined) {
                  value = soilValue
                  break
                }
              }
            }

            // Apply coloring based on data type (optimized color assignment)
            if (value !== undefined) {
              if (isNDVIData) {
                // NDVI color scheme: Red (poor) -> Yellow (moderate) -> Green (excellent)
                fillColor = ndviToColor(value)
                fillOpacity = 0.7
              } else if (colorData && colorData.min !== undefined && colorData.max !== undefined) {
                // Soil data color scheme: grayscale
                fillColor = valueToGrayscale(value, colorData.min, colorData.max)
                fillOpacity = 0.7 // Higher opacity for colored farmlands
              }
            }
          }

          return {
            color: '#666', // Darker border for contrast
            weight: 1.5,
            opacity: 0.8,
            fillColor: fillColor,
            fillOpacity: fillOpacity
          }
        },
        onEachFeature: (feature, layer) => {
          // Lazy popup binding - only generate content when popup is opened
          if (feature.properties) {
            layer.bindPopup(() => {
              // Find matching farmland stats
              const possibleFeatureIds = [
                feature.properties.objectid,
                feature.properties.OBJECTID,
                feature.properties.id,
                feature.properties.fid,
                feature.properties.ID,
                feature.properties.FID,
                feature.id
              ].filter(id => id !== undefined && id !== null)

              let farmlandStats
              let matchedId

              // Try to find matching stats
              for (const id of possibleFeatureIds) {
                farmlandStats = farmlandValueMap!.get(`stats_${id}`)
                if (farmlandStats) {
                  matchedId = id
                  break
                }
              }

              const sections = []

              // Add statistics based on data type
              if (farmlandStats) {
                if (farmlandStats.type === 'ndvi') {
                  // NDVI Time Series Display
                  const ndviItems = []
                  ndviItems.push({ label: 'Data Points', value: `${farmlandStats.dataCount} measurements` })

                  // Get the actual time series data
                  const ndviTimeSeries = farmlandValueMap!.get(`ndvi_${matchedId}`)
                  if (ndviTimeSeries && Array.isArray(ndviTimeSeries)) {
                    // Calculate summary statistics from time series
                    const validValues = ndviTimeSeries
                      .map((point: any) => point.mean)
                      .filter((value: any) => value !== undefined && value !== null && !isNaN(value))

                    if (validValues.length > 0) {
                      const avg = validValues.reduce((sum: number, val: number) => sum + val, 0) / validValues.length
                      const min = Math.min(...validValues)
                      const max = Math.max(...validValues)

                      ndviItems.push({ label: 'Average NDVI', value: formatNumber(avg, 3) })
                      ndviItems.push({ label: 'Range', value: `${formatNumber(min, 3)} to ${formatNumber(max, 3)}` })

                      // Show health status
                      const healthStatus = avg > 0.5 ? 'Excellent' : avg > 0.2 ? 'Good' : 'Poor'
                      ndviItems.push({ label: 'Health Status', value: healthStatus })

                      // Show recent measurements (last 3)
                      const recentData = ndviTimeSeries.slice(-3)
                      const recentMeasurements = recentData
                        .map((point: any) => {
                          const date = new Date(point.date).toLocaleDateString()
                          const value = formatNumber(point.mean, 3)
                          return `${date}: ${value}`
                        })
                        .join(', ')

                      if (recentMeasurements) {
                        ndviItems.push({ label: 'Recent', value: recentMeasurements })
                      }
                    }
                  }

                  sections.push({
                    title: '📊 NDVI Crop Health Time Series',
                    items: ndviItems,
                    className: 'background: #f0f9ff; padding: 8px; border-radius: 4px; margin-bottom: 8px; border-left: 3px solid #10b981;'
                  })
                } else {
                  // Original soil statistics display
                  const soilItems = [
                    { label: 'Average', value: farmlandStats.average },
                    { label: 'Min', value: farmlandStats.min },
                    { label: 'Max', value: farmlandStats.max },
                    { label: 'Count', value: farmlandStats.count ? `${farmlandStats.count.toLocaleString()} pixels` : null }
                  ]

                  if (farmlandStats.stddev) {
                    soilItems.push({ label: 'Std Dev', value: farmlandStats.stddev })
                  }

                  sections.push({
                    title: '🌾 Soil Statistics',
                    items: soilItems.filter(item => item.value !== null && item.value !== undefined)
                  })
                }
              }

              // Add feature properties
              const propertyItems = Object.entries(feature.properties)
                .slice(0, 4) // Show first 4 properties
                .map(([key, value]) => ({ label: key, value }))

              if (propertyItems.length > 0) {
                sections.push({
                  title: 'Properties',
                  items: propertyItems,
                  className: 'font-size: 12px; color: #666; background: transparent; padding: 0; margin: 0;'
                })
              }

              return createSafePopupContent({
                title: matchedId ? `Farmland ID: ${matchedId}` : undefined,
                sections
              })
            }, {
              maxWidth: 300,
              className: 'farmland-popup'
            })
          }
        }
      })

      farmlandLayer.addTo(mapInstanceRef.current)
      farmlandLayerRef.current = farmlandLayer

      // Farmland GeoJSON will automatically be above the imagePane

      // Cleanup function
      return () => {
        if (farmlandLayer && mapInstanceRef.current) {
          try {
            mapInstanceRef.current.removeLayer(farmlandLayer)
          } catch (e) {
            // Layer might already be removed, ignore error
          }
        }
      }
    } else {
      // Clear cache when no farmland data
      farmlandValueMapRef.current = null
    }
  }, [farmlandGeoJSON])

  // Handle sample points display
  useEffect(() => {
    if (!mapInstanceRef.current) return

    // Remove existing sample points layer
    const currentSamplePointsLayer = samplePointsLayerRef.current
    if (currentSamplePointsLayer) {
      mapInstanceRef.current.removeLayer(currentSamplePointsLayer)
      samplePointsLayerRef.current = null
    }

    // Add new sample points if available
    if (samplePoints && samplePoints.length > 0) {
      const samplePointsLayer = L.layerGroup()

      samplePoints.forEach((point) => {
        // Create a circle marker with ID label
        const marker = L.circleMarker([point.y, point.x], {
          radius: 12,
          fillColor: '#3b82f6',
          color: '#1e40af',
          weight: 2,
          opacity: 1,
          fillOpacity: 0.8
        })

        // Add ID label inside the circle
        const divIcon = L.divIcon({
          html: `<div style="display: flex; align-items: center; justify-content: center; width: 24px; height: 24px; background-color: #3b82f6; border: 2px solid #1e40af; border-radius: 50%; color: white; font-size: 12px; font-weight: bold;">${point.id}</div>`,
          iconSize: [24, 24],
          iconAnchor: [12, 12],
          className: 'sample-point-marker'
        })

        const labelMarker = L.marker([point.y, point.x], { icon: divIcon })

        // Add tooltip with coordinates on hover
        const lat = point.y.toFixed(6)
        const lng = point.x.toFixed(6)
        labelMarker.bindTooltip(`Sample Point ${point.id}<br/>Lat: ${lat}<br/>Lng: ${lng}`, {
          permanent: false,
          direction: 'top',
          offset: [0, -12]
        })

        samplePointsLayer.addLayer(labelMarker)
      })

      samplePointsLayerRef.current = samplePointsLayer
      mapInstanceRef.current.addLayer(samplePointsLayer)
    }
  }, [samplePoints])

  // Render persistent farm polygons
  const farms = useFarmStore((s) => s.farms)
  const selectedFarmId = useFarmStore((s) => s.selectedFarmId)

  useEffect(() => {
    if (!mapInstanceRef.current) return

    // Remove existing farm polygons layer
    if (farmPolygonsLayerRef.current) {
      mapInstanceRef.current.removeLayer(farmPolygonsLayerRef.current)
      farmPolygonsLayerRef.current = null
    }

    if (farms.length === 0) return

    const layerGroup = L.layerGroup()

    farms.forEach((farm) => {
      const isSelected = farm.id === selectedFarmId
      try {
        const geoLayer = L.geoJSON(farm.geometry as GeoJSON.GeoJsonObject, {
          style: {
            color: isSelected ? '#10b981' : '#9ca3af',
            weight: isSelected ? 3 : 1.5,
            opacity: isSelected ? 1 : 0.6,
            fillOpacity: isSelected ? 0.05 : 0,
            dashArray: isSelected ? undefined : '6 4',
          },
        })
        geoLayer.bindTooltip(farm.name, {
          permanent: false,
          direction: 'center',
          className: 'farm-tooltip',
        })
        layerGroup.addLayer(geoLayer)
      } catch {
        // Skip invalid geometries
      }
    })

    layerGroup.addTo(mapInstanceRef.current)
    farmPolygonsLayerRef.current = layerGroup

    return () => {
      if (layerGroup && mapInstanceRef.current) {
        try {
          mapInstanceRef.current.removeLayer(layerGroup)
        } catch {
          // ignore
        }
      }
    }
  }, [farms, selectedFarmId])

  return (
    <div className="relative h-full w-full">
      <div ref={mapRef} className="h-full w-full" />

      {/* Weather Overlay - positioned over map */}
      <WeatherOverlay />

      {/* Location button */}
      <button
        onClick={panToUserLocation}
        className="absolute top-32 right-3 bg-white dark:bg-gray-800 border-2 border-gray-300 dark:border-gray-600 rounded p-2 shadow-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-all duration-200 group z-[1000]"
        title="Go to my location"
        aria-label="Go to my location"
        style={{ marginTop: '60px' }}
        data-tutorial="location-button"
      >
        <svg
          width="20"
          height="20"
          viewBox="0 0 24 24"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          className="text-gray-600 dark:text-gray-300 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors"
        >
          <circle cx="12" cy="12" r="3" fill="currentColor" />
          <path
            d="M12 2V8M12 16V22M22 12H16M8 12H2"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
          />
          <circle
            cx="12"
            cy="12"
            r="10"
            stroke="currentColor"
            strokeWidth="2"
            fill="none"
          />
        </svg>
      </button>

      {/* Map attribution */}
      <div className="absolute bottom-2 right-2 bg-background/90 border px-2 py-1 text-xs rounded text-muted-foreground">
        © OpenStreetMap contributors
      </div>
    </div>
  )
}

// Memoize the component to prevent unnecessary re-renders
export default memo(LeafletMapComponent)
