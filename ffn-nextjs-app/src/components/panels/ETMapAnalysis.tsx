'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { useMapStore } from '@/lib/stores/mapStore'
import { apiClient } from '@/lib/api/client'
import { AlertCircle, RefreshCw, Droplets, CheckCircle, Clock, XCircle, Download, History, ChevronDown, ChevronUp, Trash2, BarChart3, Share2, Bot } from 'lucide-react'
import { toast } from 'react-hot-toast'
import { GeoJSONGeometry, ETMapStatusResponse, ETMapStatus } from '@/lib/types/api'
import { useChatStore } from '@/lib/stores/chatStore'
import { cn } from '@/lib/utils'
import { formatDateString, parseDate } from '@/lib/utils'

// Status stages definition - maps backend statuses to UI display stages
// Multiple backend statuses can map to the same UI stage
const STAGES: { key: ETMapStatus; label: string; alternateKeys?: ETMapStatus[] }[] = [
  { key: 'queued', label: 'Queued', alternateKeys: ['claimed', 'checking_coverage'] },
  { key: 'landsat_started', label: 'Fetching Landsat Data', alternateKeys: ['landsat_done', 'landsat_skipped_covered'] },
  { key: 'prism_started', label: 'Fetching PRISM Data', alternateKeys: ['prism_done', 'prism_skipped_covered'] },
  { key: 'nldas_started', label: 'Fetching NLDAS Data', alternateKeys: ['nldas_done', 'nldas_skipped_covered', 'success'] },
  { key: 'calculation_started', label: 'Calculating ET Map' },
  { key: 'calculation_complete', label: 'Completed' },
]

// Helper to find which stage index a status belongs to
function getStageIndex(status: ETMapStatus | undefined): number {
  if (!status) return -1
  for (let i = 0; i < STAGES.length; i++) {
    const stage = STAGES[i]
    if (stage.key === status) return i
    if (stage.alternateKeys?.includes(status)) return i
  }
  // Handle error statuses - show at the stage they failed
  if (status === 'landsat_error') return 1
  if (status === 'prism_error') return 2
  if (status === 'nldas_error') return 3
  if (status === 'calculation_failed') return 4
  if (status === 'failed') return 0 // General failure at start
  return -1
}

// Helper to check if status is complete
const isStatusComplete = (status: ETMapStatus | undefined): boolean => {
  return status === 'calculation_complete'
}

// Helper to check if status is a failure
const isStatusFailed = (status: ETMapStatus | undefined): boolean => {
  return status === 'failed' || status === 'landsat_error' || status === 'prism_error' ||
         status === 'nldas_error' || status === 'calculation_failed'
}

// History entry type
interface ETMapHistoryEntry {
  requestId: string
  timestamp: string
  dateFrom: string
  dateTo: string
  geometry: GeoJSONGeometry[]
  status?: ETMapStatus
}

// localStorage keys
const LOCALSTORAGE_KEY = 'etmap_request_id'
const HISTORY_KEY = 'etmap_history'
const MAX_HISTORY_ENTRIES = 20

// History helpers
function loadHistory(): ETMapHistoryEntry[] {
  try {
    const stored = localStorage.getItem(HISTORY_KEY)
    if (stored) {
      return JSON.parse(stored) as ETMapHistoryEntry[]
    }
  } catch (e) {
    console.error('Failed to load ETMap history:', e)
  }
  return []
}

function saveHistory(history: ETMapHistoryEntry[]): void {
  try {
    // Keep only the most recent entries
    const trimmed = history.slice(0, MAX_HISTORY_ENTRIES)
    localStorage.setItem(HISTORY_KEY, JSON.stringify(trimmed))
  } catch (e) {
    console.error('Failed to save ETMap history:', e)
  }
}

function addToHistory(entry: ETMapHistoryEntry): ETMapHistoryEntry[] {
  const history = loadHistory()
  // Remove any existing entry with the same requestId
  const filtered = history.filter(h => h.requestId !== entry.requestId)
  // Add new entry at the beginning
  const updated = [entry, ...filtered]
  saveHistory(updated)
  return updated
}

function removeFromHistory(requestId: string): ETMapHistoryEntry[] {
  const history = loadHistory()
  const updated = history.filter(h => h.requestId !== requestId)
  saveHistory(updated)
  return updated
}

function formatTimestamp(isoString: string): string {
  try {
    const date = new Date(isoString)
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  } catch {
    return isoString
  }
}

/**
 * Calculate approximate area in square miles using Shoelace formula
 */
function calculateAreaSqMiles(polygons: GeoJSONGeometry[]): number {
  if (!polygons.length) return 0

  let totalArea = 0

  polygons.forEach(polygon => {
    if (!polygon.coordinates || !polygon.coordinates[0]) return

    const coords = polygon.coordinates[0] as [number, number][]
    if (coords.length < 3) return

    let area = 0
    for (let i = 0; i < coords.length; i++) {
      const j = (i + 1) % coords.length
      const [lng1, lat1] = coords[i]
      const [lng2, lat2] = coords[j]
      area += lng1 * lat2
      area -= lng2 * lat1
    }
    area = Math.abs(area) / 2

    const latMid = coords.reduce((sum, c) => sum + c[1], 0) / coords.length
    const latMiles = 69.0
    const lngMiles = 69.0 * Math.cos(latMid * Math.PI / 180)

    totalArea += area * latMiles * lngMiles
  })

  return totalArea
}

/**
 * Calculate MBR (Minimum Bounding Rectangle) for multiple polygons
 */
function calculateMultiPolygonBounds(polygons: GeoJSONGeometry[]): [[number, number], [number, number]] | null {
  if (!polygons || polygons.length === 0) return null

  let minLat = Infinity, maxLat = -Infinity
  let minLng = Infinity, maxLng = -Infinity

  polygons.forEach((polygon) => {
    if (polygon && polygon.coordinates && Array.isArray(polygon.coordinates) && polygon.coordinates[0]) {
      const coords = polygon.coordinates[0] as [number, number][]
      if (Array.isArray(coords)) {
        coords.forEach(([lng, lat]: [number, number]) => {
          minLat = Math.min(minLat, lat)
          maxLat = Math.max(maxLat, lat)
          minLng = Math.min(minLng, lng)
          maxLng = Math.max(maxLng, lng)
        })
      }
    }
  })

  return [[minLat, minLng], [maxLat, maxLng]]
}

const MAX_AREA_SQ_MILES = 60
const MAX_POLL_TIME_MS = 60 * 60 * 1000 // 1 hour
const INITIAL_POLL_DELAY = 2000 // 2 seconds
const MAX_POLL_DELAY = 30000 // 30 seconds

export function ETMapAnalysis() {
  const {
    selectedDateRange,
    drawnPolygons,
    setSelectedDateRange,
    setETMapImageOverlay,
    setETMapRequestId,
    setEtmapAnalysisResult,
    etmapRequestId,
    clearTrigger,
    drawnItems,
    setDrawnPolygons,
    mapInstance
  } = useMapStore()

  const { openChat, setPendingImage, setPendingMessage } = useChatStore()

  const handleAnalyzeWithAI = async () => {
    if (!requestId) return
    try {
      const blob = await apiClient.getETMapImage(requestId)
      const file = new File([blob], 'et-map.png', { type: 'image/png' })
      const reader = new FileReader()
      reader.onload = (e) => {
        const preview = e.target?.result as string
        setPendingImage(file, preview)
        setPendingMessage('Please evaluate my ET map and explain what the colors mean for my farm.')
        openChat()
      }
      reader.readAsDataURL(blob)
    } catch {
      toast.error('Could not load ET map image for AI analysis')
    }
  }

  const [jobStatus, setJobStatus] = useState<ETMapStatusResponse | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [requestId, setRequestId] = useState<string | null>(null)
  const [history, setHistory] = useState<ETMapHistoryEntry[]>([])
  const [isHistoryExpanded, setIsHistoryExpanded] = useState(false)
  const [isStatusExpanded, setIsStatusExpanded] = useState(false) // Collapsed by default
  const [selectedHistoryId, setSelectedHistoryId] = useState<string | null>(null)
  const [savedRequestId, setSavedRequestId] = useState<string | null>(null) // Saved but not yet loaded
  const [showDateRangeWarning, setShowDateRangeWarning] = useState(false) // Confirmation dialog for long date ranges

  const isPollingRef = useRef(false)
  const pollStartTimeRef = useRef<number | null>(null)
  const pollTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const currentGeometryRef = useRef<GeoJSONGeometry[]>([])

  // Load history on mount
  useEffect(() => {
    setHistory(loadHistory())
  }, [])

  // Check for shared ETMap link on mount
  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search)
    const sharedEtmapId = urlParams.get('etmapId')

    if (sharedEtmapId && !requestId && !savedRequestId) {
      loadSharedRequest(sharedEtmapId)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Keep currentGeometryRef in sync with drawnPolygons
  useEffect(() => {
    if (drawnPolygons && drawnPolygons.length > 0) {
      currentGeometryRef.current = drawnPolygons
    }
  }, [drawnPolygons])

  // Clear state when clearTrigger changes
  useEffect(() => {
    if (clearTrigger > 0) {
      setJobStatus(null)
      setRequestId(null)
      setETMapRequestId(null)
      setETMapImageOverlay(null, null)
      setEtmapAnalysisResult(null)
      setSelectedHistoryId(null)
      isPollingRef.current = false
      pollStartTimeRef.current = null
      if (pollTimeoutRef.current) {
        clearTimeout(pollTimeoutRef.current)
        pollTimeoutRef.current = null
      }
      localStorage.removeItem(LOCALSTORAGE_KEY)
    }
  }, [clearTrigger, setETMapImageOverlay, setETMapRequestId, setEtmapAnalysisResult])

  // Check for saved request_id from localStorage on mount (don't auto-load)
  useEffect(() => {
    const savedId = localStorage.getItem(LOCALSTORAGE_KEY)
    if (savedId && !requestId) {
      // Don't auto-load - just store it so user can choose to resume
      setSavedRequestId(savedId)
    }
  }, [requestId])

  // Start polling when requestId is set
  useEffect(() => {
    if (requestId && !isPollingRef.current) {
      pollStatus()
    }

    return () => {
      if (pollTimeoutRef.current) {
        clearTimeout(pollTimeoutRef.current)
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [requestId])

  const pollStatus = useCallback(async () => {
    if (!requestId || isPollingRef.current) return

    isPollingRef.current = true
    pollStartTimeRef.current = Date.now()
    let delay = INITIAL_POLL_DELAY

    const poll = async () => {
      if (!requestId) {
        isPollingRef.current = false
        return
      }

      // Check if we've exceeded max poll time
      if (pollStartTimeRef.current && Date.now() - pollStartTimeRef.current > MAX_POLL_TIME_MS) {
        isPollingRef.current = false
        toast.error('Polling timed out after 1 hour. The calculation may still be running - click Refresh to check status manually.', {
          duration: 10000
        })
        return
      }

      try {
        const status = await apiClient.getETMapStatus(requestId)
        setJobStatus(status)

        // Update history with latest status
        const currentHistory = loadHistory()
        const updatedHistory = currentHistory.map(h =>
          h.requestId === requestId ? { ...h, status: status.status } : h
        )
        saveHistory(updatedHistory)
        setHistory(updatedHistory)

        if (isStatusComplete(status.status)) {
          isPollingRef.current = false
          // Load and display PNG using the geometry from ref
          try {
            const imageBlob = await apiClient.getETMapImage(requestId)
            const imageUrl = URL.createObjectURL(imageBlob)
            const bounds = calculateMultiPolygonBounds(currentGeometryRef.current)
            if (bounds) {
              setETMapImageOverlay(imageUrl, bounds)
            }
            if (status.statistics) {
              setEtmapAnalysisResult(status.statistics)
            }
            toast.success('ET Map calculation complete!')
          } catch (imgError) {
            console.error('Failed to load ET Map image:', imgError)
            toast.error('ET Map complete but failed to load image')
          }
          return
        }

        if (isStatusFailed(status.status)) {
          isPollingRef.current = false
          toast.error(status.message || status.error_message || 'ET Map calculation failed')
          return
        }

        // Schedule next poll with exponential backoff
        delay = Math.min(delay * 1.5, MAX_POLL_DELAY)
        pollTimeoutRef.current = setTimeout(poll, delay)

      } catch (error) {
        console.error('Polling error:', error)
        // Continue polling despite errors
        delay = Math.min(delay * 1.5, MAX_POLL_DELAY)
        pollTimeoutRef.current = setTimeout(poll, delay)
      }
    }

    poll()
  }, [requestId, setETMapImageOverlay, setEtmapAnalysisResult])

  // Helper to calculate total days in date range
  const calculateTotalDays = () => {
    const fromDate = parseDate(selectedDateRange.from)
    const toDate = parseDate(selectedDateRange.to)
    return Math.ceil((toDate.getTime() - fromDate.getTime()) / (1000 * 60 * 60 * 24)) + 1
  }

  const handleSubmit = async () => {
    if (!drawnPolygons || drawnPolygons.length === 0) {
      toast.error('Please draw a farm area on the map first!')
      return
    }

    // Calculate and validate area
    const areaSqMiles = calculateAreaSqMiles(drawnPolygons)
    if (areaSqMiles > MAX_AREA_SQ_MILES) {
      toast.error(`Area too large (${areaSqMiles.toFixed(1)} sq mi). Maximum is ${MAX_AREA_SQ_MILES} sq miles.`)
      return
    }

    // Validate date range - max 31 days (1 month)
    const totalDays = calculateTotalDays()
    if (totalDays > 31) {
      toast.error('Date range too large. Maximum allowed is 1 month (31 days).')
      return
    }

    // Show warning for date ranges > 3 days
    if (totalDays > 3) {
      setShowDateRangeWarning(true)
      return
    }

    // Proceed with submission
    await submitETMapRequest()
  }

  const submitETMapRequest = async () => {
    setShowDateRangeWarning(false)

    // Clear previous results
    setJobStatus(null)
    setETMapImageOverlay(null, null)
    setSelectedHistoryId(null)
    isPollingRef.current = false
    if (pollTimeoutRef.current) {
      clearTimeout(pollTimeoutRef.current)
    }

    setIsSubmitting(true)

    try {
      // Create geometry - MultiPolygon if multiple polygons, single polygon if one
      let geometry: GeoJSONGeometry
      if (drawnPolygons.length === 1) {
        geometry = drawnPolygons[0]
      } else {
        geometry = {
          type: 'MultiPolygon',
          coordinates: drawnPolygons.map(polygon => polygon.coordinates)
        } as GeoJSONGeometry
      }

      const response = await apiClient.submitETMapRequest({
        geometry,
        dateFrom: selectedDateRange.from,
        dateTo: selectedDateRange.to
      })

      const newRequestId = response.request_id
      setRequestId(newRequestId)
      setETMapRequestId(newRequestId)
      localStorage.setItem(LOCALSTORAGE_KEY, newRequestId)

      // Save to history
      const historyEntry: ETMapHistoryEntry = {
        requestId: newRequestId,
        timestamp: new Date().toISOString(),
        dateFrom: selectedDateRange.from,
        dateTo: selectedDateRange.to,
        geometry: [...drawnPolygons],
        status: response.status || 'pending'
      }
      const updatedHistory = addToHistory(historyEntry)
      setHistory(updatedHistory)
      setSelectedHistoryId(newRequestId)

      // Update geometry ref
      currentGeometryRef.current = [...drawnPolygons]

      setJobStatus({
        request_id: newRequestId,
        status: response.status || 'pending'
      })

      toast.success('ET Map request submitted! Processing will begin shortly.')

    } catch (error: any) {
      console.error('Submit error:', error)
      toast.error(error.message || 'Failed to submit ET Map request')
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleSelectHistoryEntry = async (entry: ETMapHistoryEntry) => {
    // Stop any existing polling
    isPollingRef.current = false
    if (pollTimeoutRef.current) {
      clearTimeout(pollTimeoutRef.current)
      pollTimeoutRef.current = null
    }

    // Clear current map state
    setETMapImageOverlay(null, null)
    setJobStatus(null)

    // Clear drawn items on map and add the history geometry
    if (drawnItems) {
      drawnItems.clearLayers()

      // Dynamically import Leaflet (only runs on client)
      const L = (await import('leaflet')).default

      // Draw the geometry from history
      entry.geometry.forEach(geom => {
        if (geom.coordinates && geom.coordinates[0]) {
          const coords = geom.coordinates[0] as [number, number][]
          // Convert [lng, lat] to [lat, lng] for Leaflet
          const latLngs = coords.map(([lng, lat]) => [lat, lng] as [number, number])
          const polygon = L.polygon(latLngs, {
            color: '#3388ff',
            weight: 2,
            fillOpacity: 0.2
          })
          drawnItems.addLayer(polygon)
        }
      })
    }

    // Update store with the geometry
    setDrawnPolygons(entry.geometry)
    currentGeometryRef.current = entry.geometry

    // Update date range
    setSelectedDateRange({
      from: entry.dateFrom,
      to: entry.dateTo
    })

    // Set the request ID
    setSelectedHistoryId(entry.requestId)
    localStorage.setItem(LOCALSTORAGE_KEY, entry.requestId)

    // Zoom to the geometry
    if (mapInstance && entry.geometry.length > 0) {
      const bounds = calculateMultiPolygonBounds(entry.geometry)
      if (bounds) {
        mapInstance.fitBounds(bounds, { padding: [50, 50] })
      }
    }

    // Immediately fetch the current status
    try {
      const status = await apiClient.getETMapStatus(entry.requestId)
      setJobStatus(status)

      // Update history with latest status
      const currentHistory = loadHistory()
      const updatedHistory = currentHistory.map(h =>
        h.requestId === entry.requestId ? { ...h, status: status.status } : h
      )
      saveHistory(updatedHistory)
      setHistory(updatedHistory)

      // If completed, load the image
      if (isStatusComplete(status.status)) {
        try {
          const imageBlob = await apiClient.getETMapImage(entry.requestId)
          const imageUrl = URL.createObjectURL(imageBlob)
          const bounds = calculateMultiPolygonBounds(entry.geometry)
          if (bounds) {
            setETMapImageOverlay(imageUrl, bounds)
          }
          toast.success('ET Map loaded from history!')
        } catch (imgError) {
          console.error('Failed to load ET Map image:', imgError)
        }
      } else if (isStatusFailed(status.status)) {
        toast.error(status.message || status.error_message || 'This request failed')
      } else {
        // Still processing - start polling
        toast.success('Loaded history entry - still processing...')
        setRequestId(entry.requestId)
        setETMapRequestId(entry.requestId)
      }
    } catch (error: any) {
      console.error('Failed to fetch status:', error)
      toast.error('Failed to fetch status for this request')
      // Still set the request ID so user can try refresh
      setRequestId(entry.requestId)
      setETMapRequestId(entry.requestId)
    }
  }

  const handleDeleteHistoryEntry = (e: React.MouseEvent, requestId: string) => {
    e.stopPropagation()
    const updated = removeFromHistory(requestId)
    setHistory(updated)

    if (selectedHistoryId === requestId) {
      setSelectedHistoryId(null)
    }

    toast.success('History entry removed')
  }

  const handleManualRefresh = async () => {
    if (!requestId) return

    setIsRefreshing(true)
    try {
      const status = await apiClient.getETMapStatus(requestId)
      setJobStatus(status)

      // Update history with latest status
      const currentHistory = loadHistory()
      const updatedHistory = currentHistory.map(h =>
        h.requestId === requestId ? { ...h, status: status.status } : h
      )
      saveHistory(updatedHistory)
      setHistory(updatedHistory)

      if (isStatusComplete(status.status)) {
        // Load PNG if not already loaded
        try {
          const imageBlob = await apiClient.getETMapImage(requestId)
          const imageUrl = URL.createObjectURL(imageBlob)
          const bounds = calculateMultiPolygonBounds(currentGeometryRef.current)
          if (bounds) {
            setETMapImageOverlay(imageUrl, bounds)
          }
          toast.success('ET Map loaded!')
        } catch (imgError) {
          console.error('Failed to load ET Map image:', imgError)
        }
      } else if (isStatusFailed(status.status)) {
        toast.error(status.message || status.error_message || 'ET Map calculation failed')
      } else {
        toast.success('Status updated')
      }
    } catch (error: any) {
      toast.error(error.message || 'Failed to refresh status')
    } finally {
      setIsRefreshing(false)
    }
  }

  const handleDownloadTif = async () => {
    if (!requestId) return

    const loadingToast = toast.loading('Preparing GeoTIFF download...')
    try {
      const blob = await apiClient.downloadETMapTif(requestId)

      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `etmap-${requestId}.tif`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      URL.revokeObjectURL(url)

      toast.dismiss(loadingToast)
      toast.success('GeoTIFF downloaded!')
    } catch (error: any) {
      toast.dismiss(loadingToast)
      toast.error(error.message || 'Failed to download GeoTIFF')
    }
  }

  const handleDownloadPng = async () => {
    if (!requestId) return

    const loadingToast = toast.loading('Preparing PNG download...')
    try {
      const blob = await apiClient.getETMapImage(requestId)

      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `etmap-${requestId}.png`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      URL.revokeObjectURL(url)

      toast.dismiss(loadingToast)
      toast.success('PNG downloaded!')
    } catch (error: any) {
      toast.dismiss(loadingToast)
      toast.error(error.message || 'Failed to download PNG')
    }
  }

  const handleDateChange = (field: 'from' | 'to', value: string) => {
    setSelectedDateRange({
      ...selectedDateRange,
      [field]: value
    })
  }

  const handleClearRequest = () => {
    setJobStatus(null)
    setRequestId(null)
    setETMapRequestId(null)
    setETMapImageOverlay(null, null)
    setSelectedHistoryId(null)
    setSavedRequestId(null)
    isPollingRef.current = false
    if (pollTimeoutRef.current) {
      clearTimeout(pollTimeoutRef.current)
    }
    localStorage.removeItem(LOCALSTORAGE_KEY)
  }

  const handleLoadSavedRequest = async () => {
    if (!savedRequestId) return

    setIsRefreshing(true)
    try {
      // Fetch the current status
      const status = await apiClient.getETMapStatus(savedRequestId)
      setJobStatus(status)
      setRequestId(savedRequestId)
      setETMapRequestId(savedRequestId)
      setSavedRequestId(null) // Clear the saved state since it's now active

      // Try to find geometry from history and restore map state
      const historyEntry = history.find(h => h.requestId === savedRequestId)
      if (historyEntry) {
        currentGeometryRef.current = historyEntry.geometry
        setSelectedHistoryId(savedRequestId)

        // Draw the geometry on the map (same as handleSelectHistoryEntry)
        if (drawnItems) {
          drawnItems.clearLayers()

          // Dynamically import Leaflet (only runs on client)
          const L = (await import('leaflet')).default

          // Draw the geometry from history
          historyEntry.geometry.forEach(geom => {
            if (geom.coordinates && geom.coordinates[0]) {
              const coords = geom.coordinates[0] as [number, number][]
              // Convert [lng, lat] to [lat, lng] for Leaflet
              const latLngs = coords.map(([lng, lat]) => [lat, lng] as [number, number])
              const polygon = L.polygon(latLngs, {
                color: '#3388ff',
                weight: 2,
                fillOpacity: 0.2
              })
              drawnItems.addLayer(polygon)
            }
          })
        }

        // Update store with the geometry
        setDrawnPolygons(historyEntry.geometry)

        // Update date range
        setSelectedDateRange({
          from: historyEntry.dateFrom,
          to: historyEntry.dateTo
        })

        // Zoom to the geometry (with safety check for map readiness)
        if (mapInstance && historyEntry.geometry.length > 0) {
          const bounds = calculateMultiPolygonBounds(historyEntry.geometry)
          if (bounds) {
            // Use setTimeout to ensure map is fully ready
            setTimeout(() => {
              try {
                mapInstance.fitBounds(bounds, { padding: [50, 50] })
              } catch (e) {
                console.warn('Map not ready for fitBounds:', e)
              }
            }, 100)
          }
        }
      }

      // If completed, load the image
      if (isStatusComplete(status.status)) {
        try {
          const imageBlob = await apiClient.getETMapImage(savedRequestId)
          const imageUrl = URL.createObjectURL(imageBlob)
          const bounds = calculateMultiPolygonBounds(currentGeometryRef.current)
          if (bounds) {
            setETMapImageOverlay(imageUrl, bounds)
          }
          toast.success('ET Map loaded!')
        } catch (imgError) {
          console.error('Failed to load ET Map image:', imgError)
        }
      } else if (isStatusFailed(status.status)) {
        toast.error(status.message || status.error_message || 'This request failed')
      } else {
        toast.success('Resuming request - processing in progress...')
      }
    } catch (error: any) {
      console.error('Failed to load saved request:', error)
      toast.error('Failed to load saved request')
      // Clear the invalid saved request
      setSavedRequestId(null)
      localStorage.removeItem(LOCALSTORAGE_KEY)
    } finally {
      setIsRefreshing(false)
    }
  }

  const handleDismissSavedRequest = () => {
    setSavedRequestId(null)
    localStorage.removeItem(LOCALSTORAGE_KEY)
  }

  const loadSharedRequest = async (sharedId: string) => {
    setIsRefreshing(true)
    try {
      const status = await apiClient.getETMapStatus(sharedId)
      setJobStatus(status)
      setRequestId(sharedId)
      setETMapRequestId(sharedId)

      // Restore geometry from backend response (cross-device compatible)
      if (status.request?.geometry) {
        const geometry = status.request.geometry
        const geometries: GeoJSONGeometry[] = geometry.type === 'MultiPolygon'
          ? (geometry.coordinates as number[][][][]).map(coords => ({ type: 'Polygon' as const, coordinates: coords }))
          : [geometry]

        currentGeometryRef.current = geometries
        setDrawnPolygons(geometries)

        // Draw on map
        if (drawnItems) {
          drawnItems.clearLayers()
          const L = (await import('leaflet')).default
          geometries.forEach(geom => {
            if (geom.coordinates && geom.coordinates[0]) {
              const coords = geom.coordinates[0] as [number, number][]
              const latLngs = coords.map(([lng, lat]) => [lat, lng] as [number, number])
              const polygon = L.polygon(latLngs, { color: '#3388ff', weight: 2, fillOpacity: 0.2 })
              drawnItems.addLayer(polygon)
            }
          })
        }

        // Update date range
        if (status.request.date_from && status.request.date_to) {
          setSelectedDateRange({ from: status.request.date_from, to: status.request.date_to })
        }

        // Zoom to geometry
        if (mapInstance && geometries.length > 0) {
          const bounds = calculateMultiPolygonBounds(geometries)
          if (bounds) {
            setTimeout(() => {
              try {
                mapInstance.fitBounds(bounds, { padding: [50, 50] })
              } catch (e) {
                console.warn('Map not ready:', e)
              }
            }, 100)
          }
        }
      }

      // Load image if complete
      if (isStatusComplete(status.status)) {
        try {
          const imageBlob = await apiClient.getETMapImage(sharedId)
          const imageUrl = URL.createObjectURL(imageBlob)
          const bounds = calculateMultiPolygonBounds(currentGeometryRef.current)
          if (bounds) {
            setETMapImageOverlay(imageUrl, bounds)
          }
          toast.success('Shared ET Map loaded!')
        } catch (imgError) {
          console.error('Failed to load ET Map image:', imgError)
        }
      } else if (isStatusFailed(status.status)) {
        toast.error(status.error_message || 'This ET Map request failed')
      } else {
        toast.success('Loading shared request - still processing...')
      }
    } catch (error: any) {
      console.error('Failed to load shared request:', error)
      toast.error('Failed to load shared ET Map')
      // Clear invalid param from URL
      const url = new URL(window.location.href)
      url.searchParams.delete('etmapId')
      window.history.replaceState({}, '', url.toString())
    } finally {
      setIsRefreshing(false)
    }
  }

  const handleShare = async () => {
    if (!requestId) return

    const url = new URL(window.location.origin)
    url.searchParams.set('tab', 'etmap')
    url.searchParams.set('etmapId', requestId)

    // Include map position for proper zoom on shared link
    if (mapInstance) {
      const center = mapInstance.getCenter()
      const zoom = mapInstance.getZoom()
      url.searchParams.set('lat', center.lat.toFixed(6))
      url.searchParams.set('lng', center.lng.toFixed(6))
      url.searchParams.set('zoom', zoom.toString())
    }

    // Include date range
    url.searchParams.set('dateFrom', selectedDateRange.from)
    url.searchParams.set('dateTo', selectedDateRange.to)

    try {
      await navigator.clipboard.writeText(url.toString())
      toast.success('Share link copied to clipboard!')
    } catch {
      toast.error('Failed to copy link')
    }
  }

  // Calculate current stage index for display
  const currentStageIdx = getStageIndex(jobStatus?.status)
  const isFailed = isStatusFailed(jobStatus?.status)
  const isCompleted = isStatusComplete(jobStatus?.status)

  return (
    <div className="p-4 space-y-6" data-tutorial="etmap-panel">
      {/* Parameters Section */}
      <div className="space-y-4">
        <h3 className="font-medium text-foreground flex items-center">
          <Droplets className="h-4 w-4 mr-2" />
          Evapotranspiration Map
        </h3>

        <div className="text-sm text-muted-foreground mb-4">
          Calculate evapotranspiration (ET) for your farm area using satellite data and meteorological models.
        </div>

        <div className="space-y-4" data-tutorial="etmap-date-selector">
          <div className="flex items-baseline gap-4">
            <label className="text-sm font-medium text-foreground w-20 shrink-0">
              Start Date
            </label>
            <Input
              type="date"
              value={selectedDateRange.from}
              onChange={(e) => handleDateChange('from', e.target.value)}
              max={selectedDateRange.to}
              className="flex-1"
              style={{ height: '40px', padding: '8px 12px', fontSize: '14px', lineHeight: '20px' }}
            />
          </div>

          <div className="flex items-baseline gap-4">
            <label className="text-sm font-medium text-foreground w-20 shrink-0">
              End Date
            </label>
            <Input
              type="date"
              value={selectedDateRange.to}
              onChange={(e) => handleDateChange('to', e.target.value)}
              min={selectedDateRange.from}
              className="flex-1"
              style={{ height: '40px', padding: '8px 12px', fontSize: '14px', lineHeight: '20px' }}
            />
          </div>
        </div>

        {/* Date range info */}
        <div className="text-sm bg-muted/30 p-3 rounded-lg border">
          <p className="font-medium text-foreground">
            Selected time period:
          </p>
          <p className="text-muted-foreground mt-1">
            From {formatDateString(selectedDateRange.from)} to {formatDateString(selectedDateRange.to)}
          </p>
          <p className="text-muted-foreground">
            Total days: {(() => {
              const fromDate = parseDate(selectedDateRange.from)
              const toDate = parseDate(selectedDateRange.to)
              return Math.ceil((toDate.getTime() - fromDate.getTime()) / (1000 * 60 * 60 * 24)) + 1
            })()}
          </p>
        </div>

      </div>

      {/* Submit Button */}
      <div className="space-y-3">
        <Button
          onClick={handleSubmit}
          disabled={!drawnPolygons || drawnPolygons.length === 0 || isSubmitting || calculateAreaSqMiles(drawnPolygons) > MAX_AREA_SQ_MILES}
          className="w-full"
          data-tutorial="etmap-calculate-button"
        >
          {isSubmitting ? (
            <>
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
              Submitting...
            </>
          ) : (
            <>
              <Droplets className="h-4 w-4 mr-2" />
              Calculate ET Map
            </>
          )}
        </Button>

        {/* Warning Messages */}
        {(!drawnPolygons || drawnPolygons.length === 0) && (
          <div className="mt-2 flex items-start space-x-2 text-amber-600 dark:text-amber-400 text-sm">
            <AlertCircle className="h-4 w-4 mt-0.5 flex-shrink-0" />
            <p>Draw a polygon on the map to define your farm area</p>
          </div>
        )}
      </div>

      {/* Resume Previous Request Prompt */}
      {savedRequestId && !requestId && (
        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4 space-y-3">
          <div className="flex items-start gap-2">
            <Clock className="h-5 w-5 text-blue-500 mt-0.5 flex-shrink-0" />
            <div className="flex-1">
              <p className="text-sm font-medium text-blue-800 dark:text-blue-200">
                Previous Request Found
              </p>
              <p className="text-xs text-blue-600 dark:text-blue-400 mt-1">
                You have an unfinished ET Map request. Would you like to load its status?
              </p>
              <p className="text-xs text-muted-foreground mt-1 break-all">
                ID: {savedRequestId}
              </p>
            </div>
          </div>
          <div className="flex gap-2">
            <Button
              size="sm"
              onClick={handleLoadSavedRequest}
              disabled={isRefreshing}
              className="flex-1"
            >
              {isRefreshing ? (
                <>
                  <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                  Loading...
                </>
              ) : (
                <>
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Load Status
                </>
              )}
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={handleDismissSavedRequest}
              disabled={isRefreshing}
            >
              Dismiss
            </Button>
          </div>
        </div>
      )}

      {/* Status Section - Collapsible */}
      {(jobStatus || requestId) && (
        <div className="space-y-4 border-t pt-4" data-tutorial="etmap-status">
          {/* Collapsible Header */}
          <button
            onClick={() => setIsStatusExpanded(!isStatusExpanded)}
            className="flex items-center justify-between w-full text-left"
          >
            <div className="flex items-center gap-3 flex-1">
              <h4 className="font-medium text-foreground">Processing Status</h4>
              {/* Current status badge (always visible) */}
              <div className={cn(
                "flex items-center gap-1.5 px-2 py-1 rounded-full text-xs font-medium",
                isCompleted && "bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-300",
                isFailed && "bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300",
                !isCompleted && !isFailed && "bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300"
              )}>
                {isCompleted && <CheckCircle className="h-3 w-3" />}
                {isFailed && <XCircle className="h-3 w-3" />}
                {!isCompleted && !isFailed && (
                  <div className="h-3 w-3 border-2 border-blue-300 border-t-blue-600 rounded-full animate-spin" />
                )}
                <span>
                  {isCompleted ? 'Completed' : isFailed ? 'Failed' : STAGES[currentStageIdx]?.label || 'Processing'}
                </span>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={(e) => {
                  e.stopPropagation()
                  handleClearRequest()
                }}
                className="text-xs"
              >
                Clear
              </Button>
              {isStatusExpanded ? (
                <ChevronUp className="h-4 w-4 text-muted-foreground" />
              ) : (
                <ChevronDown className="h-4 w-4 text-muted-foreground" />
              )}
            </div>
          </button>

          {/* Expanded Content */}
          {isStatusExpanded && (
            <>
              {/* Stage indicators */}
              <div className="space-y-1">
                {STAGES.map((stage, idx) => {
                  const isPast = idx < currentStageIdx || isCompleted
                  const isCurrent = idx === currentStageIdx && !isCompleted
                  const isFuture = idx > currentStageIdx && !isCompleted
                  const isCompletedStage = stage.key === 'calculation_complete' && isCompleted

                  return (
                    <div
                      key={stage.key}
                      className={cn(
                        "flex items-center gap-3 p-2.5 rounded-lg text-sm transition-all duration-200",
                        isCurrent && !isFailed && "bg-blue-100 dark:bg-blue-900/50 font-semibold text-blue-800 dark:text-blue-200 border-2 border-blue-400 dark:border-blue-500 shadow-md",
                        isPast && !isCompletedStage && "text-green-600 dark:text-green-400 bg-green-50 dark:bg-green-900/20",
                        isCompletedStage && "bg-green-100 dark:bg-green-900/40 font-semibold text-green-700 dark:text-green-300 border-2 border-green-400 dark:border-green-500",
                        isFuture && "text-muted-foreground opacity-40",
                        isFailed && isCurrent && "bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400 border-2 border-red-400 font-semibold"
                      )}
                    >
                      {isPast && !isCompletedStage && <CheckCircle className="h-5 w-5 text-green-500" />}
                      {isCompletedStage && <CheckCircle className="h-5 w-5 text-green-500 animate-bounce" />}
                      {isCurrent && !isFailed && (
                        <div className="h-5 w-5 border-2 border-blue-300 dark:border-blue-600 border-t-blue-600 dark:border-t-blue-300 rounded-full animate-spin" />
                      )}
                      {isCurrent && isFailed && <XCircle className="h-5 w-5 text-red-500" />}
                      {isFuture && <div className="h-5 w-5 rounded-full border-2 border-muted-foreground/30" />}
                      <span>{stage.label}</span>
                      {isCurrent && !isFailed && (
                        <span className="ml-auto text-xs text-blue-600 dark:text-blue-300 font-normal">In Progress...</span>
                      )}
                    </div>
                  )
                })}
              </div>

              {/* Error message */}
              {isFailed && jobStatus?.message && (
                <div className="text-sm text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/20 p-3 rounded">
                  {jobStatus.message}
                </div>
              )}

              {/* Request ID */}
              {requestId && (
                <div className="text-xs text-muted-foreground">
                  Request ID: {requestId}
                </div>
              )}
            </>
          )}

          {/* Statistics Section - Always visible when completed */}
          {isCompleted && jobStatus?.statistics && (
            <div className="bg-muted/30 rounded-lg p-4 border" data-tutorial="etmap-statistics">
              <h5 className="font-medium text-foreground flex items-center gap-2 mb-3">
                <BarChart3 className="h-4 w-4" />
                ET Map Statistics
              </h5>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Band:</span>
                  <span className="font-medium">{jobStatus.statistics.band_name ?? 'N/A'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Coverage:</span>
                  <span className="font-medium">{jobStatus.statistics.coverage_percent != null ? jobStatus.statistics.coverage_percent.toFixed(2) : 'N/A'}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Min:</span>
                  <span className="font-medium">{jobStatus.statistics.min != null ? jobStatus.statistics.min.toFixed(2) : 'N/A'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Max:</span>
                  <span className="font-medium">{jobStatus.statistics.max != null ? jobStatus.statistics.max.toFixed(2) : 'N/A'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Mean:</span>
                  <span className="font-medium">{jobStatus.statistics.mean != null ? jobStatus.statistics.mean.toFixed(2) : 'N/A'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Median:</span>
                  <span className="font-medium">{jobStatus.statistics.median != null ? jobStatus.statistics.median.toFixed(2) : 'N/A'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Std Dev:</span>
                  <span className="font-medium">{jobStatus.statistics.std != null ? jobStatus.statistics.std.toFixed(2) : 'N/A'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Valid Pixels:</span>
                  <span className="font-medium">{jobStatus.statistics.valid_pixels != null ? jobStatus.statistics.valid_pixels.toLocaleString() : 'N/A'}</span>
                </div>
              </div>
            </div>
          )}

          {/* Action buttons */}
          <div className="flex gap-2 flex-wrap" data-tutorial="etmap-downloads">
            <Button
              variant="outline"
              size="sm"
              onClick={handleManualRefresh}
              disabled={isRefreshing || !requestId}
            >
              <RefreshCw className={cn("h-4 w-4 mr-2", isRefreshing && "animate-spin")} />
              Refresh
            </Button>

            <Button
              variant="outline"
              size="sm"
              onClick={handleShare}
              disabled={!requestId}
              data-tutorial="etmap-share-button"
            >
              <Share2 className="h-4 w-4 mr-2" />
              Share
            </Button>

            {isCompleted && requestId && (
              <Button
                id="etmap-analyze-ai-btn"
                variant="outline"
                size="sm"
                onClick={handleAnalyzeWithAI}
              >
                <Bot className="h-4 w-4 mr-2" />
                Analyze with AI
              </Button>
            )}

            <Button
              variant="outline"
              size="sm"
              onClick={handleDownloadPng}
              disabled={!isCompleted}
            >
              <Download className="h-4 w-4 mr-2" />
              PNG
            </Button>

            <Button
              variant="outline"
              size="sm"
              onClick={handleDownloadTif}
              disabled={!isCompleted}
            >
              <Download className="h-4 w-4 mr-2" />
              TIF
            </Button>
          </div>
        </div>
      )}

      {/* History Section */}
      {history.length > 0 && (
        <div className="border-t pt-4" data-tutorial="etmap-history">
          <button
            onClick={() => setIsHistoryExpanded(!isHistoryExpanded)}
            className="flex items-center justify-between w-full text-left"
          >
            <h4 className="font-medium text-foreground flex items-center">
              <History className="h-4 w-4 mr-2" />
              Request History ({history.length})
            </h4>
            {isHistoryExpanded ? (
              <ChevronUp className="h-4 w-4 text-muted-foreground" />
            ) : (
              <ChevronDown className="h-4 w-4 text-muted-foreground" />
            )}
          </button>

          {isHistoryExpanded && (
            <div className="mt-3 space-y-2 max-h-64 overflow-y-auto">
              {history.map((entry) => (
                <div
                  key={entry.requestId}
                  onClick={() => handleSelectHistoryEntry(entry)}
                  className={cn(
                    "p-3 rounded-lg border cursor-pointer transition-all hover:bg-muted/50",
                    selectedHistoryId === entry.requestId && "border-primary bg-primary/5"
                  )}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium text-foreground">
                        {formatTimestamp(entry.timestamp)}
                      </div>
                      <div className="text-xs text-muted-foreground mt-1">
                        {entry.dateFrom} to {entry.dateTo}
                      </div>
                      <div className="text-xs text-muted-foreground break-all">
                        ID: {entry.requestId}
                      </div>
                      {entry.status && (
                        <div className={cn(
                          "text-xs mt-1 inline-flex items-center gap-1 px-1.5 py-0.5 rounded",
                          isStatusComplete(entry.status) && "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400",
                          isStatusFailed(entry.status) && "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400",
                          !isStatusComplete(entry.status) && !isStatusFailed(entry.status) && "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400"
                        )}>
                          {isStatusComplete(entry.status) && <CheckCircle className="h-3 w-3" />}
                          {isStatusFailed(entry.status) && <XCircle className="h-3 w-3" />}
                          {!isStatusComplete(entry.status) && !isStatusFailed(entry.status) && <Clock className="h-3 w-3" />}
                          {isStatusComplete(entry.status) ? 'Completed' : isStatusFailed(entry.status) ? 'Failed' : 'Processing'}
                        </div>
                      )}
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={(e) => handleDeleteHistoryEntry(e, entry.requestId)}
                      className="h-8 w-8 p-0 text-muted-foreground hover:text-destructive"
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Info Section */}
      <div className="text-xs text-muted-foreground border-t pt-4 space-y-2">
        <p><strong>Data Sources:</strong></p>
        <ul className="list-disc list-inside space-y-1 ml-2">
          <li>Landsat satellite imagery (vegetation indices)</li>
          <li>PRISM climate data (temperature, precipitation)</li>
          <li>NLDAS meteorological data (hourly)</li>
        </ul>
        <p className="mt-2">
          <strong>Note:</strong> Processing may take several minutes depending on the area size and date range.
        </p>
      </div>

      {/* Date Range Warning Dialog */}
      {showDateRangeWarning && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-background border rounded-lg shadow-lg p-6 max-w-md mx-4 space-y-4">
            <div className="flex items-start gap-3">
              <AlertCircle className="h-6 w-6 text-amber-500 flex-shrink-0 mt-0.5" />
              <div>
                <h3 className="font-semibold text-foreground text-lg">Processing Time Notice</h3>
                <p className="text-muted-foreground mt-2">
                  You have selected a date range of <strong>{calculateTotalDays()} days</strong>.
                  Processing may take several minutes to complete.
                </p>
                <p className="text-muted-foreground mt-2">
                  Please check back after some time for results.
                </p>
              </div>
            </div>
            <div className="flex gap-3 justify-end pt-2">
              <Button
                variant="outline"
                onClick={() => setShowDateRangeWarning(false)}
              >
                Cancel
              </Button>
              <Button
                onClick={submitETMapRequest}
              >
                Ok, I'll Wait
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
