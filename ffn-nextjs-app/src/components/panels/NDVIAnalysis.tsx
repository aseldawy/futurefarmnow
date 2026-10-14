'use client'

import { useState, useMemo, useEffect, useCallback, useRef } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select, SelectOption } from '@/components/ui/select'
import { useMapStore } from '@/lib/stores/mapStore'
import { useMutation } from '@tanstack/react-query'
import { apiClient } from '@/lib/api/client'
import { TimeSeriesChart } from '@/components/visualizations/TimeSeriesChart'
import { NDVILegend } from '@/components/visualizations/NDVILegend'
import { AlertCircle, Calendar, Download, GitCompare } from 'lucide-react'
import { toast } from 'react-hot-toast'
import { formatDateString, parseDate } from '@/lib/utils'
import { GeoJSONGeometry } from '@/lib/types/api'

export function NDVIAnalysis() {
  const {
    selectedDateRange,
    drawnPolygons,
    currentZoom,
    currentBounds,
    selectedDataset,
    setSelectedDateRange,
    setSoilImageOverlay,
    setFarmlandGeoJSON,
    setNDVIImageOverlay,
    clearTrigger,
    setNdviAnalysisResult
  } = useMapStore()

  const [results, setResults] = useState<any>(null)
  const [comparisonResults, setComparisonResults] = useState<any>(null)
  const [analyzingType, setAnalyzingType] = useState<'polygon' | 'farmland' | null>(null)
  const [farmlandResults, setFarmlandResults] = useState<any>(null) // Store individual farmland data
  const [comparisonYear, setComparisonYear] = useState<string>('')
  const [isComparison, setIsComparison] = useState(false)

  // Time slider states
  const [imageMetadata, setImageMetadata] = useState<any>(null)
  const [currentSliderDate, setCurrentSliderDate] = useState<string>('')
  const [currentImageUrl, setCurrentImageUrl] = useState<string>('')
  const [imageCache, setImageCache] = useState<Map<string, string>>(new Map())
  const [isLoadingImage, setIsLoadingImage] = useState(false)
  const [showTimeSlider, setShowTimeSlider] = useState(false)
  const [isChangingImage, setIsChangingImage] = useState(false)
  const [isInitialLoading, setIsInitialLoading] = useState(false)

  // Data source selection
  const [dataSource, setDataSource] = useState<string>('ndvi')

  // Calculate MBR (Minimum Bounding Rectangle) for multiple polygons
  const calculateMultiPolygonBounds = useCallback((polygons: GeoJSONGeometry[]): [[number, number], [number, number]] | null => {
    if (!polygons || polygons.length === 0) return null

    let minLat = Infinity, maxLat = -Infinity
    let minLng = Infinity, maxLng = -Infinity

    polygons.forEach((polygon) => {
      if (polygon && polygon.coordinates && Array.isArray(polygon.coordinates) && polygon.coordinates[0]) {
        const coords = polygon.coordinates[0] as [number, number][] // First ring of polygon
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
  }, [])

  // Clear results and cache when data source changes
  useEffect(() => {
    // Clear all NDVI results and images when switching data sources
    setResults(null)
    setComparisonResults(null)
    setFarmlandResults(null)
    setIsComparison(false)
    setImageMetadata(null)
    setCurrentSliderDate('')
    setCurrentImageUrl('')
    setImageCache(new Map())
    setShowTimeSlider(false)
    setNDVIImageOverlay(null, null)
    setAnalyzingType(null)
    setIsInitialLoading(false)
  }, [dataSource, setNDVIImageOverlay])

  // Clear state when clearTrigger changes
  useEffect(() => {
    if (clearTrigger > 0) {
      setResults(null)
      setComparisonResults(null)
      setAnalyzingType(null)
      setFarmlandResults(null)
      setComparisonYear('')
      setIsComparison(false)
      setImageMetadata(null)
      setCurrentSliderDate('')
      setCurrentImageUrl('')
      setImageCache(new Map())
      setIsLoadingImage(false)
      setShowTimeSlider(false)
      setNDVIImageOverlay(null, null)
      setIsInitialLoading(false)
    }
  }, [clearTrigger, setNDVIImageOverlay])

  // Set default comparison year when results are loaded
  useEffect(() => {
    if (results && results.length > 0 && !comparisonYear) {
      const currentYear = parseDate(selectedDateRange.from).getFullYear()
      const previousYear = currentYear - 1
      setComparisonYear(previousYear.toString())
    }
  }, [results, selectedDateRange.from, comparisonYear])

  // Function to load a single image for a specific date
  const loadImageForDate = useCallback(async (date: string, showOnMap: boolean = false) => {
    if (!drawnPolygons || drawnPolygons.length === 0) return null

    // Create geometry - MultiPolygon if multiple polygons, single polygon if one
    let geometry: GeoJSONGeometry;
    if (drawnPolygons.length === 1) {
      geometry = drawnPolygons[0];
    } else {
      // Combine multiple polygons into a MultiPolygon
      geometry = {
        type: 'MultiPolygon',
        coordinates: drawnPolygons.map(polygon => polygon.coordinates)
      } as GeoJSONGeometry;
    }

    try {
      const imageBlob = await apiClient.getNDVIImage({
        date,
        geometry,
        source: dataSource
      })

      // Convert blob to data URL
      const imageUrl = URL.createObjectURL(imageBlob)
      const dataUrl = await new Promise<string>((resolve) => {
        const img = new Image()
        img.onload = () => {
          const canvas = document.createElement('canvas')
          canvas.width = img.width
          canvas.height = img.height
          const ctx = canvas.getContext('2d')!
          ctx.drawImage(img, 0, 0)
          resolve(canvas.toDataURL('image/png'))
        }
        img.src = imageUrl
      })

      // Clean up blob URL
      URL.revokeObjectURL(imageUrl)

      // Cache the image
      setImageCache(prev => new Map(prev).set(date, dataUrl))

      // Show on map if requested
      if (showOnMap) {
        setCurrentSliderDate(date)
        setCurrentImageUrl(dataUrl)
        setShowTimeSlider(true)

        // Calculate bounds for multiple polygons and show overlay
        const bounds = calculateMultiPolygonBounds(drawnPolygons)
        if (bounds) {
          const timestampedUrl = `${dataUrl}#${Date.now()}`
          setNDVIImageOverlay(timestampedUrl, bounds)
        }
      }

      return dataUrl
    } catch (error) {
      console.error(`Failed to load image for date ${date}:`, error)
      return null
    }
  }, [drawnPolygons, dataSource, setImageCache, setCurrentSliderDate, setCurrentImageUrl, setShowTimeSlider, setNDVIImageOverlay, calculateMultiPolygonBounds])

  // Function to load remaining images asynchronously in parallel
  const loadRemainingImagesAsync = async (allDates: string[], datesToLoad: string[]) => {
    if (!drawnPolygons || drawnPolygons.length === 0 || datesToLoad.length === 0) return

    // Set up metadata with ALL available dates immediately (including first date already loaded)
    setImageMetadata({
      available_dates: allDates,
      token: 'simplified',
      polygon_hash: 'simplified',
      statistics_per_date: {}
    })

    // Load remaining images in parallel and wait for completion
    const promises = datesToLoad.map(date => loadImageForDate(date, false)) // No map show for parallel loading
    await Promise.all(promises)

    // All images loaded
    setIsInitialLoading(false)
    toast.success('All satellite images loaded!')
  }

  const ndviAnalysisMutation = useMutation({
    mutationFn: async (params: any) => {
      if (params.isFarmlandAnalysis) {
        return await apiClient.getNDVIForRegion('farmland', params.from, params.to, params.bbox, params.source)
      }
      return await apiClient.getNDVIForPolygon(params)
    },
    onSuccess: async (data: any) => {
      const resultsData = data.results || data

      // Handle different result structures
      if (Array.isArray(resultsData) && resultsData.length > 0 && resultsData[0].objectid) {
        // Farmland analysis: array of {objectid, results: [...]} 
        // Store individual farmland data for popups
        setFarmlandResults(resultsData)

        // Flatten all time series data from all farmlands for charts/tables
        const allTimePoints: any[] = []
        resultsData.forEach((farmland: any) => {
          if (farmland.results && Array.isArray(farmland.results)) {
            farmland.results.forEach((point: any) => {
              allTimePoints.push({
                ...point,
                objectid: farmland.objectid // Keep track of which farmland
              })
            })
          }
        })
        setResults(allTimePoints)

        // Also fetch and display farmland boundaries with NDVI data for popups
        if (currentBounds) {
          // Calculate min/max NDVI values for legend
          const allNDVIValues: number[] = []
          resultsData.forEach((farmland: any) => {
            if (farmland.results && Array.isArray(farmland.results)) {
              farmland.results.forEach((point: any) => {
                if (point.mean !== undefined && point.mean !== null && !isNaN(point.mean)) {
                  allNDVIValues.push(point.mean)
                }
              })
            }
          })

          const minNDVI = allNDVIValues.length > 0 ? Math.min(...allNDVIValues) : -1
          const maxNDVI = allNDVIValues.length > 0 ? Math.max(...allNDVIValues) : 1

          apiClient.getFarmlandGeoJSON(currentBounds)
            .then(geoJsonData => {
              // Combine GeoJSON with NDVI results and statistics for enhanced popups and legend
              setFarmlandGeoJSON({
                geoJSON: geoJsonData,
                ndviData: resultsData, // Pass the NDVI time series data
                colorData: {
                  type: 'ndvi',
                  min: minNDVI,
                  max: maxNDVI,
                  farmlands: resultsData
                }
              })
            })
            .catch(error => {
              console.error('Failed to load farmland boundaries:', error)
              // Still show results even if GeoJSON fails
            })
        }

        setAnalyzingType(null)
        const farmlandCount = resultsData.length
        const totalPoints = allTimePoints.length
        toast.success(`Found crop health data for ${farmlandCount} farmlands with ${totalPoints} total measurements!`)
      } else {
        // Polygon analysis: flat array of time series data
        setResults(resultsData)
        setFarmlandResults(null)
        setAnalyzingType(null)
        if (Array.isArray(resultsData) && resultsData.length > 0) {
          const ndviValues = resultsData.map((p: any) => p.mean).filter((v: any) => v != null)
          setNdviAnalysisResult({
            fromDate: selectedDateRange.from,
            toDate: selectedDateRange.to,
            averageNdvi: ndviValues.length ? Math.round((ndviValues.reduce((a: number, b: number) => a + b, 0) / ndviValues.length) * 1000) / 1000 : null,
            maxNdvi: ndviValues.length ? Math.round(Math.max(...ndviValues) * 1000) / 1000 : null,
            minNdvi: ndviValues.length ? Math.round(Math.min(...ndviValues) * 1000) / 1000 : null,
            dataPoints: resultsData.length,
          })
        }
        const dataCount = Array.isArray(resultsData) ? resultsData.length : 0
        if (dataCount > 0) {
          toast.success(`Found crop health data for ${dataCount} time points!`)

          // After getting time series data, load the first image for the first available date
          if (drawnPolygons && drawnPolygons.length > 0 && dataCount > 0) {
            setIsInitialLoading(true)
            toast('Loading satellite images...', { icon: '🛰️' })

            const allDates = resultsData.map((point: any) => point.date).sort()
            const firstDate = allDates[0]
            const remainingDates = allDates.slice(1)

            await loadImageForDate(firstDate, true) // Load first image and show on map

            // Start loading other images in parallel asynchronously
            // Pass all dates (including first) so metadata is set immediately
            await loadRemainingImagesAsync(allDates, remainingDates)
          }
        } else {
          toast.success('Analysis completed - but no data found for this period')
        }
      }
    },
    onError: (error: any) => {
      setAnalyzingType(null)
      toast.error(error.message || 'Could not get crop health data. Please try again.')
    }
  })


  // Helper functions for time slider

  const handleSliderDateChange = useCallback(async (date: string) => {
    setCurrentSliderDate(date)

    setIsChangingImage(true)

    // Check if image is already cached
    const cachedImageUrl = imageCache.get(date)
    if (cachedImageUrl) {
      setCurrentImageUrl(cachedImageUrl)

      // Show the image as an overlay on the map if we have drawn polygons
      if (drawnPolygons && drawnPolygons.length > 0) {
        // Calculate bounds for multiple polygons
        const bounds = calculateMultiPolygonBounds(drawnPolygons)
        if (bounds) {
          // Add timestamp to URL to force update
          const timestampedUrl = `${cachedImageUrl}#${Date.now()}`
          setNDVIImageOverlay(timestampedUrl, bounds)
        }
      }
      setIsChangingImage(false)
    } else {
      // Image not cached, load it on demand
      console.log('Loading image on demand for date:', date)

      // Only show individual notifications if not during initial loading
      if (!isInitialLoading) {
        toast('Loading satellite image...', { icon: '📊' })
      }

      const imageUrl = await loadImageForDate(date, true) // Load and show on map

      if (!isInitialLoading) {
        if (imageUrl) {
          toast.success('Satellite image loaded!')
        } else {
          toast.error('Failed to load satellite image')
        }
      }
      setIsChangingImage(false)
    }
  }, [imageCache, drawnPolygons, setNDVIImageOverlay, loadImageForDate, isInitialLoading, calculateMultiPolygonBounds])

  const canAnalyzeFarmlands = useMemo(() => {
    return selectedDataset === 'farmland' && currentZoom >= 12 && currentBounds
  }, [selectedDataset, currentZoom, currentBounds])

  // Cleanup blob URLs on unmount
  useEffect(() => {
    return () => {
      // Clean up all cached blob URLs when component unmounts
      imageCache.forEach((url) => {
        if (url.startsWith('blob:')) {
          URL.revokeObjectURL(url)
        }
      })
    }
  }, [imageCache])

  const handleAnalyzePolygon = () => {
    if (!drawnPolygons || drawnPolygons.length === 0) {
      toast.error('Please draw your farm area on the map first! Use the drawing tools on the map.')
      return
    }

    // Clear any farmland results when analyzing polygon
    setResults(null)
    setComparisonResults(null)
    setFarmlandResults(null)
    setIsComparison(false)
    setAnalyzingType('polygon')

    // Clear existing image data
    setImageMetadata(null)
    setCurrentSliderDate('')
    setCurrentImageUrl('')
    setImageCache(new Map())
    setShowTimeSlider(false)
    setIsLoadingImage(false)

    // Clear soil overlay and farmland data when switching to NDVI analysis
    setSoilImageOverlay(null, null)
    setFarmlandGeoJSON(null)

    // Create geometry - MultiPolygon if multiple polygons, single polygon if one
    let geometry: GeoJSONGeometry;
    if (drawnPolygons.length === 1) {
      geometry = drawnPolygons[0];
    } else {
      // Combine multiple polygons into a MultiPolygon
      geometry = {
        type: 'MultiPolygon',
        coordinates: drawnPolygons.map(polygon => polygon.coordinates)
      } as GeoJSONGeometry;
    }

    const params = {
      from: selectedDateRange.from,
      to: selectedDateRange.to,
      geometry,
      source: dataSource
    }
    toast('Checking your crop health over time...', { icon: '⏳' })
    ndviAnalysisMutation.mutate(params)
  }

  const handleAnalyzeFarmlands = () => {
    if (!canAnalyzeFarmlands) {
      toast.error('Zoom in further to enable farmland analysis')
      return
    }

    // Clear any polygon results when analyzing farmlands
    setResults(null)
    setComparisonResults(null)
    setFarmlandResults(null)
    setIsComparison(false)
    setAnalyzingType('farmland')

    // Clear existing image data from polygon analysis
    setImageMetadata(null)
    setCurrentSliderDate('')
    setCurrentImageUrl('')
    setImageCache(new Map())
    setShowTimeSlider(false)
    setIsLoadingImage(false)

    // Clear soil overlay, farmland data, and NDVI overlay when switching to farmland analysis
    setSoilImageOverlay(null, null)
    setFarmlandGeoJSON(null)
    setNDVIImageOverlay(null, null)

    const params = {
      from: selectedDateRange.from,
      to: selectedDateRange.to,
      bbox: currentBounds,
      isFarmlandAnalysis: true,
      source: dataSource
    }
    toast('Analyzing crop health for all farmlands in view...', { icon: '⏳' })
    ndviAnalysisMutation.mutate(params)
  }


  const handleCompareYear = () => {
    if (!comparisonYear || !results) return

    // For date ranges, try to match the same time period in the comparison year
    const fromDate = parseDate(selectedDateRange.from)
    const toDate = parseDate(selectedDateRange.to)

    const comparisonFrom = `${comparisonYear}-${String(fromDate.getMonth() + 1).padStart(2, '0')}-${String(fromDate.getDate()).padStart(2, '0')}`
    const comparisonTo = `${comparisonYear}-${String(toDate.getMonth() + 1).padStart(2, '0')}-${String(toDate.getDate()).padStart(2, '0')}`

    const currentYear = parseDate(selectedDateRange.from).getFullYear().toString()


    setAnalyzingType('polygon')
    toast(`Comparing ${currentYear} with ${comparisonYear}...`, { icon: '⏳' })

    // Create a separate mutation for comparison
    if (drawnPolygons && drawnPolygons.length > 0) {
      // Create geometry - MultiPolygon if multiple polygons, single polygon if one
      let geometry: GeoJSONGeometry;
      if (drawnPolygons.length === 1) {
        geometry = drawnPolygons[0];
      } else {
        // Combine multiple polygons into a MultiPolygon
        geometry = {
          type: 'MultiPolygon',
          coordinates: drawnPolygons.map(polygon => polygon.coordinates)
        } as GeoJSONGeometry;
      }

      apiClient.getNDVIForPolygon({ from: comparisonFrom, to: comparisonTo, geometry, source: dataSource })
        .then(data => {
          const results = data.results || data
          if (results && results.length > 0) {
            setComparisonResults(results)
            setIsComparison(true)
            setAnalyzingType(null)
            toast.success(`Comparison with ${comparisonYear} loaded! Found ${results.length} data points.`)
          } else {
            setAnalyzingType(null)
            toast.error(`No records found for ${comparisonYear}. Try a different year or check if data is available for that period.`)
          }
        })
        .catch(error => {
          setAnalyzingType(null)
          toast.error(`Failed to load ${comparisonYear} data: ${error.message}`)
        })
    } else if (canAnalyzeFarmlands) {
      apiClient.getNDVIForRegion('farmland', comparisonFrom, comparisonTo, currentBounds || undefined, dataSource)
        .then(data => {
          const resultsData = data.results || data
          if (resultsData && resultsData.length > 0) {
            // Handle farmland comparison structure - flatten the results
            const allTimePoints: any[] = []
            resultsData.forEach((farmland: any) => {
              if (farmland.results && Array.isArray(farmland.results)) {
                farmland.results.forEach((point: any) => {
                  allTimePoints.push({
                    ...point,
                    objectid: farmland.objectid
                  })
                })
              }
            })
            setComparisonResults(allTimePoints)
            setIsComparison(true)
            setAnalyzingType(null)
            toast.success(`Comparison with ${comparisonYear} loaded! Found ${resultsData.length} farmlands with ${allTimePoints.length} measurements.`)
          } else {
            setAnalyzingType(null)
            toast.error(`No farmland records found for ${comparisonYear}. Try a different year or check if data is available for that period.`)
          }
        })
        .catch(error => {
          setAnalyzingType(null)
          toast.error(`Failed to load ${comparisonYear} data: ${error.message}`)
        })
    }
  }

  const handleExportCSV = () => {
    if (!results || results.length === 0) {
      toast.error('No data to export')
      return
    }

    try {
      toast('Generating CSV file...', { icon: '⏳' })

      // Simple CSV header - only current period data
      let csvContent = 'Date,NDVI Value,Health Status\n'

      // Export only current period data
      results.forEach((point: any) => {
        if (point.mean !== undefined && point.mean !== null && !isNaN(point.mean)) {
          const healthStatus = point.mean > 0.5 ? 'Excellent' : point.mean > 0.2 ? 'Good' : 'Poor'
          csvContent += `${new Date(point.date).toLocaleDateString()},${point.mean.toFixed(3)},${healthStatus}\n`
        }
      })

      // Create and download the file
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
      const link = document.createElement('a')
      const url = URL.createObjectURL(blob)
      link.setAttribute('href', url)
      link.setAttribute('download', `ndvi-data-${selectedDateRange.from}-to-${selectedDateRange.to}-${new Date().toISOString().split('T')[0]}.csv`)
      link.style.visibility = 'hidden'
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      URL.revokeObjectURL(url)

      toast.success('CSV file exported successfully!')
    } catch (error) {
      toast.error('Failed to export CSV file')
    }
  }

  const handleExportPDF = async () => {
    if (!results || results.length === 0) {
      toast.error('No data to export')
      return
    }

    try {
      toast('Generating PDF report...', { icon: '⏳' })

      const jsPDF = (await import('jspdf')).default
      const html2canvas = (await import('html2canvas')).default

      const pdf = new jsPDF('p', 'mm', 'a4')
      const pageWidth = pdf.internal.pageSize.getWidth()
      const pageHeight = pdf.internal.pageSize.getHeight()
      let yPosition = 20

      // Title
      pdf.setFontSize(18)
      pdf.setFont('helvetica', 'bold')
      pdf.text('FutureFarmNow NDVI Analysis Report', pageWidth / 2, yPosition, { align: 'center' })
      yPosition += 15

      // Analysis details
      pdf.setFontSize(12)
      pdf.setFont('helvetica', 'normal')
      pdf.text(`Analysis Period: ${selectedDateRange.from} to ${selectedDateRange.to}`, 20, yPosition)
      yPosition += 8
      pdf.text(`Analysis Date: ${new Date().toLocaleDateString()}`, 20, yPosition)
      yPosition += 8
      const totalDays = Math.ceil((new Date(selectedDateRange.to).getTime() - new Date(selectedDateRange.from).getTime()) / (1000 * 60 * 60 * 24))
      pdf.text(`Duration: ${totalDays} days`, 20, yPosition)
      yPosition += 8
      pdf.text(`Data Points: ${results.length} measurements`, 20, yPosition)
      yPosition += 15

      // Chart
      const chartElement = document.getElementById('ndvi-chart-container')
      if (chartElement) {
        const chartCanvas = await html2canvas(chartElement, {
          backgroundColor: '#ffffff',
          scale: 2,
          logging: false
        })
        const chartImgData = chartCanvas.toDataURL('image/png')
        const chartWidth = pageWidth - 40
        const chartHeight = (chartCanvas.height * chartWidth) / chartCanvas.width

        if (yPosition + chartHeight > pageHeight - 20) {
          pdf.addPage()
          yPosition = 20
        }

        pdf.setFontSize(12)
        pdf.setFont('helvetica', 'bold')
        pdf.text('NDVI Time Series Chart:', 20, yPosition)
        yPosition += 10

        pdf.addImage(chartImgData, 'PNG', 20, yPosition, chartWidth, chartHeight)
        yPosition += chartHeight + 15
      }

      // Statistics
      if (yPosition > pageHeight - 60) {
        pdf.addPage()
        yPosition = 20
      }

      pdf.setFontSize(14)
      pdf.setFont('helvetica', 'bold')
      pdf.text('Statistical Summary:', 20, yPosition)
      yPosition += 10

      const validResults = results.filter((point: any) => point.mean !== undefined && point.mean !== null)
      const avgHealth = validResults.length > 0 ? (validResults.reduce((sum: number, point: any) => sum + point.mean, 0) / validResults.length).toFixed(3) : 'N/A'
      const minHealth = validResults.length > 0 ? Math.min(...validResults.map((point: any) => point.mean)).toFixed(3) : 'N/A'
      const maxHealth = validResults.length > 0 ? Math.max(...validResults.map((point: any) => point.mean)).toFixed(3) : 'N/A'

      pdf.setFontSize(11)
      pdf.setFont('helvetica', 'normal')

      // Analysis period statistics
      pdf.setFontSize(12)
      pdf.setFont('helvetica', 'bold')
      pdf.text(`Analysis Period Statistics:`, 25, yPosition)
      yPosition += 8

      pdf.setFontSize(11)
      pdf.setFont('helvetica', 'normal')
      const stats = [
        `Average NDVI: ${avgHealth}`,
        `Minimum NDVI: ${minHealth}`,
        `Maximum NDVI: ${maxHealth}`,
        `Health Status: ${parseFloat(avgHealth) > 0.5 ? 'Excellent' : parseFloat(avgHealth) > 0.2 ? 'Good' : 'Poor'}`
      ]

      stats.forEach(stat => {
        pdf.text(`• ${stat}`, 30, yPosition)
        yPosition += 7
      })

      // Data table
      yPosition += 10
      if (yPosition > pageHeight - 100) {
        pdf.addPage()
        yPosition = 20
      }

      pdf.setFontSize(12)
      pdf.setFont('helvetica', 'bold')
      pdf.text('Detailed Measurements:', 20, yPosition)
      yPosition += 10

      pdf.setFontSize(9)
      pdf.setFont('helvetica', 'normal')

      // Always use simple table format (no comparison data)
      // Table header for primary data only
      pdf.text('Date', 25, yPosition)
      pdf.text('NDVI Value', 80, yPosition)
      pdf.text('Health Status', 130, yPosition)
      yPosition += 7

      // Table data - only primary results
      results.forEach((point: any) => {
        if (point.mean !== undefined && point.mean !== null && !isNaN(point.mean)) {
          if (yPosition > pageHeight - 20) {
            pdf.addPage()
            yPosition = 20
          }

          const healthStatus = point.mean > 0.5 ? 'Excellent' : point.mean > 0.2 ? 'Good' : 'Poor'
          pdf.text(new Date(point.date).toLocaleDateString(), 25, yPosition)
          pdf.text(point.mean.toFixed(3), 80, yPosition)
          pdf.text(healthStatus, 130, yPosition)
          yPosition += 6
        }
      })

      // Footer
      pdf.setFontSize(8)
      pdf.setFont('helvetica', 'normal')
      pdf.text('Generated by FutureFarmNow - University of California, Riverside', pageWidth / 2, pageHeight - 10, { align: 'center' })

      // Save
      const filename = `ndvi-analysis-${selectedDateRange.from}-to-${selectedDateRange.to}-${new Date().toISOString().split('T')[0]}.pdf`
      pdf.save(filename)

      toast.success('PDF report exported successfully!')
    } catch (error) {
      toast.error('Failed to export PDF report')
    }
  }

  const handleDateChange = (field: 'from' | 'to', value: string) => {
    setSelectedDateRange({
      ...selectedDateRange,
      [field]: value
    })
  }

  return (
    <div className="p-4 space-y-6">
      {/* Parameters Section */}
      <div className="space-y-4">
        <h3 className="font-medium text-foreground flex items-center">
          <Calendar className="h-4 w-4 mr-2" />
          Track Your Crop Health Over Time
        </h3>

        <div className="text-sm text-muted-foreground mb-4">
          NDVI (Normalized Difference Vegetation Index) measures vegetation health using satellite data. Values range -1 to 1: higher = greener/healthier crops.
        </div>

        <div className="space-y-4">
          {/* Data Source Selection */}
          <div className="flex items-baseline gap-4 w-full" data-tutorial="data-source-selector">
            <label className="text-sm font-medium text-foreground w-20 shrink-0">
              Source
            </label>
            <div className="flex-1">
              <Select
                value={dataSource}
                onChange={(e) => setDataSource(e.target.value)}
                className="w-full"
                style={{ height: '40px', padding: '8px 12px', fontSize: '14px', lineHeight: '20px' }}
              >
                <SelectOption value="ndvi">Sentinel-2</SelectOption>
                <SelectOption value="landsat">Landsat 8/9</SelectOption>
              </Select>
            </div>
          </div>

          <div className="flex items-baseline gap-4" data-tutorial="date-selector">
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

      {/* Analysis Buttons */}
      <div className="space-y-3" data-tutorial="ndvi-analysis-buttons">
        {/* Polygon Analysis Button */}
        <div>
          <Button
            onClick={handleAnalyzePolygon}
            disabled={!drawnPolygons || drawnPolygons.length === 0 || analyzingType === 'farmland'}
            className={`w-full transition-opacity ${analyzingType === 'farmland' ? 'opacity-50' : ''}`}
          >
            {analyzingType === 'polygon' ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                Analyzing...
              </>
            ) : (
              'Check Crop Health'
            )}
          </Button>
        </div>

        {/* Farmland Analysis Button (conditional) */}
        {canAnalyzeFarmlands && (
          <div>
            <Button
              onClick={handleAnalyzeFarmlands}
              disabled={analyzingType === 'polygon'}
              variant="outline"
              className={`w-full transition-opacity ${analyzingType === 'polygon' ? 'opacity-50' : ''}`}
              data-tutorial="analyze-ndvi-farmland-button"
            >
              {analyzingType === 'farmland' ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary mr-2" />
                  Analyzing...
                </>
              ) : (
                'Analyze All Farmlands in View'
              )}
            </Button>
          </div>
        )}


        {/* Warning Messages */}
        {(!drawnPolygons || drawnPolygons.length === 0) && !canAnalyzeFarmlands && (
          <div className="mt-2 flex items-start space-x-2 text-amber-600 dark:text-amber-400 text-sm">
            <AlertCircle className="h-4 w-4 mt-0.5 flex-shrink-0" />
            <p>Draw a polygon on the map to enable analysis or zoom in further to check all farmlands</p>
          </div>
        )}

        {canAnalyzeFarmlands && (
          <div className="mt-2 flex items-start space-x-2 text-green-600 dark:text-green-400 text-sm">
            <AlertCircle className="h-4 w-4 mt-0.5 flex-shrink-0" />
            <p>You can draw a specific area or analyze all farmland visible in the current view</p>
          </div>
        )}
      </div>

      {/* Results Section */}
      {results && results.length > 0 && (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold text-foreground flex items-center">
              <Calendar className="h-5 w-5 mr-2 text-green-600" />
              Your Crop Health Over Time
            </h3>
            <div className="flex gap-2">
              <Button
                onClick={handleExportPDF}
                variant="outline"
                size="sm"
                disabled={!results || results.length === 0}
              >
                <Download className="h-4 w-4 mr-1" />
                Export PDF
              </Button>
              <Button
                onClick={() => {
                  setResults(null)
                  setComparisonResults(null)
                  setFarmlandResults(null)
                  setIsComparison(false)
                  // Clear image cache and overlays
                  setImageCache(new Map())
                  setImageMetadata(null)
                  setCurrentSliderDate('')
                  setCurrentImageUrl('')
                  setShowTimeSlider(false)
                  setNDVIImageOverlay(null, null)
                }}
                variant="outline"
                size="sm"
              >
                Clear Results
              </Button>
            </div>
          </div>

          {/* NDVI Color Legend - Show for both polygon and farmland analysis */}
          <NDVILegend
            min={farmlandResults && farmlandResults.length > 0
              ? Math.min(...farmlandResults.flatMap((f: any) => f.results?.map((r: any) => r.mean) || []).filter((v: any) => v != null && !isNaN(v)))
              : Math.min(...results.map((r: any) => r.mean).filter((v: any) => v != null && !isNaN(v)))
            }
            max={farmlandResults && farmlandResults.length > 0
              ? Math.max(...farmlandResults.flatMap((f: any) => f.results?.map((r: any) => r.mean) || []).filter((v: any) => v != null && !isNaN(v)))
              : Math.max(...results.map((r: any) => r.mean).filter((v: any) => v != null && !isNaN(v)))
            }
          />

          <div className="bg-card border rounded-lg">
            <div className="p-3 border-b">
              <h4 className="text-sm font-medium text-foreground">NDVI Over Time</h4>
              <div className="text-xs text-muted-foreground mt-1">
                {comparisonResults ? 'Comparison view - current period (green) vs selected year (blue)' : 'Track vegetation health changes over your selected time period'}
                {showTimeSlider && !isComparison && (
                  <span className="block mt-1">Click any point to view the satellite image for that date</span>
                )}
              </div>
            </div>
            <div id="ndvi-chart-container" className="p-3">
              <TimeSeriesChart
                data={results}
                comparisonData={comparisonResults}
                showComparison={isComparison}
                primaryLabel={isComparison ? new Date(selectedDateRange.from).getFullYear().toString() : undefined}
                comparisonLabel={isComparison ? comparisonYear : undefined}
                onDateClick={farmlandResults && farmlandResults.length > 0 ? undefined : handleSliderDateChange}
                selectedDate={farmlandResults && farmlandResults.length > 0 ? undefined : currentSliderDate}
                showNavigation={showTimeSlider && !isComparison && (!farmlandResults || farmlandResults.length === 0)}
                availableDates={farmlandResults && farmlandResults.length > 0 ? undefined : imageMetadata?.available_dates}
                enableAutoPlay={showTimeSlider && !isComparison && (!farmlandResults || farmlandResults.length === 0)}
                onNavigate={farmlandResults && farmlandResults.length > 0 ? undefined : (direction) => {
                  const currentIdx = imageMetadata?.available_dates?.indexOf(currentSliderDate) ?? -1
                  if (currentIdx !== -1 && imageMetadata?.available_dates) {
                    const totalDates = imageMetadata.available_dates.length
                    if (direction === 'prev') {
                      // If at first, go to last; otherwise go to previous
                      const newIdx = currentIdx === 0 ? totalDates - 1 : currentIdx - 1
                      handleSliderDateChange(imageMetadata.available_dates[newIdx])
                    } else if (direction === 'next') {
                      // If at last, go to first; otherwise go to next
                      const newIdx = currentIdx === totalDates - 1 ? 0 : currentIdx + 1
                      handleSliderDateChange(imageMetadata.available_dates[newIdx])
                    }
                  }
                }}
              />
            </div>
          </div>


          <div className="bg-blue-50 dark:bg-blue-950/30 p-4 rounded-lg border-l-4 border-blue-400 space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <h4 className="font-medium text-blue-800 dark:text-blue-200 mb-2">Compare with Another Year</h4>
                <p className="text-sm text-blue-700 dark:text-blue-300">Compare your current date range with the same period in a different year</p>
              </div>
              <div className="flex items-center gap-3">
                <Input
                  type="number"
                  placeholder="2023"
                  value={comparisonYear}
                  onChange={(e) => setComparisonYear(e.target.value)}
                  className="w-24 text-sm"
                  min="2015"
                  max="2035"
                />
                <Button
                  onClick={handleCompareYear}
                  size="sm"
                  disabled={!comparisonYear || comparisonYear === parseDate(selectedDateRange.from).getFullYear().toString()}
                >
                  <GitCompare className="h-4 w-4 mr-2" />
                  Compare
                </Button>
              </div>
            </div>
            {isComparison && comparisonResults && (
              <div className="flex items-center justify-between pt-2 border-t border-blue-200 dark:border-blue-800">
                <span className="text-sm text-blue-700 dark:text-blue-300">
                  Comparing {new Date(selectedDateRange.from).getFullYear()} with {comparisonYear} • {comparisonResults.length} data points
                </span>
                <Button
                  onClick={() => {
                    setIsComparison(false)
                    setComparisonResults(null)
                    setComparisonYear('')
                    // Clear image cache and overlays when clearing comparison
                    setImageCache(new Map())
                    setImageMetadata(null)
                    setCurrentSliderDate('')
                    setCurrentImageUrl('')
                    setShowTimeSlider(false)
                    setNDVIImageOverlay(null, null)
                  }}
                  variant="ghost"
                  size="sm"
                  className="text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-200"
                >
                  Clear Comparison
                </Button>
              </div>
            )}
          </div>

          {/* Summary Statistics */}
          <div className="bg-card p-4 rounded-lg border space-y-4">
            <h4 className="font-medium text-foreground mb-3">Summary of Your Crop Health:</h4>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-muted/20 p-3 rounded-lg">
                <dt className="text-sm font-medium text-muted-foreground">Measurements Taken:</dt>
                <dd className="text-lg font-semibold text-foreground">{results.length}</dd>
                <div className="text-xs text-muted-foreground mt-1">Number of satellite images analyzed</div>
              </div>

              <div className="bg-muted/20 p-3 rounded-lg">
                <dt className="text-sm font-medium text-muted-foreground">Average Health Score:</dt>
                <dd className="text-lg font-semibold text-foreground">
                  {(() => {
                    const validPoints = results.filter((point: any) => point.mean !== undefined && point.mean !== null)
                    return validPoints.length > 0
                      ? (validPoints.reduce((sum: number, point: any) => sum + point.mean, 0) / validPoints.length).toFixed(2)
                      : 'N/A'
                  })()}
                </dd>
                <div className="text-xs text-muted-foreground mt-1">Overall crop health (-1.0 to 1.0 scale)</div>
              </div>

              <div className="bg-muted/20 p-3 rounded-lg">
                <dt className="text-sm font-medium text-muted-foreground">Lowest Health Score:</dt>
                <dd className="text-lg font-semibold text-foreground">
                  {(() => {
                    const validPoints = results.filter((point: any) => point.mean !== undefined && point.mean !== null)
                    return validPoints.length > 0
                      ? Math.min(...validPoints.map((point: any) => point.mean)).toFixed(2)
                      : 'N/A'
                  })()}
                </dd>
                <div className="text-xs text-muted-foreground mt-1">Worst day in the period</div>
              </div>

              <div className="bg-muted/20 p-3 rounded-lg">
                <dt className="text-sm font-medium text-muted-foreground">Highest Health Score:</dt>
                <dd className="text-lg font-semibold text-foreground">
                  {(() => {
                    const validPoints = results.filter((point: any) => point.mean !== undefined && point.mean !== null)
                    return validPoints.length > 0
                      ? Math.max(...validPoints.map((point: any) => point.mean)).toFixed(2)
                      : 'N/A'
                  })()}
                </dd>
                <div className="text-xs text-muted-foreground mt-1">Best day in the period</div>
              </div>
            </div>
          </div>

          {/* Data Table */}
          <div className="bg-card border rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <h4 className="font-medium text-foreground">Daily Health Scores</h4>
              <Button
                onClick={handleExportCSV}
                variant="outline"
                size="sm"
                className="text-xs"
                disabled={!results || results.length === 0}
              >
                <Download className="h-3 w-3 mr-1" />
                CSV
              </Button>
            </div>
            <div className="max-h-48 overflow-y-auto border rounded">
              <table className="w-full text-sm">
                <thead className="bg-background sticky top-0 z-10 border-b">
                  <tr>
                    <th className="px-3 py-2 text-left font-medium text-foreground bg-background">Date</th>
                    <th className="px-3 py-2 text-right font-medium text-foreground bg-background">Health Score</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {results.filter((point: any) => point.mean !== undefined && point.mean !== null && !isNaN(point.mean)).map((point: any, index: number) => {
                    const healthScore = point.mean
                    const healthStatus = healthScore > 0.5 ? 'Excellent' : healthScore > 0.2 ? 'Good' : 'Poor'
                    const statusColor = healthScore > 0.5 ? 'text-green-600' : healthScore > 0.2 ? 'text-yellow-600' : 'text-red-600'

                    return (
                      <tr key={index} className="hover:bg-muted/20">
                        <td className="px-3 py-2 text-foreground">
                          {new Date(point.date).toLocaleDateString()}
                        </td>
                        <td className="px-3 py-2 text-right">
                          <div className="flex flex-col items-end">
                            <span className="font-mono font-medium text-foreground">
                              {point.mean.toFixed(2)}
                            </span>
                            <span className={`text-xs ${statusColor}`}>
                              {healthStatus}
                            </span>
                          </div>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>

        </div>
      )}

      {results && results.length === 0 && (
        <div className="text-center py-8 text-muted-foreground">
          <Calendar className="h-12 w-12 mx-auto mb-4 text-muted-foreground/50" />
          <div className="space-y-2">
            <p className="font-medium">No crop health data found</p>
            <p className="text-sm">
              There might be no satellite images available for your selected area and time period.
            </p>
            <p className="text-xs text-muted-foreground/80">
              Try selecting a different date range or check if your farm area is drawn correctly.
            </p>
          </div>
        </div>
      )}
    </div>
  )
}