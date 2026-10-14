'use client'

import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Select, SelectOption } from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import { useMapStore } from '@/lib/stores/mapStore'
import { useMutation } from '@tanstack/react-query'
import { apiClient } from '@/lib/api/client'
import { SoilLayerEnum, GeoJSONGeometry } from '@/lib/types/api'
import { BoxWhiskerPlot } from '@/components/visualizations/BoxWhiskerPlot'
import { Legend } from '@/components/visualizations/Legend'
import { SoilLegend } from '@/components/visualizations/SoilLegend'
import { AlertCircle, BarChart3, MapPin, X, Download, Layers } from 'lucide-react'
import { toast } from '@/lib/utils/toast'

const SOIL_LAYERS: { value: SoilLayerEnum; label: string; description: string }[] = [
  { value: 'alpha', label: 'Water Retention (alpha)', description: 'How well soil holds water - higher values indicate better water retention' },
  { value: 'bd', label: 'Bulk Density (bd)', description: 'Soil compaction - lower values mean less compacted, better for roots' },
  { value: 'clay', label: 'Clay Content (clay)', description: 'Percentage of fine particles that hold nutrients and water well' },
  { value: 'hb', label: 'Air Entry Pressure (hb)', description: 'Pressure needed for air to enter soil pores' },
  { value: 'ksat', label: 'Water Flow Rate (ksat)', description: 'How fast water moves through soil - higher is better drainage' },
  { value: 'lambda', label: 'Pore Distribution (lambda)', description: 'How uniform the soil pore sizes are' },
  { value: 'n', label: 'Porosity (n)', description: 'Amount of air space in soil - higher means more pore space' },
  { value: 'om', label: 'Organic Matter (om)', description: 'Decomposed plant material - higher is better for soil health' },
  { value: 'ph', label: 'Soil pH (ph)', description: 'Acidity level - 6.5-7.0 is ideal for most crops' },
  { value: 'sand', label: 'Sand Content (sand)', description: 'Percentage of large particles that improve drainage' },
  { value: 'silt', label: 'Silt Content (silt)', description: 'Percentage of medium particles good for water retention' },
  { value: 'theta_r', label: 'Min Water Content (theta_r)', description: 'Minimum water the soil can hold when very dry' },
  { value: 'theta_s', label: 'Max Water Content (theta_s)', description: 'Maximum water the soil can hold when saturated' }
]

const SOIL_DEPTHS = [
  { value: '0-5', label: 'Surface (0-5 cm / 0-2 inches)', description: 'Top soil layer where seeds germinate' },
  { value: '5-15', label: 'Shallow (5-15 cm / 2-6 inches)', description: 'Root zone for small plants' },
  { value: '15-30', label: 'Medium (15-30 cm / 6-12 inches)', description: 'Main root zone for most crops' },
  { value: '30-60', label: 'Deep (30-60 cm / 1-2 feet)', description: 'Deep root zone for large plants' },
  { value: '60-100', label: 'Very Deep (60-100 cm / 2-3 feet)', description: 'Subsoil layer' },
  { value: '100-200', label: 'Deepest (100-200 cm / 3-6 feet)', description: 'Deep subsoil layer' },
  { value: 'custom', label: 'Custom Range', description: 'Select your own depth range' }
]

export function SoilAnalysis() {
  const {
    selectedSoilLayer,
    selectedSoilDepth,
    drawnPolygon,
    drawnPolygons,
    currentBounds,
    currentZoom,
    setSelectedSoilLayer,
    setSelectedSoilDepth,
    setSoilImageOverlay,
    setFarmlandGeoJSON,
    setNDVIImageOverlay,
    clearNDVICache,
    clearTrigger,
    setSoilAnalysisResult
  } = useMapStore()

  const [results, setResults] = useState<any>(null)
  const [farmlandResults, setFarmlandResults] = useState<any>(null)
  const [legendData, setLegendData] = useState<any>(null)
  const [soilImageUrl, setSoilImageUrl] = useState<string | null>(null)
  const [analysisType, setAnalysisType] = useState<'polygon' | 'farmland' | null>(null)
  const [customDepthRange, setCustomDepthRange] = useState<[number, number]>([0, 30])
  const [isCustomDepth, setIsCustomDepth] = useState(false)

  // Clear state when clearTrigger changes
  useEffect(() => {
    if (clearTrigger > 0) {
      setResults(null)
      setFarmlandResults(null)
      setLegendData(null)
      setAnalysisType(null)
      setSoilImageUrl(prevUrl => {
        if (prevUrl) {
          URL.revokeObjectURL(prevUrl)
        }
        return null
      })
    }
  }, [clearTrigger])

  // Initialize custom depth state from selectedSoilDepth URL parameter
  useEffect(() => {
    // Check if selectedSoilDepth is a custom format (not in predefined list)
    const predefinedDepths = ['0-5', '5-15', '15-30', '30-60', '60-100', '100-200']
    
    if (selectedSoilDepth && !predefinedDepths.includes(selectedSoilDepth)) {
      // Parse custom depth format like "10-45"
      const match = selectedSoilDepth.match(/^(\d+)-(\d+)$/)
      if (match) {
        const start = parseInt(match[1])
        const end = parseInt(match[2])
        setCustomDepthRange([start, end])
        setIsCustomDepth(true)
      }
    } else {
      // Reset to default if it's a predefined depth
      setIsCustomDepth(false)
    }
  }, [selectedSoilDepth])

  // Handle custom depth range changes with validation
  const handleCustomDepthChange = (values: number[]) => {
    const [start, end] = values
    
    // Ensure start doesn't exceed 100
    const constrainedStart = Math.min(start, 100)
    
    // Ensure end is always greater than start
    const constrainedEnd = Math.max(end, constrainedStart + 5) // Minimum 5cm gap
    
    setCustomDepthRange([constrainedStart, constrainedEnd])
    
    // Update global store with custom depth string for URL sync
    const customDepthString = `${constrainedStart}-${constrainedEnd}`
    setSelectedSoilDepth(customDepthString)
  }

  // Calculate bounds from drawn polygon
  const calculatePolygonBounds = (geometry: any): [[number, number], [number, number]] | null => {
    if (!geometry || !geometry.coordinates || !geometry.coordinates[0]) return null
    
    const coords = geometry.coordinates[0] // First ring of polygon
    let minLat = Infinity, maxLat = -Infinity
    let minLng = Infinity, maxLng = -Infinity
    
    coords.forEach(([lng, lat]: [number, number]) => {
      minLat = Math.min(minLat, lat)
      maxLat = Math.max(maxLat, lat)
      minLng = Math.min(minLng, lng)
      maxLng = Math.max(maxLng, lng)
    })
    
    return [[minLat, minLng], [maxLat, maxLng]]
  }

  // Calculate MBR (Minimum Bounding Rectangle) for multiple polygons
  const calculateMultiPolygonBounds = (polygons: GeoJSONGeometry[]): [[number, number], [number, number]] | null => {
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
  }

  const soilAnalysisMutation = useMutation({
    mutationFn: (params: any) => apiClient.getSoilStatsForPolygon(params),
    onSuccess: (data) => {
      if (data.results && Object.keys(data.results).length > 0) {
        setResults(data.results)
        setAnalysisType('polygon')
        setSoilAnalysisResult({
          layer: selectedSoilLayer,
          depth: selectedSoilDepth,
          mean: data.results.mean ?? null,
          min: data.results.min ?? null,
          max: data.results.max ?? null,
          median: data.results.median ?? null,
          stddev: data.results.stddev ?? null,
          count: data.results.count ?? null,
        })
        // Don't show success toast here - wait for image to complete
      } else {
        toast.error('No soil data available for the selected area')
      }
    },
    onError: (error: any) => {
      toast.error(error.message || 'Analysis failed')
    }
  })

  const farmlandAnalysisMutation = useMutation({
    mutationFn: (params: { soildepth: string; layer: string; bbox: any }) => 
      apiClient.getSoilStatsForFarmland(params.soildepth, params.layer, params.bbox),
    onSuccess: (data) => {
      
      // The farmland API returns results in different formats
      // Check if it's an array of farmland results or a single stats object
      if (Array.isArray(data) && data.length > 0) {
        // Array of farmland results
        let globalMin = Infinity
        let globalMax = -Infinity
        
        // Find global min/max using average/mean values
        data.forEach((farmland: any, index: number) => {
          const stats = farmland.results || farmland
          const value = stats.average ?? stats.mean
          if (value !== undefined && value !== null && !isNaN(value)) {
            if (value < globalMin) globalMin = value
            if (value > globalMax) globalMax = value
          }
        })
        
        
        if (globalMin !== Infinity && globalMax !== -Infinity) {
          // Set farmland results with calculated global values
          setFarmlandResults({
            min: globalMin,
            max: globalMax,
            farmlands: data
          })
        } else {
          toast.error('No farmland data available in the current view')
        }
      } else if (data.results && Array.isArray(data.results) && data.results.length > 0) {
        // Results property contains array
        let globalMin = Infinity
        let globalMax = -Infinity
        
        data.results.forEach((result: any, index: number) => {
          const value = result.average ?? result.mean
          if (value !== undefined && value !== null && !isNaN(value)) {
            if (value < globalMin) globalMin = value
            if (value > globalMax) globalMax = value
          }
        })
        
        
        if (globalMin !== Infinity && globalMax !== -Infinity) {
          setFarmlandResults({
            min: globalMin,
            max: globalMax,
            farmlands: data.results
          })
        } else {
          toast.error('No farmland data available in the current view')
        }
      } else if (data.results && Object.keys(data.results).length > 0) {
        // Single result case - use its min/max directly
        setFarmlandResults(data.results)
      } else if (data && Object.keys(data).length > 0) {
        // Direct stats object
        setFarmlandResults(data)
      } else {
        toast.error('No farmland data available in the current view')
      }
      
      setAnalysisType('farmland')
      // Don't show success toast here - wait for GeoJSON to complete
    },
    onError: (error: any) => {
      if (error.message?.includes('Request is too costly')) {
        toast.error('Area too large! Please zoom in further to reduce the analysis area.')
      } else {
        toast.error(error.message || 'Farmland analysis failed')
      }
    }
  })

  const farmlandGeoJSONMutation = useMutation({
    mutationFn: (bbox: any) => apiClient.getFarmlandGeoJSON(bbox),
    onSuccess: (data) => {
      
      // If we have farmland results, pass them along with the GeoJSON
      if (farmlandResults) {
        setFarmlandGeoJSON({
          geoJSON: data,
          colorData: {
            min: farmlandResults.min,
            max: farmlandResults.max,
            farmlands: farmlandResults.farmlands
          }
        })
      } else {
        setFarmlandGeoJSON({ geoJSON: data })
      }
      // Show single success message when farmland analysis is complete
      if (farmlandResults) {
        toast.success('Farmland analysis completed')
      }
    },
    onError: (error: any) => {
      toast.error('Failed to load farmland boundaries')
    }
  })

  const soilImageMutation = useMutation({
    mutationFn: (params: any) => apiClient.getSoilImage(params),
    onSuccess: (blob: Blob) => {
      // Create object URL from blob
      const imageUrl = URL.createObjectURL(blob)
      setSoilImageUrl(imageUrl)
      
      // Calculate MBR bounds for all polygons and set overlay on map
      if (drawnPolygons && drawnPolygons.length > 0) {
        const bounds = calculateMultiPolygonBounds(drawnPolygons)
        if (bounds) {
          setSoilImageOverlay(imageUrl, bounds)
        }
      }
      
      // Show single success message when both stats and image are complete
      if (results) {
        toast.success('Soil analysis completed')
      }
    },
    onError: (error: any) => {
      if (error.message?.includes('CORS') || error.code === 'ERR_NETWORK') {
        toast.error('Network error - please check your connection and try again')
      } else {
        toast.error('Failed to load soil visualization: ' + (error.message || 'Unknown error'))
      }
    }
  })

  const handleAnalyze = () => {
    if (!drawnPolygons || drawnPolygons.length === 0) {
      toast.error('Please draw your farm area on the map first! Use the drawing tools on the map.')
      return
    }

    // Custom depth validation is handled by the slider constraints

    // Clear previous results first (both polygon and farmland)
    setResults(null)
    setFarmlandResults(null)
    setLegendData(null)
    setAnalysisType(null)
    // Clean up previous image URL to prevent memory leaks
    if (soilImageUrl) {
      URL.revokeObjectURL(soilImageUrl)
      setSoilImageUrl(null)
    }

    // Create geometry - MultiPolygon if multiple polygons, single polygon if one
    let geometry;
    if (drawnPolygons.length === 1) {
      geometry = drawnPolygons[0];
    } else {
      // Combine multiple polygons into a MultiPolygon
      geometry = {
        type: 'MultiPolygon',
        coordinates: drawnPolygons.map(polygon => polygon.coordinates)
      };
    }

    const params = {
      soildepth: isCustomDepth ? `${customDepthRange[0]}-${customDepthRange[1]}` : selectedSoilDepth,
      layer: selectedSoilLayer,
      geometry: geometry
    }

    // Clear any farmland overlay and NDVI overlay when doing polygon analysis
    setFarmlandGeoJSON(null)
    clearNDVICache()
    
    soilAnalysisMutation.mutate(params)
    soilImageMutation.mutate(params)
  }

  const handleFarmlandAnalyze = () => {
    if (!currentBounds) {
      toast.error('Map bounds not available. Please move the map and try again.')
      return
    }

    // Custom depth validation is handled by the slider constraints

    // Clear previous results first (both polygon and farmland)
    setResults(null)
    setFarmlandResults(null)
    setLegendData(null)
    setAnalysisType(null)
    // Clean up previous image URL to prevent memory leaks
    if (soilImageUrl) {
      URL.revokeObjectURL(soilImageUrl)
      setSoilImageUrl(null)
    }

    const params = {
      soildepth: isCustomDepth ? `${customDepthRange[0]}-${customDepthRange[1]}` : selectedSoilDepth,
      layer: selectedSoilLayer,
      bbox: currentBounds
    }

    // Clear any polygon overlay and NDVI overlay when doing farmland analysis
    setSoilImageOverlay(null, null)
    clearNDVICache()

    farmlandAnalysisMutation.mutate(params)
    farmlandGeoJSONMutation.mutate(currentBounds)
  }

  const handleReset = () => {
    setResults(null)
    setFarmlandResults(null)
    setLegendData(null)
    setAnalysisType(null)
    if (soilImageUrl) {
      URL.revokeObjectURL(soilImageUrl)
      setSoilImageUrl(null)
    }
    // Clear map overlays
    setSoilImageOverlay(null, null)
    clearNDVICache()
    setFarmlandGeoJSON(null)
    toast.success('Analysis results cleared')
  }

  const handleExportPDF = async () => {
    if (!results || !soilImageUrl) {
      toast.error('No analysis results to export')
      return
    }

    try {
      const jsPDF = (await import('jspdf')).default
      const html2canvas = (await import('html2canvas')).default
      
      const pdf = new jsPDF('p', 'mm', 'a4')
      const pageWidth = pdf.internal.pageSize.getWidth()
      const pageHeight = pdf.internal.pageSize.getHeight()
      let yPosition = 20

      // Title
      pdf.setFontSize(18)
      pdf.setFont('helvetica', 'bold')
      pdf.text('FutureFarmNow Soil Analysis Report', pageWidth / 2, yPosition, { align: 'center' })
      yPosition += 15

      // Analysis details
      pdf.setFontSize(12)
      pdf.setFont('helvetica', 'normal')
      const selectedLayer = SOIL_LAYERS.find(l => l.value === selectedSoilLayer)
      pdf.text(`Property Tested: ${selectedLayer?.label || selectedSoilLayer}`, 20, yPosition)
      yPosition += 7
      pdf.text(`Soil Depth: ${selectedSoilDepth} cm`, 20, yPosition)
      yPosition += 7
      pdf.text(`Analysis Date: ${new Date().toLocaleDateString()}`, 20, yPosition)
      yPosition += 15

      // Location details
      if (drawnPolygons && drawnPolygons.length > 0) {
        pdf.setFontSize(12)
        pdf.setFont('helvetica', 'bold')
        pdf.text('Location Information:', 20, yPosition)
        yPosition += 10

        pdf.setFontSize(10)
        pdf.setFont('helvetica', 'normal')

        // Calculate overall bounds from all polygons
        let minLat = Infinity, maxLat = -Infinity
        let minLng = Infinity, maxLng = -Infinity
        let totalApproxArea = 0
        
        drawnPolygons.forEach((polygon) => {
          const coords = polygon.coordinates[0] as number[][] // First ring of polygon
          
          coords.forEach((coord) => {
            const [lng, lat] = coord
            minLat = Math.min(minLat, lat)
            maxLat = Math.max(maxLat, lat)
            minLng = Math.min(minLng, lng)
            maxLng = Math.max(maxLng, lng)
          })
          
          // Area calculation for this polygon
          const polyCoords = polygon.coordinates[0] as number[][]
          let polyMinLat = Infinity, polyMaxLat = -Infinity
          let polyMinLng = Infinity, polyMaxLng = -Infinity
          
          polyCoords.forEach((coord) => {
            const [lng, lat] = coord
            polyMinLat = Math.min(polyMinLat, lat)
            polyMaxLat = Math.max(polyMaxLat, lat)
            polyMinLng = Math.min(polyMinLng, lng)
            polyMaxLng = Math.max(polyMaxLng, lng)
          })
          
          const width = polyMaxLng - polyMinLng
          const height = polyMaxLat - polyMinLat
          const approxAreaDegrees = width * height
          const approxAreaKm2 = approxAreaDegrees * 111 * 111 // Very rough conversion
          totalApproxArea += approxAreaKm2
        })

        const centerLat = (minLat + maxLat) / 2
        const centerLng = (minLng + maxLng) / 2

        pdf.text(`Number of Polygons: ${drawnPolygons.length}`, 20, yPosition)
        yPosition += 6
        pdf.text(`Center Coordinates: ${centerLat.toFixed(6)}°N, ${centerLng.toFixed(6)}°W`, 20, yPosition)
        yPosition += 6
        pdf.text(`Bounding Box: ${minLat.toFixed(6)}° to ${maxLat.toFixed(6)}°N, ${minLng.toFixed(6)}° to ${maxLng.toFixed(6)}°W`, 20, yPosition)
        yPosition += 6
        pdf.text(`Total Approximate Area: ${totalApproxArea.toFixed(2)} km²`, 20, yPosition)
        yPosition += 10

        // WKT String
        pdf.setFontSize(10)
        pdf.setFont('helvetica', 'bold')
        pdf.text('WKT Geometry String:', 20, yPosition)
        yPosition += 6

        // Generate WKT string from GeoJSON
        let wktString
        if (drawnPolygons.length === 1) {
          const coords = drawnPolygons[0].coordinates[0] as number[][]
          const wktCoords = coords.map((coord) => `${coord[0]} ${coord[1]}`).join(', ')
          wktString = `POLYGON((${wktCoords}))`
        } else {
          // MultiPolygon WKT
          const polygonWkts = drawnPolygons.map((polygon) => {
            const coords = polygon.coordinates[0] as number[][]
            const wktCoords = coords.map((coord) => `${coord[0]} ${coord[1]}`).join(', ')
            return `((${wktCoords}))`
          }).join(', ')
          wktString = `MULTIPOLYGON(${polygonWkts})`
        }
        
        pdf.setFontSize(8)
        pdf.setFont('helvetica', 'normal')
        
        // Split WKT string to fit within page width
        const wktLines = pdf.splitTextToSize(wktString, pageWidth - 40)
        pdf.text(wktLines, 20, yPosition)
        yPosition += (wktLines.length * 4) + 10
      }

      // Capture the box plot chart
      try {
        const chartElement = document.querySelector('.soil-analysis-chart') as HTMLElement
        if (chartElement) {
          const chartCanvas = await html2canvas(chartElement, {
            backgroundColor: '#ffffff',
            scale: 2,
            useCORS: true
          })
          
          const chartImgData = chartCanvas.toDataURL('image/png')
          const chartWidth = 160
          const chartHeight = (chartCanvas.height * chartWidth) / chartCanvas.width
          
          if (yPosition + chartHeight > pageHeight - 20) {
            pdf.addPage()
            yPosition = 20
          }

          pdf.setFontSize(12)
          pdf.setFont('helvetica', 'bold')
          pdf.text('Statistical Analysis Chart:', 20, yPosition)
          yPosition += 10

          pdf.addImage(chartImgData, 'PNG', 20, yPosition, chartWidth, chartHeight)
          yPosition += chartHeight + 15
        }
      } catch (error) {
      }

      // Statistical results as text backup
      pdf.setFontSize(14)
      pdf.setFont('helvetica', 'bold')
      pdf.text('Statistical Summary:', 20, yPosition)
      yPosition += 10

      pdf.setFontSize(11)
      pdf.setFont('helvetica', 'normal')
      const stats = [
        `Minimum Value: ${results.min.toFixed(3)}`,
        `Maximum Value: ${results.max.toFixed(3)}`,
        `Mean (Average): ${results.mean.toFixed(3)}`,
        `Median: ${results.median.toFixed(3)}`,
        `Standard Deviation: ${results.stddev.toFixed(3)}`,
        `Data Points: ${results.count.toLocaleString()}`
      ]

      stats.forEach(stat => {
        pdf.text(stat, 25, yPosition)
        yPosition += 6
      })

      yPosition += 10

      // Add soil image directly
      if (soilImageUrl) {
        try {
          if (yPosition + 120 > pageHeight - 20) {
            pdf.addPage()
            yPosition = 20
          }

          pdf.setFontSize(12)
          pdf.setFont('helvetica', 'bold')
          pdf.text('Soil Analysis Visualization:', 20, yPosition)
          yPosition += 10

          // Convert blob URL to base64 for PDF
          const response = await fetch(soilImageUrl)
          const blob = await response.blob()
          const reader = new FileReader()
          
          await new Promise((resolve) => {
            reader.onloadend = () => {
              const base64data = reader.result as string
              const soilWidth = 170
              const soilHeight = 120
              
              pdf.addImage(base64data, 'PNG', 20, yPosition, soilWidth, soilHeight)
              yPosition += soilHeight + 15
              resolve(void 0)
            }
            reader.readAsDataURL(blob)
          })
        } catch (error) {
        }
      }

      // Capture the legend scale component
      try {
        const legendElement = document.querySelector('.soil-legend') as HTMLElement
        if (legendElement) {
          if (yPosition + 60 > pageHeight - 20) {
            pdf.addPage()
            yPosition = 20
          }

          pdf.setFontSize(12)
          pdf.setFont('helvetica', 'bold')
          pdf.text('Soil Value Legend:', 20, yPosition)
          yPosition += 10

          const legendCanvas = await html2canvas(legendElement, {
            backgroundColor: '#ffffff',
            scale: 2,
            useCORS: true
          })
          
          const legendImgData = legendCanvas.toDataURL('image/png')
          const legendWidth = 170
          const legendHeight = (legendCanvas.height * legendWidth) / legendCanvas.width
          
          pdf.addImage(legendImgData, 'PNG', 20, yPosition, legendWidth, legendHeight)
          yPosition += legendHeight + 15
        }
      } catch (error) {
      }

      // Capture the results component with values - ensure full rendering
      try {
        const resultsElement = document.querySelector('.soil-results-summary') as HTMLElement
        if (resultsElement) {
          if (yPosition + 150 > pageHeight - 20) {
            pdf.addPage()
            yPosition = 20
          }

          pdf.setFontSize(12)
          pdf.setFont('helvetica', 'bold')
          pdf.text('Detailed Results:', 20, yPosition)
          yPosition += 10

          const resultsCanvas = await html2canvas(resultsElement, {
            backgroundColor: '#ffffff',
            scale: 2,
            useCORS: true,
            height: resultsElement.scrollHeight, // Ensure full height is captured
            width: resultsElement.scrollWidth
          })
          
          const resultsImgData = resultsCanvas.toDataURL('image/png')
          const resultsWidth = 170
          const resultsHeight = (resultsCanvas.height * resultsWidth) / resultsCanvas.width
          
          pdf.addImage(resultsImgData, 'PNG', 20, yPosition, resultsWidth, resultsHeight)
          yPosition += resultsHeight + 10
        }
      } catch (error) {
      }

      // Description
      pdf.setFontSize(12)
      pdf.setFont('helvetica', 'bold')
      pdf.text('What This Means:', 20, yPosition)
      yPosition += 7

      pdf.setFontSize(10)
      pdf.setFont('helvetica', 'normal')
      const description = selectedLayer?.description || 'Soil property analysis results'
      const splitDescription = pdf.splitTextToSize(description, pageWidth - 40)
      pdf.text(splitDescription, 20, yPosition)

      // Footer
      pdf.setFontSize(8)
      pdf.setFont('helvetica', 'normal')
      pdf.text('Generated by FutureFarmNow - University of California, Riverside', pageWidth / 2, pageHeight - 10, { align: 'center' })

      // Save the PDF
      const filename = `soil-analysis-${selectedSoilLayer}-${selectedSoilDepth}-${new Date().toISOString().split('T')[0]}.pdf`
      pdf.save(filename)
      
      toast.success('PDF report exported successfully!')
    } catch (error) {
      toast.error('Failed to export PDF report')
    }
  }

  // Cleanup image URL on unmount to prevent memory leaks
  useEffect(() => {
    return () => {
      if (soilImageUrl) {
        URL.revokeObjectURL(soilImageUrl)
      }
    }
  }, [soilImageUrl])

  // Update farmland colors when both results and GeoJSON are available
  useEffect(() => {
    if (farmlandResults && farmlandGeoJSONMutation.data && analysisType === 'farmland') {
      // Clear any polygon overlay when showing farmland results
      setSoilImageOverlay(null, null)
      
      setFarmlandGeoJSON({
        geoJSON: farmlandGeoJSONMutation.data,
        colorData: {
          min: farmlandResults.min,
          max: farmlandResults.max,
          farmlands: farmlandResults.farmlands || []
        }
      })
    } else if (analysisType === 'polygon') {
      // Clear farmland overlay when showing polygon results
      setFarmlandGeoJSON(null)
    }
  }, [farmlandResults, farmlandGeoJSONMutation.data, analysisType, setFarmlandGeoJSON, setSoilImageOverlay])

  // Determine if user can analyze farmland based on zoom level
  const canAnalyzeFarmland = currentZoom >= 12 // Show farmland analysis at zoom 12+
  const highZoomLevel = currentZoom >= 14 // High zoom for better precision

  return (
    <div className="p-4 space-y-6">
      {/* Parameters Section */}
      <div className="space-y-4">
        <h3 className="font-medium text-foreground flex items-center">
          <BarChart3 className="h-4 w-4 mr-2" />
          What Would You Like to Test?
        </h3>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-foreground mb-2">
              Choose Soil Property to Test
            </label>
            <Select
              value={selectedSoilLayer}
              onChange={(e) => setSelectedSoilLayer(e.target.value as SoilLayerEnum)}
              data-tutorial="soil-layer-selector"
            >
              {SOIL_LAYERS.map(layer => (
                <SelectOption key={layer.value} value={layer.value} title={layer.description}>
                  {layer.label}
                </SelectOption>
              ))}
            </Select>
            <div className="mt-1 text-xs text-muted-foreground">
              {SOIL_LAYERS.find(l => l.value === selectedSoilLayer)?.description}
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-foreground mb-2">
              Choose Soil Depth to Test
            </label>
            <Select
              value={isCustomDepth ? 'custom' : selectedSoilDepth}
              onChange={(e) => {
                if (e.target.value === 'custom') {
                  setIsCustomDepth(true)
                } else {
                  setIsCustomDepth(false)
                  setSelectedSoilDepth(e.target.value)
                }
              }}
              data-tutorial="soil-depth-selector"
            >
              {SOIL_DEPTHS.map(depth => (
                <SelectOption key={depth.value} value={depth.value} title={depth.description}>
                  {depth.label}
                </SelectOption>
              ))}
            </Select>
            <div className="mt-1 text-xs text-muted-foreground">
              {isCustomDepth ? 'Select your custom depth range below' : SOIL_DEPTHS.find(d => d.value === selectedSoilDepth)?.description}
            </div>
          </div>

          {/* Custom Depth Range Slider */}
          {isCustomDepth && (
            <div className="space-y-4 p-4 bg-muted/30 rounded-lg border">
              <div className="flex items-center space-x-2">
                <Layers className="h-4 w-4 text-primary" />
                <span className="text-sm font-medium">Custom Depth Range</span>
              </div>
              
              <div className="space-y-3">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Start: {customDepthRange[0]} cm</span>
                  <span className="text-muted-foreground">End: {customDepthRange[1]} cm</span>
                </div>
                
                <Slider
                  value={customDepthRange}
                  onValueChange={handleCustomDepthChange}
                  min={0}
                  max={200}
                  step={5}
                  className="w-full"
                />
                
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>0 cm</span>
                  <span className="text-amber-600">100 cm (start limit)</span>
                  <span>200 cm</span>
                </div>
                
                {/* Visual depth indicator */}
                <div className="mt-4 p-3 bg-background rounded-md">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-medium">Selected Range:</span>
                    <span className="text-xs font-bold text-primary">
                      {customDepthRange[0]}-{customDepthRange[1]} cm
                    </span>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {customDepthRange[1] - customDepthRange[0]} cm depth coverage
                    {customDepthRange[0] > 100 && (
                      <span className="text-amber-600 ml-2">(Start limited to 100cm)</span>
                    )}
                  </div>
                  
                  {/* Constraint info */}
                  <div className="mt-2 p-2 bg-muted/30 rounded text-xs">
                    <div className="flex items-center space-x-1 text-muted-foreground">
                      <span>💡</span>
                      <span>Start depth: 0-100cm | End depth: {Math.max(customDepthRange[0] + 5, 5)}-200cm</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Analysis Buttons */}
      <div className="space-y-3" data-tutorial="soil-analysis-buttons">
        {/* Polygon Analysis Button */}
        <Button
          onClick={handleAnalyze}
          disabled={!drawnPolygons || drawnPolygons.length === 0 || soilAnalysisMutation.isPending || soilImageMutation.isPending}
          className="w-full"
          data-tutorial="analyze-soil-button"
        >
          {(soilAnalysisMutation.isPending || soilImageMutation.isPending) ? (
            <>
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
              Analyzing...
            </>
          ) : (
            <>
              <MapPin className="h-4 w-4 mr-2" />
              Analyze Selected Area
            </>
          )}
        </Button>

        {/* Farmland Analysis Button (only show at high zoom) */}
        {canAnalyzeFarmland && (
          <Button
            onClick={handleFarmlandAnalyze}
            disabled={farmlandAnalysisMutation.isPending}
            variant="outline"
            className="w-full"
            data-tutorial="analyze-farmland-button"
          >
            {farmlandAnalysisMutation.isPending ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-current mr-2" />
                Analyzing Farmland...
              </>
            ) : (
              <>
                <BarChart3 className="h-4 w-4 mr-2" />
                Analyze All Farmland in View
              </>
            )}
          </Button>
        )}

        {/* Status Messages */}
        {!drawnPolygon && !canAnalyzeFarmland && (
          <div className="mt-2 flex items-start space-x-2 text-amber-600 text-sm">
            <AlertCircle className="h-4 w-4 mt-0.5 flex-shrink-0" />
            <p>Draw a polygon on the map to enable analysis or zoom in further to query all farmlands</p>
          </div>
        )}
        
        {!drawnPolygon && canAnalyzeFarmland && (
          <div className="mt-2 flex items-start space-x-2 text-blue-600 text-sm">
            <AlertCircle className="h-4 w-4 mt-0.5 flex-shrink-0" />
            <p>You can draw a specific area or analyze all farmland visible in the current view</p>
          </div>
        )}
      </div>

      {/* Results Section - Polygon Analysis */}
      {results && analysisType === 'polygon' && (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold text-foreground flex items-center">
              <BarChart3 className="h-5 w-5 mr-2 text-green-600" />
              Your Soil Test Results
            </h3>
            <div className="flex items-center space-x-2">
              <Button
                variant="outline"
                size="sm"
                onClick={handleExportPDF}
                className="text-xs"
              >
                <Download className="h-3 w-3 mr-1" />
                Export PDF
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={handleReset}
                className="text-xs"
              >
                Clear Results
              </Button>
            </div>
          </div>
          
          <div className="bg-muted/30 p-4 rounded-lg border soil-analysis-chart">
            <div className="mb-3 text-sm text-muted-foreground">
              Visual breakdown of your soil values across the tested area:
            </div>
            <BoxWhiskerPlot results={results} />
          </div>

          <div className="bg-card p-4 rounded-lg border space-y-4 soil-results-summary">
            <h4 className="font-medium text-foreground mb-3">Key Numbers for Your Farm:</h4>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-muted/20 p-3 rounded-lg">
                <dt className="text-sm font-medium text-muted-foreground">Average Value:</dt>
                <dd className="text-lg font-semibold text-foreground">{results.mean?.toFixed(2)}</dd>
                <div className="text-xs text-muted-foreground mt-1">Most common level in your field</div>
              </div>
              
              <div className="bg-muted/20 p-3 rounded-lg">
                <dt className="text-sm font-medium text-muted-foreground">Middle Value:</dt>
                <dd className="text-lg font-semibold text-foreground">{results.median?.toFixed(2)}</dd>
                <div className="text-xs text-muted-foreground mt-1">Half your field is above/below this</div>
              </div>
              
              <div className="bg-muted/20 p-3 rounded-lg">
                <dt className="text-sm font-medium text-muted-foreground">Lowest Spot:</dt>
                <dd className="text-lg font-semibold text-foreground">{results.min?.toFixed(2)}</dd>
                <div className="text-xs text-muted-foreground mt-1">Your field's minimum value</div>
              </div>
              
              <div className="bg-muted/20 p-3 rounded-lg">
                <dt className="text-sm font-medium text-muted-foreground">Highest Spot:</dt>
                <dd className="text-lg font-semibold text-foreground">{results.max?.toFixed(2)}</dd>
                <div className="text-xs text-muted-foreground mt-1">Your field's maximum value</div>
              </div>
              
              <div className="bg-muted/20 p-3 rounded-lg">
                <dt className="text-sm font-medium text-muted-foreground">Variation:</dt>
                <dd className="text-lg font-semibold text-foreground">{results.stddev?.toFixed(2)}</dd>
                <div className="text-xs text-muted-foreground mt-1">How much values differ across your field</div>
              </div>
              
              <div className="bg-muted/20 p-3 rounded-lg">
                <dt className="text-sm font-medium text-muted-foreground">Data Points:</dt>
                <dd className="text-lg font-semibold text-foreground">{results.count?.toLocaleString()}</dd>
                <div className="text-xs text-muted-foreground mt-1">Number of measurements taken</div>
              </div>
            </div>

            <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-950/30 rounded-lg border-l-4 border-blue-400">
              <div className="text-sm text-blue-800 dark:text-blue-200">
                <strong>💡 What this means:</strong> These numbers help you understand how your soil varies across your farm. 
                Look for big differences between lowest and highest spots - those areas might need different treatment.
              </div>
            </div>
          </div>

          {/* Soil Value Legend */}
          {soilImageUrl && results && (
            <SoilLegend 
              soilLayer={SOIL_LAYERS.find(l => l.value === selectedSoilLayer)?.label || selectedSoilLayer}
              min={results.min}
              max={results.max}
            />
          )}

          {/* How to read the map - Only with polygon results */}
          {soilImageUrl && results && (
            <div className="p-3 bg-green-50 dark:bg-green-950/30 rounded-lg border-l-4 border-green-400">
              <div className="text-sm text-green-800 dark:text-green-200">
                <strong>🗺️ How to read the map:</strong> Different colors on the map represent different soil values. 
                Use the colored overlay on the map alongside the analysis results above to identify patterns and plan targeted treatments for your field.
              </div>
            </div>
          )}

          {/* Loading indicator for image */}
          {soilImageMutation.isPending && (
            <div className="bg-card p-4 rounded-lg border">
              <div className="flex items-center space-x-3">
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-primary"></div>
                <div>
                  <h4 className="font-medium text-foreground">Loading soil visualization...</h4>
                  <p className="text-sm text-muted-foreground">Creating color-coded map of your soil data</p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Results Section - Farmland Analysis (Scale Only) */}
      {farmlandResults && analysisType === 'farmland' && (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold text-foreground flex items-center">
              <BarChart3 className="h-5 w-5 mr-2 text-blue-600" />
              Farmland Analysis Results
            </h3>
            <div className="flex items-center space-x-2">
              <Button
                variant="outline"
                size="sm"
                onClick={handleReset}
                className="text-xs"
              >
                Clear Results
              </Button>
            </div>
          </div>

          {/* Farmland Analysis Legend */}
          {farmlandResults?.min !== undefined && farmlandResults?.max !== undefined && (
            <SoilLegend 
              soilLayer={SOIL_LAYERS.find(l => l.value === selectedSoilLayer)?.label || selectedSoilLayer}
              min={farmlandResults.min}
              max={farmlandResults.max}
            />
          )}

          <div className="p-3 bg-blue-50 dark:bg-blue-950/30 rounded-lg border-l-4 border-blue-400">
            <div className="text-sm text-blue-800 dark:text-blue-200">
              <strong>📊 Farmland Analysis:</strong> This shows the soil value range across all farmland visible in your current map view. 
              Zoom in and draw a specific area for detailed statistics and visualizations.
            </div>
          </div>

          {/* Loading indicator for farmland analysis */}
          {(farmlandAnalysisMutation.isPending || farmlandGeoJSONMutation.isPending) && (
            <div className="bg-card p-4 rounded-lg border">
              <div className="flex items-center space-x-3">
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-primary"></div>
                <div>
                  <h4 className="font-medium text-foreground">Loading farmland analysis...</h4>
                  <p className="text-sm text-muted-foreground">Analyzing all farmland in current view</p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Legend */}
      {legendData && (
        <div className="space-y-2">
          <h3 className="font-medium text-gray-900">Legend</h3>
          <Legend data={legendData} />
        </div>
      )}

    </div>
  )
}