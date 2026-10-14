'use client'

import { X, ChevronLeft, ChevronRight, HelpCircle, MapPin, Trash2, Maximize2, Minimize2, BookOpen, Copy, Check, Home, Save } from 'lucide-react'
import Image from 'next/image'
import dynamic from 'next/dynamic'
import { Button } from '@/components/ui/button'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { ThemeToggle } from '@/components/ui/theme-toggle'
import { DatasetSelector } from './DatasetSelector'
import { FarmPanel } from '@/components/farm/FarmPanel'
import { useMapStore } from '@/lib/stores/mapStore'
import { useFarmStore } from '@/lib/stores/farmStore'
import { useState } from 'react'
import { toast } from '@/lib/utils/toast'
import { cn } from '@/lib/utils'
import { Loading } from '@/components/common/Loading'

// Dynamic imports for code splitting - only load analysis panels when needed
// Using named exports with dynamic import wrapper
const SoilAnalysis = dynamic(
  () => import('./SoilAnalysis').then(mod => ({ default: mod.SoilAnalysis })),
  {
    loading: () => <Loading />,
    ssr: false
  }
)

const NDVIAnalysis = dynamic(
  () => import('./NDVIAnalysis').then(mod => ({ default: mod.NDVIAnalysis })),
  {
    loading: () => <Loading />,
    ssr: false
  }
)

const SamplePoints = dynamic(
  () => import('./SamplePoints').then(mod => ({ default: mod.SamplePoints })),
  {
    loading: () => <Loading />,
    ssr: false
  }
)

const ETMapAnalysis = dynamic(
  () => import('./ETMapAnalysis').then(mod => ({ default: mod.ETMapAnalysis })),
  {
    loading: () => <Loading />,
    ssr: false
  }
)

interface SidebarProps {
  onClose?: () => void
  onShowHelp?: () => void
  onClearAnalysis?: () => void
  onStartTutorial?: () => void
}

export function Sidebar({ onClose, onShowHelp, onClearAnalysis, onStartTutorial }: SidebarProps) {
  const { activeTab, setActiveTab, drawnPolygon, drawnPolygons } = useMapStore()
  const setIsAddingFarm = useFarmStore((s) => s.setIsAddingFarm)
  const [isCollapsed, setIsCollapsed] = useState(false)
  const [isExpanded, setIsExpanded] = useState(true)
  const [copiedGeoJSON, setCopiedGeoJSON] = useState(false)

  return (
    <div className={cn(
      "h-full flex flex-col bg-background border-r transition-all duration-300 ease-in-out",
      isCollapsed ? "w-16" : isExpanded ? "w-[480px]" : "w-96"
    )}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b min-h-[73px]">
        <div className="flex items-center space-x-2 overflow-hidden">
          <div className="text-2xl flex-shrink-0" role="img" aria-label="Farm">
            🌾
          </div>
          {!isCollapsed && (
            <h1 className="text-lg font-semibold text-foreground whitespace-nowrap">
              FutureFarmNow
            </h1>
          )}
        </div>
        <div className="flex items-center space-x-1" data-tutorial="sidebar-controls">
          {!isCollapsed && (
            <>
              <Button
                variant="ghost"
                size="icon"
                onClick={onShowHelp}
                title="Help"
                data-tutorial="help-button"
              >
                <HelpCircle className="h-4 w-4" />
              </Button>

              <Button
                variant="ghost"
                size="icon"
                onClick={onStartTutorial}
                title="Start Tutorial"
                className="bg-gradient-to-r from-blue-500/10 to-green-500/10 hover:from-blue-500/20 hover:to-green-500/20"
              >
                <BookOpen className="h-4 w-4" />
              </Button>

              <ThemeToggle />
            </>
          )}

          {/* Desktop expand toggle */}
          {!isCollapsed && (
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setIsExpanded(!isExpanded)}
              className="hidden md:flex"
              title={isExpanded ? "Contract sidebar" : "Expand sidebar"}
            >
              {isExpanded ? (
                <Minimize2 className="h-4 w-4" />
              ) : (
                <Maximize2 className="h-4 w-4" />
              )}
            </Button>
          )}

          {/* Desktop collapse toggle */}
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setIsCollapsed(!isCollapsed)}
            className="hidden md:flex"
            title={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {isCollapsed ? (
              <ChevronRight className="h-4 w-4" />
            ) : (
              <ChevronLeft className="h-4 w-4" />
            )}
          </Button>

          {/* Mobile close button */}
          {onClose && (
            <Button
              variant="ghost"
              size="icon"
              onClick={onClose}
              className="md:hidden"
              title="Close sidebar"
            >
              <X className="h-4 w-4" />
            </Button>
          )}
        </div>
      </div>

      {/* Area Status and Clear Analysis - Combined */}
      {!isCollapsed && (
        <div className="px-4 py-3 border-b">
          <div className="flex items-center justify-between gap-2">
            {/* Area Status */}
            {drawnPolygon ? (
              <div className="flex items-center space-x-1.5 text-green-600 dark:text-green-400 min-w-0">
                <MapPin className="h-3.5 w-3.5 flex-shrink-0" />
                <span className="text-xs font-medium truncate">Area Selected ✓</span>
              </div>
            ) : (
              <div className="flex items-center space-x-1.5 text-muted-foreground min-w-0">
                <MapPin className="h-3.5 w-3.5 flex-shrink-0" />
                <span className="text-xs font-medium truncate">No area selected</span>
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex items-center space-x-1 flex-shrink-0">
              {/* Save as Farm Button */}
              {drawnPolygon && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setIsAddingFarm(true)}
                  className="text-xs px-2 py-1 h-7"
                  title="Save current area as a farm"
                >
                  <Save className="h-3 w-3 sm:mr-1" />
                  <span className="hidden sm:inline">Save Farm</span>
                </Button>
              )}

              {/* Copy GeoJSON Button - only show when area is selected */}
              {drawnPolygon && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    const geoJsonData = drawnPolygons && drawnPolygons.length > 1
                      ? {
                        type: 'MultiPolygon',
                        coordinates: drawnPolygons.map(p => p.coordinates)
                      }
                      : drawnPolygons && drawnPolygons.length === 1
                        ? drawnPolygons[0]
                        : drawnPolygon;

                    navigator.clipboard.writeText(JSON.stringify(geoJsonData, null, 2))
                      .then(() => {
                        setCopiedGeoJSON(true)
                        toast.success('GeoJSON copied to clipboard!')
                        setTimeout(() => setCopiedGeoJSON(false), 2000)
                      })
                      .catch(() => {
                        toast.error('Failed to copy GeoJSON')
                      })
                  }}
                  className="text-xs px-2 py-1 h-7"
                  title="Copy GeoJSON to clipboard"
                >
                  {copiedGeoJSON ? (
                    <>
                      <Check className="h-3 w-3 mr-1 text-green-600" />
                      <span className="hidden sm:inline">Copied!</span>
                    </>
                  ) : (
                    <>
                      <Copy className="h-3 w-3 sm:mr-1" />
                      <span className="hidden sm:inline">Copy</span>
                    </>
                  )}
                </Button>
              )}

              {/* Clear All Button - clears both results AND selected area */}
              <Button
                variant="outline"
                size="sm"
                onClick={onClearAnalysis}
                className="text-xs px-2 py-1 h-7"
                title="Clear all results and selected area"
              >
                <Trash2 className="h-3 w-3 sm:mr-1" />
                <span className="hidden sm:inline">Clear All</span>
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Collapsed state - only show icons */}
      {isCollapsed ? (
        <div className="flex flex-col items-center py-4 space-y-4">
          {/* Area Status Icon */}
          <div className="flex items-center justify-center" title={drawnPolygon ? "Area Selected" : "No Area Selected"}>
            {drawnPolygon ? (
              <MapPin className="h-5 w-5 text-green-600 dark:text-green-400" />
            ) : (
              <MapPin className="h-5 w-5 text-muted-foreground" />
            )}
          </div>

          {/* Farms icon */}
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setIsCollapsed(false)}
            title="My Farms"
          >
            <Home className="h-4 w-4" />
          </Button>

          <Button
            variant="ghost"
            size="icon"
            onClick={onShowHelp}
            title="Help"
          >
            <HelpCircle className="h-4 w-4" />
          </Button>
          <Button
            variant={activeTab === 'soil' ? 'default' : 'ghost'}
            size="icon"
            onClick={() => setActiveTab('soil')}
            title="Soil Analysis"
            className="text-xs font-medium"
          >
            S
          </Button>
          <Button
            variant={activeTab === 'ndvi' ? 'default' : 'ghost'}
            size="icon"
            onClick={() => setActiveTab('ndvi')}
            title="NDVI Analysis"
            className="text-xs font-medium"
          >
            N
          </Button>
          <Button
            variant={activeTab === 'sample' ? 'default' : 'ghost'}
            size="icon"
            onClick={() => setActiveTab('sample')}
            title="Sample Points"
            className="text-xs font-medium"
          >
            P
          </Button>
          <Button
            variant={activeTab === 'etmap' ? 'default' : 'ghost'}
            size="icon"
            onClick={() => setActiveTab('etmap')}
            title="ET Map"
            className="text-xs font-medium"
          >
            ET
          </Button>
        </div>
      ) : (
        <>
          {/* My Farms */}
          <div id="sidebar-farm-panel" className="px-4 py-3">
            <FarmPanel />
          </div>

          {/* Dataset Selection */}
          <div className="px-4 py-3 border-b bg-muted/50" data-tutorial="dataset-selector">
            <DatasetSelector />
          </div>

          {/* Analysis Tabs */}
          <div className="flex-1 overflow-hidden flex flex-col">
            <div className="px-4 py-3 border-b" data-tutorial="analysis-tabs">
              <Tabs className="w-full">
                <TabsList className="grid grid-cols-4 w-full">
                  <TabsTrigger
                    value="soil"
                    active={activeTab === 'soil'}
                    onClick={() => setActiveTab('soil')}
                  >
                    Soil
                  </TabsTrigger>
                  <TabsTrigger
                    value="ndvi"
                    active={activeTab === 'ndvi'}
                    onClick={() => setActiveTab('ndvi')}
                  >
                    NDVI
                  </TabsTrigger>
                  <TabsTrigger
                    value="sample"
                    active={activeTab === 'sample'}
                    onClick={() => setActiveTab('sample')}
                  >
                    Sample
                  </TabsTrigger>
                  <TabsTrigger
                    value="etmap"
                    active={activeTab === 'etmap'}
                    onClick={() => setActiveTab('etmap')}
                  >
                    ETMap
                  </TabsTrigger>
                </TabsList>
              </Tabs>
            </div>

            {/* Tab Content */}
            <div className="flex-1 overflow-y-auto">
              <Tabs>
                <TabsContent value="soil" active={activeTab === 'soil'} data-tutorial="soil-panel">
                  <SoilAnalysis />
                </TabsContent>

                <TabsContent value="ndvi" active={activeTab === 'ndvi'} data-tutorial="ndvi-panel">
                  <NDVIAnalysis />
                </TabsContent>

                <TabsContent value="sample" active={activeTab === 'sample'}>
                  <SamplePoints />
                </TabsContent>

                <TabsContent value="etmap" active={activeTab === 'etmap'}>
                  <ETMapAnalysis />
                </TabsContent>
              </Tabs>
            </div>
          </div>

          {/* Footer */}
          <div className="p-4 border-t space-y-3">
            <div className="flex items-center justify-center space-x-3">
              <Image
                src="/UCR_logo.png"
                alt="UC Riverside"
                width={120}
                height={32}
                className="h-8 w-auto"
                priority
              />
              <a
                href="https://ai4sa.ucr.edu"
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 font-medium transition-colors"
              >
                AI4SA
              </a>
            </div>

            {/* Copyright */}
            <div className="text-center text-xs text-muted-foreground">
              <p>© 2025 FutureFarmNow</p>
              <p>University of California, Riverside</p>
            </div>
          </div>
        </>
      )}

    </div>
  )
}