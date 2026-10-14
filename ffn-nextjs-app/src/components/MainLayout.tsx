'use client'

import { useState } from 'react'
import { MapContainer } from '@/components/map/MapContainer'
import { Sidebar } from '@/components/panels/Sidebar'
import { HelpModal } from '@/components/common/HelpModal'
import { TutorialOverlay, TutorialBodyClass } from '@/components/tutorial/TutorialOverlay'
import { Button } from '@/components/ui/button'
import { Menu } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useMapStore } from '@/lib/stores/mapStore'
import { useFarmStore } from '@/lib/stores/farmStore'
import { toast } from '@/lib/utils/toast'
import { useURLSync } from '@/hooks/useURLSync'
import { useWeather } from '@/hooks/useWeather'
import { useTutorial } from '@/lib/contexts/TutorialContext'
import { ChatWidget } from '@/components/ai/ChatWidget'

export function MainLayout() {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [helpModalOpen, setHelpModalOpen] = useState(false)
  const { clearArea, clearAllAnalysis } = useMapStore()
  const { startTutorial } = useTutorial()
  const selectFarm = useFarmStore((s) => s.selectFarm)

  // Initialize URL synchronization
  useURLSync()

  // Initialize weather hook
  useWeather()

  const handleClearArea = () => {
    clearArea()
    selectFarm(null)
    toast.success('Area cleared! You can now draw a new area.')
  }

  const handleClearAnalysis = () => {
    clearAllAnalysis()
    selectFarm(null)
    toast.success('All analysis results cleared!')
  }

  return (
    <div className="h-screen overflow-hidden bg-background main-container" data-tutorial="main-container" style={{ position: 'relative' }}>
      {/* Tutorial system */}
      <TutorialOverlay />
      <TutorialBodyClass />
      
      {/* Menu button - show when sidebar is closed */}
      {!sidebarOpen && (
        <div className="fixed top-4 left-4" style={{ zIndex: 10000 }}>
          <Button
            variant="outline"
            size="icon"
            onClick={() => setSidebarOpen(true)}
            className="bg-background shadow-md border"
          >
            <Menu className="h-4 w-4" />
          </Button>
        </div>
      )}

      {/* Sidebar - Always fixed positioned */}
      {sidebarOpen && (
        <aside 
          className="fixed inset-y-0 left-0 shadow-lg bg-background animate-in slide-in-from-left duration-300" 
          style={{ 
            zIndex: 9999,
            backgroundColor: 'var(--background)',
            border: '1px solid var(--border)'
          }}
        >
          <Sidebar 
            onClose={() => setSidebarOpen(false)}
            onShowHelp={() => setHelpModalOpen(true)}
            onClearAnalysis={handleClearAnalysis}
            onStartTutorial={startTutorial}
          />
        </aside>
      )}

      {/* Overlay - show on mobile when sidebar is open */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-30 bg-black bg-opacity-25 md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Main content - Map (always full width, sidebar overlays it) */}
      <main className="h-full w-full" data-tutorial="map-container">
        <MapContainer />
      </main>

      {/* Help Modal - Global */}
      <HelpModal isOpen={helpModalOpen} onClose={() => setHelpModalOpen(false)} />

      {/* FarmBot AI Chat Widget */}
      <ChatWidget />
    </div>
  )
}