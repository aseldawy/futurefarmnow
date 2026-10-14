'use client';

import React, { useState, useCallback } from 'react';
import { Trash2 } from 'lucide-react';
import { Farm } from '@/lib/types/farm';
import { useFarmStore } from '@/lib/stores/farmStore';
import { useMapStore } from '@/lib/stores/mapStore';

interface FarmListItemProps {
  farm: Farm;
}

export function FarmListItem({ farm }: FarmListItemProps) {
  const [confirmDelete, setConfirmDelete] = useState(false);
  const selectedFarmId = useFarmStore((s) => s.selectedFarmId);
  const selectFarm = useFarmStore((s) => s.selectFarm);
  const removeFarm = useFarmStore((s) => s.removeFarm);
  const setIsWeatherPopupOpen = useFarmStore((s) => s.setIsWeatherPopupOpen);
  const mapInstance = useMapStore((s) => s.mapInstance);
  const drawnItems = useMapStore((s) => s.drawnItems);
  const addDrawnPolygon = useMapStore((s) => s.addDrawnPolygon);
  const clearArea = useMapStore((s) => s.clearArea);
  const isSelected = selectedFarmId === farm.id;

  const handleClick = useCallback(() => {
    // Toggle: if already selected, unselect
    if (isSelected) {
      selectFarm(null);
      clearArea();
      return;
    }

    // Clear existing drawn area first
    if (drawnItems) {
      drawnItems.clearLayers();
    }

    // Select the farm and open weather
    selectFarm(farm.id);
    setIsWeatherPopupOpen(true);

    if (mapInstance) {
      try {
        const L = (window as unknown as { L: typeof import('leaflet') }).L;

        // Draw the farm geometry onto drawnItems so analysis features work
        const geoLayer = L.geoJSON(farm.geometry as GeoJSON.GeoJsonObject, {
          style: {
            color: '#3b82f6',
            weight: 2,
            opacity: 1,
            fillColor: '#3b82f6',
            fillOpacity: 0,
          },
        });

        if (drawnItems) {
          geoLayer.eachLayer((layer) => {
            drawnItems.addLayer(layer);
          });
        }

        // Set the polygon in mapStore so analysis panels can use it
        addDrawnPolygon(farm.geometry);

        // Navigate map to the farm
        const bounds = L.geoJSON(farm.geometry as GeoJSON.GeoJsonObject).getBounds();
        mapInstance.fitBounds(bounds.pad(0.2), {
          maxZoom: farm.zoom,
          animate: true,
        });
      } catch {
        mapInstance.setView(farm.center, farm.zoom, { animate: true });
      }
    }
  }, [farm, isSelected, selectFarm, setIsWeatherPopupOpen, mapInstance, drawnItems, addDrawnPolygon, clearArea]);

  const handleDelete = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      if (confirmDelete) {
        removeFarm(farm.id);
      } else {
        setConfirmDelete(true);
        setTimeout(() => setConfirmDelete(false), 3000);
      }
    },
    [confirmDelete, farm.id, removeFarm]
  );

  return (
    <div
      id={`farm-item-${farm.id}`}
      onClick={handleClick}
      className={`flex items-center justify-between px-2 py-1.5 rounded-md cursor-pointer transition-colors group ${
        isSelected
          ? 'bg-primary/10 border border-primary/30'
          : 'hover:bg-muted border border-transparent'
      }`}
    >
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium truncate">{farm.name}</p>
        <p className="text-[10px] text-muted-foreground capitalize">{farm.dataset}</p>
      </div>

      <button
        onClick={handleDelete}
        className={`ml-2 p-1 rounded transition-colors flex-shrink-0 ${
          confirmDelete
            ? 'text-destructive bg-destructive/10'
            : 'text-muted-foreground opacity-0 group-hover:opacity-100 hover:text-destructive'
        }`}
        title={confirmDelete ? 'Click again to confirm delete' : 'Delete farm'}
      >
        <Trash2 className="h-3.5 w-3.5" />
      </button>
    </div>
  );
}
