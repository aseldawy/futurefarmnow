'use client';

import React, { useCallback } from 'react';
import { Check, X } from 'lucide-react';
import { useFarmStore } from '@/lib/stores/farmStore';
import { useMapStore } from '@/lib/stores/mapStore';
import { computePolygonCenter } from '@/lib/utils/geometry';
import { Farm } from '@/lib/types/farm';
import toast from 'react-hot-toast';

export function AddFarmDialog() {
  const pendingFarmName = useFarmStore((s) => s.pendingFarmName);
  const setPendingFarmName = useFarmStore((s) => s.setPendingFarmName);
  const setIsAddingFarm = useFarmStore((s) => s.setIsAddingFarm);
  const addFarm = useFarmStore((s) => s.addFarm);
  const selectFarm = useFarmStore((s) => s.selectFarm);

  const drawnPolygon = useMapStore((s) => s.drawnPolygon);
  const selectedDataset = useMapStore((s) => s.selectedDataset);
  const currentZoom = useMapStore((s) => s.currentZoom);

  const handleSave = useCallback(() => {
    const name = pendingFarmName.trim();
    if (!name) {
      toast.error('Please enter a farm name.');
      return;
    }
    if (!drawnPolygon) {
      toast.error('Draw a polygon on the map first.');
      return;
    }

    const now = new Date().toISOString();
    const center = computePolygonCenter(drawnPolygon);

    const farm: Farm = {
      id: crypto.randomUUID(),
      name,
      geometry: drawnPolygon,
      center,
      zoom: currentZoom,
      createdAt: now,
      updatedAt: now,
      dataset: selectedDataset ?? 'farmland',
    };

    addFarm(farm);
    selectFarm(farm.id);
    setIsAddingFarm(false);
    toast.success('Farm saved!');
  }, [
    pendingFarmName,
    drawnPolygon,
    currentZoom,
    selectedDataset,
    addFarm,
    selectFarm,
    setIsAddingFarm,
  ]);

  const handleCancel = useCallback(() => {
    setIsAddingFarm(false);
  }, [setIsAddingFarm]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter') handleSave();
      if (e.key === 'Escape') handleCancel();
    },
    [handleSave, handleCancel]
  );

  return (
    <div id="add-farm-dialog" className="flex items-center gap-1.5 mb-2">
      <input
        autoFocus
        type="text"
        placeholder="Farm name..."
        value={pendingFarmName}
        onChange={(e) => setPendingFarmName(e.target.value)}
        onKeyDown={handleKeyDown}
        className="flex-1 h-8 px-2 text-sm rounded-md border border-input bg-background focus:outline-none focus:ring-1 focus:ring-primary"
      />
      <button
        onClick={handleSave}
        className="h-8 w-8 flex items-center justify-center rounded-md bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
        title="Save farm"
      >
        <Check className="h-4 w-4" />
      </button>
      <button
        onClick={handleCancel}
        className="h-8 w-8 flex items-center justify-center rounded-md border border-input hover:bg-muted transition-colors"
        title="Cancel"
      >
        <X className="h-4 w-4" />
      </button>
    </div>
  );
}
