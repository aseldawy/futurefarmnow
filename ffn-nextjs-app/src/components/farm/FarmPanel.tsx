'use client';

import React from 'react';
import { Home, Plus } from 'lucide-react';
import { useFarmStore } from '@/lib/stores/farmStore';
import { FarmList } from './FarmList';
import { AddFarmDialog } from './AddFarmDialog';

export function FarmPanel() {
  const farms = useFarmStore((s) => s.farms);
  const isAddingFarm = useFarmStore((s) => s.isAddingFarm);
  const setIsAddingFarm = useFarmStore((s) => s.setIsAddingFarm);

  return (
    <div id="farm-panel" className="border-b border-border pb-3">
      <div id="farm-panel-header" className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Home className="h-4 w-4 text-primary" />
          <span className="text-sm font-semibold">My Farms</span>
          {farms.length > 0 && (
            <span className="text-xs text-muted-foreground bg-muted px-1.5 py-0.5 rounded-full">
              {farms.length}
            </span>
          )}
        </div>
        <button
          id="add-farm-btn"
          onClick={() => setIsAddingFarm(true)}
          className="flex items-center gap-1 text-xs text-primary hover:text-primary/80 transition-colors"
          title="Add Farm"
        >
          <Plus className="h-3.5 w-3.5" />
          <span>Add Farm</span>
        </button>
      </div>

      {isAddingFarm && <AddFarmDialog />}
      <FarmList />
    </div>
  );
}
