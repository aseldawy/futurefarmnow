'use client';

import React from 'react';
import { MapPin } from 'lucide-react';
import { useFarmStore } from '@/lib/stores/farmStore';
import { FarmListItem } from './FarmListItem';

export function FarmList() {
  const farms = useFarmStore((s) => s.farms);

  if (farms.length === 0) {
    return (
      <div id="farm-list-empty" className="flex flex-col items-center py-4 text-center">
        <MapPin className="h-6 w-6 text-muted-foreground mb-1.5" />
        <p className="text-xs text-muted-foreground">
          No farms saved yet. Draw a polygon on the map and save it as a farm.
        </p>
      </div>
    );
  }

  return (
    <div id="farm-list" className="max-h-40 overflow-y-auto space-y-1 scrollbar-thin">
      {farms.map((farm) => (
        <FarmListItem key={farm.id} farm={farm} />
      ))}
    </div>
  );
}
