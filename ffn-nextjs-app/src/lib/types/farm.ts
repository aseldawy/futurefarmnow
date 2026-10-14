import { GeoJSONGeometry } from './api';

export interface Farm {
  id: string;
  name: string;
  geometry: GeoJSONGeometry;
  center: [number, number]; // [lat, lng]
  zoom: number;
  createdAt: string; // ISO date string
  updatedAt: string; // ISO date string
  dataset: string;
}
