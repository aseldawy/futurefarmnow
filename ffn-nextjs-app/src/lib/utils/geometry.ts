import { GeoJSONGeometry } from '../types/api';

/**
 * Compute a simple centroid from the outer ring of a polygon geometry.
 * Returns [lat, lng] suitable for Weather.gov API and Leaflet.
 */
export function computePolygonCenter(geometry: GeoJSONGeometry): [number, number] {
  let coords: number[][];

  if (geometry.type === 'Polygon') {
    coords = geometry.coordinates[0] as number[][];
  } else if (geometry.type === 'MultiPolygon') {
    // Use the first polygon's outer ring
    coords = (geometry.coordinates[0] as number[][][])[0];
  } else {
    return [0, 0];
  }

  let sumLng = 0;
  let sumLat = 0;
  // Exclude the closing coordinate (same as first)
  const n = coords.length - 1;

  for (let i = 0; i < n; i++) {
    sumLng += coords[i][0];
    sumLat += coords[i][1];
  }

  return [sumLat / n, sumLng / n];
}
