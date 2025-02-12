import rasterio
import numpy as np
from rasterio.mask import mask
from shapely.wkt import loads as wkt_loads
import os
import pandas as pd
from pyproj import Proj, transform
from pyproj import Proj, Transformer
def extract_pixel_coords(input_tif_path, geometry, target_crs="EPSG:4326", nodata = -9999):
    with rasterio.open(input_tif_path) as src:
        src_crs = src.crs
        
        masked_data, masked_transform = mask(src, [geometry], crop=True)
        masked_band = masked_data[0]
        
        nrows, ncols = masked_band.shape
        pixel_width = masked_transform.a
        pixel_height = masked_transform.e
        left, top = masked_transform.c, masked_transform.f
        
        cols, rows = np.meshgrid(np.arange(ncols), np.arange(nrows))
        x = left + cols * pixel_width + (pixel_width / 2)
        y = top + rows * pixel_height + (pixel_height / 2)
        x = np.array(x.ravel())
        y = np.array(y.ravel())
        values = np.array(masked_band.ravel())
        values[values == nodata] = np.nan
        if src_crs != target_crs:
            proj_from = Proj(src_crs) 
            proj_to = Proj(target_crs) 
            x, y = transform(proj_from, proj_to, x, y)

        return x, y, values

def find_matching_tif_files(index_csv_path, input_wkt, input_crs = 4326):
    # Read the CSV with the index of geometries and files
    df = pd.read_csv(index_csv_path, delimiter=';')
    
    input_geometry = wkt_loads(input_wkt)    
    matching_files = []

    for _, row in df.iterrows():
        row_geometry = wkt_loads(row['Geometry4326'])
        row_crs = row['SRID']

        if row_crs != input_crs:
            # Use pyproj to transform CRS if they are different
            transformer = Transformer.from_crs(row_crs, 4326, always_xy=True)
            transformed_geom = row_geometry
            # Assuming row_geometry is a shapely geometry, transform its coordinates
            transformed_coords = [transformer.transform(x, y) for x, y in zip(row_geometry.xy[0], row_geometry.xy[1])]
            transformed_geom = row_geometry.__class__(transformed_coords)  # Create a new geometry from transformed coordinates
        else:
            transformed_geom = row_geometry

        # Check if the input geometry intersects with the transformed geometry
        if input_geometry.intersects(transformed_geom):
            matching_files.append(row['FileName'])

    if matching_files:
        return matching_files
    else:
        print("No matching files found.")
        return []

#No Indexing by default
def output_from_attr(input_dir, geometry, depth_range, attribute_list=[], num_samples=0, output_name='output'):
    output_df = pd.DataFrame({'x': [], 'y': []})
    
    import soil
    for layer in attribute_list:
        matching_subdirs = soil.get_matching_subdirectories(input_dir, depth_range, layer)
        explore_depths_list = [
                                  (name, int(yyy) - int(xxx))
                                  for name in matching_subdirs
                                  if (parts := os.path.basename(name).split('_')) and len(parts) >= 3
                                  for xxx, yyy in [(parts[0], parts[1])]
                              ]
        
        print(f"Found explore depths for '{layer}': {explore_depths_list}")

        if explore_depths_list:
            combined_values = {}
            total_factor = 0

            for depth_dir, factor in explore_depths_list:
                import gridex
                matching_files = gridex.query_index(depth_dir, geometry)
                print(f"Matched files: {matching_files}")

                for file in matching_files:
                    x_coords, y_coords, pixel_values = extract_pixel_coords(os.path.join(depth_dir, file), geometry)

                    for i in range(len(x_coords)):
                        point_key = (x_coords[i], y_coords[i])
                        if point_key in combined_values:
                            combined_values[point_key] += pixel_values[i] * factor
                        else:
                            combined_values[point_key] = pixel_values[i] * factor
                
                total_factor += factor

            combined_df = pd.DataFrame(
                [(x, y, value / total_factor) for (x, y), value in combined_values.items()],
                columns=['x', 'y', layer]
            )
            output_df = pd.merge(output_df, combined_df, on=['x', 'y'], how='outer')
        else:
            print(f"No valid directories found for attribute '{layer}' within the specified depth range.")

    output_df = output_df.dropna().reset_index(drop=True)
    output_df = output_df.dropna().reset_index(drop=True)
    if len(attribute_list) <= 1:
        #if one attribute (this is done so choose_points algoirthm works, as it was not designed to do such)
        output_df[str(attribute_list[-1]) + '_dup'] = output_df.iloc[:, -1]
    if num_samples > 0 and len(output_df) >= num_samples:
        output_df.to_csv(output_name + '.csv', index=False)
        return output_df
    elif len(output_df) > 0:
        output_df.to_csv(output_name + '.csv', index=False)
        return output_df
    else:
        print("No data to output.")
        return pd.DataFrame()