import rasterio
import numpy as np
from rasterio.mask import mask
from shapely.wkt import loads as wkt_loads
import os
import pandas as pd
from pyproj import Proj, transform
from pyproj import Proj, Transformer
def extract_pixel_coords(input_tif_path, wkt, target_crs="EPSG:4326", nodata=-9999):
    # Check if the file exists before trying to open it
    if not os.path.exists(input_tif_path):
        print(f"File does not exist: {input_tif_path}")
        return None, None, None  # Return None to indicate the failure to process the file
    
    with rasterio.open(input_tif_path) as src:
        geometry = wkt_loads(wkt)
        
        src_crs = src.crs
        
        # Mask out the raster by the polygon
        masked_data, masked_transform = mask(src, [geometry], crop=True)
        masked_band = masked_data[0]
        
        # If there's no valid data, return None
        if np.all(masked_band == nodata):
            print(f"No valid data found for polygon in file: {input_tif_path}")
            return None, None, None
        
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
        
        # Transform coordinates if necessary
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



def output_from_attr(input_dir, wkt, depth_range, attribute_list=[], num_samples=0, output_name='output'):
    output_df = pd.DataFrame({'x': [], 'y': []})
    
    try:
        depth_min, depth_max = map(int, depth_range.split('-'))
    except ValueError:
        print('Invalid depth range. Correct format: "min-max" (e.g., "0-15").')
        return

    if not os.path.exists(input_dir):
        print(f"Input directory '{input_dir}' does not exist.")
        return
    
    os.chdir(input_dir)
    
    for attr in attribute_list:
        if not os.path.exists(attr):
            print(f"Directory for attribute '{attr}' does not exist.")
            continue
        
        os.chdir(attr)
        explore_depths_list = []
        
        for sub_dir in os.listdir():
            try:
                dir_depths = list(map(int, sub_dir.split('_')[:-1]))
                if len(dir_depths) == 2 and depth_min <= dir_depths[0] <= depth_max:
                    explore_depths_list.append((sub_dir, dir_depths[1] - dir_depths[0]))
            except ValueError:
                print(f"Skipping invalid directory '{sub_dir}'.")
                continue
        
        print(f"Found explore depths for '{attr}': {explore_depths_list}")

        if explore_depths_list:
            combined_values = {}
            total_factor = 0
            
            for depth_dir, factor in explore_depths_list:
                os.chdir(depth_dir)
                matching_files = find_matching_tif_files('_index.csv', wkt)
                
                if not matching_files:
                    print(f"No matching TIF files found in directory '{depth_dir}'. Skipping this directory.")
                    continue
                
                for file in matching_files:
                    if not os.path.exists(file):
                        print(f"File does not exist: {file}. Skipping.")
                        continue
                    
                    x_coords, y_coords, pixel_values = extract_pixel_coords(file, wkt)
                    
                    if x_coords is None or y_coords is None or pixel_values is None:
                        print(f"Skipping file '{file}' due to lack of overlap or valid data.")
                        continue
                    
                    for i in range(len(x_coords)):
                        point_key = (x_coords[i], y_coords[i])
                        if point_key in combined_values:
                            combined_values[point_key] += pixel_values[i] * factor
                        else:
                            combined_values[point_key] = pixel_values[i] * factor
                
                total_factor += factor
                os.chdir('..') 
            
            combined_df = pd.DataFrame(
                [(x, y, value / total_factor) for (x, y), value in combined_values.items()],
                columns=['x', 'y', attr]
            )
            output_df = pd.merge(output_df, combined_df, on=['x', 'y'], how='outer')
        else:
            print(f"No valid directories found for attribute '{attr}' within the specified depth range.")
        
        os.chdir('..') 
    os.chdir('..')
    output_df = output_df.dropna().reset_index(drop=True)
    if num_samples > 0 and len(output_df) >= num_samples:
        output_df.to_csv(output_name + '.csv', index=False)
        return output_df
    elif len(output_df) > 0:
        output_df.to_csv(output_name + '.csv', index=False)
        return output_df
    else:
        print("No data to output.")
        return pd.DataFrame()