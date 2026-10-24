#!/usr/bin/env python3
"""
File: soil_sample.py

Description:
This module provides a Flask blueprint for handling requests related to soil data sampling. It includes endpoints
for processing spatial data and calculating statistics based on user-defined parameters. The module is designed
to work as part of a larger Flask application.

Features:
- Accepts GeoJSON geometry and query parameters (`soildepth`, `layer`, `num_points`) via HTTP POST or GET requests.
- Processes soil sample data using spatial queries and calculates statistics for the selected layers.
- Supports modular integration through a Flask blueprint (`soil_sample_bp`).

Endpoints:
- `/soil/sample.json`: Main endpoint for processing soil sample data.
  - **Methods**: POST, GET
  - **Parameters**:
    - `soildepth`: Depth range (e.g., `0-15`).
    - `layer`: List of layers to process (e.g., `alpha`, `clay`).
    - `num_points`: Number of sample points to calculate statistics for.
  - **Body**:
    - GeoJSON geometry specifying the area of interest.
  - **Response**:
    - JSON object containing statistics for the selected layers.

Functions:
- `calculate_statistics(sample, original_df)`: Calculates various statistics for selected layers.
- `process_request(query_params, query_geometry)`: Processes the query parameters and geometry to generate results.
- `soil_sample()`: Flask route handler for the `/soil/sample.json` endpoint.

Dependencies:
- Flask: Used to handle HTTP requests and responses.
- Shapely: For geometry parsing and manipulation.
- Pandas: For statistical calculations on the sample data.
- Custom Modules:
  - `extract_points`: Utility functions for extracting spatial data.
  - `choose_points`: Utility functions for selecting sample points.
  - `soil`: Provides base directory and supported layers.

Usage:
- Import and register the blueprint in the main Flask app:
  ```python
  from soil_sample import soil_sample_bp
  app.register_blueprint(soil_sample_bp)
"""

import json
import pandas as pd
import os
import sys
import tempfile
from io import StringIO
from flask import Blueprint, request, jsonify, make_response
from shapely.geometry import shape, mapping
from extract_points import *
from choose_points import *
from conf import SOIL_DATA_DIR, SOIL_LAYERS

soil_sample_bp = Blueprint("soil_sample", __name__)

def calculate_statistics(sample, original_df):
    statistics = {}
    layers = original_df.columns[2:]
    sample_df = pd.DataFrame()
    for i, row in sample.iterrows():
        x_value = row['x']
        y_value = row['y']
        matching_rows = original_df[(original_df['x'] == x_value) & (original_df['y'] == y_value)]
        if not sample_df.empty:
            sample_df = pd.concat([sample_df, matching_rows], ignore_index=True)
        else:
            sample_df = matching_rows
    if sample_df.empty:
        return "No matching data found for the sample"
    elif len(sample_df) != len(sample):
        return "Duplicate Points"

    def to_python(value):
        # Convert NumPy types to Python types
        if isinstance(value, (pd.Series, pd.Index)):
            return value.to_list()  # Handles pandas series/index objects
        elif hasattr(value, "item"):
            return value.item()  # Handles numpy scalar types
        else:
            return value  # Return as is for standard Python types

    for layer in layers:
        layer_data = original_df[layer]
        sample_data = sample_df[layer]

        layer_stats = {
            "actual": {
                "max": to_python(layer_data.max()),
                "min": to_python(layer_data.min()),
                "sum": to_python(layer_data.sum()),
                "median": to_python(layer_data.median()),
                "stddev": to_python(layer_data.std()),
                "count": to_python(layer_data.count()),
                "mean": to_python(layer_data.mean()),
                "lowerquart": to_python(layer_data.quantile(0.25)),
                "upperquart": to_python(layer_data.quantile(0.75)),
            },
            "sample": {
                "max": to_python(sample_data.max()),
                "min": to_python(sample_data.min()),
                "sum": to_python(sample_data.sum()),
                "median": to_python(sample_data.median()),
                "stddev": to_python(sample_data.std()),
                "count": to_python(sample_data.count()),
                "mean": to_python(sample_data.mean()),
                "lowerquart": to_python(sample_data.quantile(0.25)),
                "upperquart": to_python(sample_data.quantile(0.75)),
            }
        }
        statistics[layer] = layer_stats

    return statistics

def process_request(query_params, query_geometry):
    soil_depth = query_params.get("soildepth")
    layers = query_params.getlist("layer")
    num_points = int(query_params.get("num_points"))
    print(f"[SAMPLE_DEBUG] process_request: soildepth={soil_depth}, layers={layers}, num_points={num_points}", file=sys.stderr)

    # Calculate layer values at each point
    df = output_from_attr(
        input_dir=SOIL_DATA_DIR,
        geometry=query_geometry,
        depth_range=soil_depth,
        attribute_list=layers,
        num_samples=num_points
    )
    print(f"[SAMPLE_DEBUG] After output_from_attr: df shape={df.shape}, columns={list(df.columns)}", file=sys.stderr)

    if df.empty:
        response = make_response(jsonify({
            "query": mapping(query_geometry),
            "message": "No soil data found for the given geometry, depth, and layers.",
            "hint": "The polygon may be too small to contain any soil raster pixel centers. Try a larger polygon.",
            "results": [],
            "statistics": {"layers": {}}
        }), 404)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response

    # The point-selection algorithm needs at least the requested number of
    # distinct raster cells. Small polygons can overlap fewer cells than the
    # requested sample size, which otherwise causes failures in PCA/KDTree.
    if len(df) < num_points:
        response = make_response(jsonify({
            "query": mapping(query_geometry),
            "error": "The polygon contains too few soil data cells for the requested sample size.",
            "requested_points": num_points,
            "available_points": len(df),
            "hint": "Select a larger polygon or request fewer sample points."
        }), 422)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response

    # Choose what points to use
    sample_df = select_points(df, num_samples=num_points, epsg_code=4326)
    print(f"[SAMPLE_DEBUG] After select_points: sample_df shape={sample_df.shape}, requested={num_points}, actual={len(sample_df)}", file=sys.stderr)

    # Calculate statistics for the layers
    statistics = calculate_statistics(sample_df, df)

    response_data = {
        "query": mapping(query_geometry),
        "results": [{"x": row['x'], "y": row['y'], "id": index} for index, row in sample_df.iterrows()],
        "statistics": {
            "layers": statistics
        }
    }
    print(f"[SAMPLE_DEBUG] Response: Returning {len(response_data['results'])} sample points", file=sys.stderr)

    response = make_response(jsonify(response_data))
    # Add CORS headers
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# Define the main endpoint
@soil_sample_bp.route('/soil/sample.json', methods=['POST', 'GET', 'OPTIONS'])
def soil_sample():
    # Handle preflight OPTIONS request for CORS
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response
    
    # Extract query parameters from the URL
    query_params = request.args  # Automatically handles QUERY_STRING
    try:
        # Read and parse GeoJSON geometry from the payload
        print("Request data:", str(request.data), file=sys.stderr)
        query_geometry = shape(request.get_json())

        # Simulate process_request function (implement your logic here)
        response_json = process_request(query_params, query_geometry)

        # Send the response
        return response_json

    except (ValueError, json.JSONDecodeError) as e:
        import traceback
        traceback.print_exc()
        response = make_response(jsonify({
            "error": "Invalid JSON payload.",
            "details": str(e),
            "stack_trace": traceback.format_exc()
        }), 400)
        # Add CORS headers to error response
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        response = make_response(jsonify({"error": str(e)}), 500)
        # Add CORS headers to error response
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response
