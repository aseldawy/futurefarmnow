#!/usr/bin/env python3

"""
File: server.py

Description:
This script serves as the main entry point for a Flask-based web server. It is meant to be used for development mode.
It combines two modular Flask applications (blueprints) - `soil_stats` and `soil_sample` - into a single unified server.
Each module provides APIs for different soil data operations:

1. **Soil Stats Module (`soil_stats`)**:
   - Provides APIs to process GeoTIFF files and calculate pixel statistics within a given polygon.
   - Handles operations like raster clipping, weighted calculations, and returning statistical results.

2. **Soil Sample Module (`soil_sample`)**:
   - Provides APIs to sample soil-related data based on user-defined parameters and spatial queries.

Features:
- Combines and registers the blueprints (`soil_stats_bp`, `soil_sample_bp`) for modular development.
- Hosts static files for development purposes under the `/public_html` directory.

Usage:
- Run this script in development mode using the command:
  ```
  python server.py
  ```
- Access the APIs:
  - `soil_stats` endpoints (e.g., `/soil/singlepolygon.json`).
      ```shell
      curl -X GET "http://127.0.0.1:5000/soil/singlepolygon.json?soildepth=0-5&layer=alpha" \
           -H "Content-Type: application/geo+json" \
           -d '{"type":"MultiPolygon","coordinates":[[[[-117.09285814562507255,33.82248877684289567],[-117.09311988685111316,33.82246974122563898],[-117.09331976020517629,33.82251733063394994],[-117.09355294847989626,33.82266485703437553],[-117.09538513346878119,33.82400211388483768],[-117.09564211542040368,33.82423530113598531],[-117.09599903494739692,33.82481588961324093],[-117.09783597921065734,33.82807099268816842],[-117.09792163986121238,33.8282327962675069],[-117.09794067516207861,33.82843742987725477],[-117.09788356836114076,33.82861826850821529],[-117.09774080090969051,33.82882290268781844],[-117.0961322847484638,33.82924644646776358],[-117.09398601100701853,33.82969378533054083],[-117.09304374385099834,33.82983179392608264],[-117.09270110214713156,33.82988890181807307],[-117.09183021871685071,33.83022678523190052],[-117.08819916096476277,33.83170205102513961],[-117.08812777723903764,33.83168301447423687],[-117.08798976906194866,33.83161639136280741],[-117.08712364400770412,33.83004118685605022],[-117.08695232270660824,33.82988414239322594],[-117.08675244755589517,33.82970806288982146],[-117.08648118867942856,33.82922741160864177],[-117.08644311807771032,33.82898470633658405],[-117.08640980585201419,33.82877531380393776],[-117.08633842122794988,33.82863730499543919],[-117.08410648683597799,33.82495865906553689],[-117.08412076376079369,33.8248920329527607],[-117.08423021898659044,33.82482540828065964],[-117.08734255445929762,33.82398783761156125],[-117.09285814562507255,33.82248877684289567]]]]}'
      ```
  - `soil_sample` endpoints (e.g., `/soil/sample.json`).
      ```shell
      curl -X POST "http://127.0.0.1:5000/soil/sample.json?soildepth=0-5&layer=alpha&layer=clay&num_points=7" \
           -H "Content-Type: application/geo+json" \
           -d '{"type":"MultiPolygon","coordinates":[[[[-117.09285814562507255,33.82248877684289567],[-117.09311988685111316,33.82246974122563898],[-117.09331976020517629,33.82251733063394994],[-117.09355294847989626,33.82266485703437553],[-117.09538513346878119,33.82400211388483768],[-117.09564211542040368,33.82423530113598531],[-117.09599903494739692,33.82481588961324093],[-117.09783597921065734,33.82807099268816842],[-117.09792163986121238,33.8282327962675069],[-117.09794067516207861,33.82843742987725477],[-117.09788356836114076,33.82861826850821529],[-117.09774080090969051,33.82882290268781844],[-117.0961322847484638,33.82924644646776358],[-117.09398601100701853,33.82969378533054083],[-117.09304374385099834,33.82983179392608264],[-117.09270110214713156,33.82988890181807307],[-117.09183021871685071,33.83022678523190052],[-117.08819916096476277,33.83170205102513961],[-117.08812777723903764,33.83168301447423687],[-117.08798976906194866,33.83161639136280741],[-117.08712364400770412,33.83004118685605022],[-117.08695232270660824,33.82988414239322594],[-117.08675244755589517,33.82970806288982146],[-117.08648118867942856,33.82922741160864177],[-117.08644311807771032,33.82898470633658405],[-117.08640980585201419,33.82877531380393776],[-117.08633842122794988,33.82863730499543919],[-117.08410648683597799,33.82495865906553689],[-117.08412076376079369,33.8248920329527607],[-117.08423021898659044,33.82482540828065964],[-117.08734255445929762,33.82398783761156125],[-117.09285814562507255,33.82248877684289567]]]]}'
      ```

- For static file hosting, place files in the `../public_html` directory relative to this script.
"""

from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS
from soil_stats import soil_stats_bp
from soil_sample import soil_sample_bp
from ndvi_timeseries import ndvi_timeseries_bp
from app.et_map import etrawdata_bp
from app.ai_assistant import ai_assistant_bp

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"], "allow_headers": "*"}})

# Register blueprints
app.register_blueprint(soil_stats_bp)
app.register_blueprint(soil_sample_bp)
app.register_blueprint(ndvi_timeseries_bp)
app.register_blueprint(etrawdata_bp)
app.register_blueprint(ai_assistant_bp)

# Global error handler
@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    import sys
    print(traceback.format_exc(), file=sys.stderr)
    return jsonify({
        "error": "An unexpected error occurred",
        "details": str(e),
        "stack_trace": traceback.format_exc().split("\n")
    }), 500

@app.route('/vectors.json', methods=['GET'])
def list_vectors():
    return send_from_directory("../data", "vectors.json")

if app.debug:
    # Serve static files only in development mode
    @app.route('/public_html/<path:filename>')
    def serve_static(filename):
        static_folder = '../public_html'
        return send_from_directory(static_folder, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5200, debug=True)