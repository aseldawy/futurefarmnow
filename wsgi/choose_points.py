import numpy as np
import pandas as pd
import geopandas as gpd

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.covariance import EllipticEnvelope

from scipy.stats import trim_mean, trimboth, chi2
from scipy.spatial import KDTree
from scipy.spatial import distance, distance_matrix
from scipy.spatial.distance import mahalanobis

from shapely.geometry import Point
from pyDOE3 import ccdesign
from pykrige.ok import OrdinaryKriging
import pysal.lib
from esda.moran import Moran, Moran_BV_matrix
from libpysal.weights import DistanceBand, KNN

import random
from tqdm.auto import tqdm
from itertools import combinations
from functools import lru_cache
import itertools
from extract_points import *

def IQR_outliers(PCs, _threshold):
    """
    Filters out rows from the input array `PCs` that contain outliers.

    Parameters:
    - PCs: A 2D numpy array where each row represents a sample and each column represents a feature.
    - _threshold: A multiplier for the IQR to determine outlier bounds. Default is 1.5.

    Returns:
    - A boolean array where True indicates rows that are within the outlier bounds.
    """
    # Calculate Q1 and Q3 for each feature
    Q1, Q3 = np.percentile(PCs, [25, 75], axis=0)

    # Calculate the IQR for each feature
    IQR = Q3 - Q1

    # Determine lower and upper bounds for outliers
    lower_bounds = (Q1 - _threshold * IQR).reshape(1, -1)
    upper_bounds = (Q3 + _threshold * IQR).reshape(1, -1)

    # Determine which rows are within the bounds
    within_bounds = (PCs >= lower_bounds) & (PCs <= upper_bounds)

    # Return a boolean array where True means the row is within bounds
    return np.all(within_bounds, axis=1)


def mahalanobis_outliers(PCs, confidence_level):
    """
    Detects outliers in the input array `PCs` using the Mahalanobis distance.

    Parameters:
    - PCs: A 2D numpy array where each row represents a sample and each column represents a feature.
    - confidence_level: The confidence level for the chi-squared distribution to determine the outlier threshold.

    Returns:
    - A boolean array where True indicates rows that are considered outliers.
    """
    # Calculate the degrees of freedom (number of features)
    df = PCs.shape[1]

    # Calculate the threshold based on the chi-squared distribution
    threshold_pca = chi2.ppf(confidence_level, df=df)

    # Calculate the covariance matrix and its inverse
    cov_matrix = np.cov(PCs, rowvar=False)
    inv_cov_matrix = np.linalg.inv(cov_matrix)

    # Calculate the mean of the data
    mean_PCs = np.mean(PCs, axis=0)

    # Calculate Mahalanobis distances for all samples in one step
    # m_distances = [mahalanobis(x, mean_PCs, inv_cov_matrix) for x in PCs]
    # faster way by vectorizing
    centered_PCs = PCs - mean_PCs
    m_distances = np.sqrt(np.sum(centered_PCs @ inv_cov_matrix * centered_PCs, axis=1))

    # Return a boolean array where True means the Mahalanobis distance exceeds the threshold
    return m_distances <= threshold_pca

def elliptic_envelope_outliers(PCs, contamination_rate):
    """
    Detects outliers in the input array `PCs` using the Elliptic Envelope method.

    Parameters:
    - PCs: A 2D numpy array where each row represents a sample and each column represents a feature.
    - contamination_rate: The proportion of outliers in the data. Default is 0.1.

    Returns:
    - A boolean array where True indicates rows that are not considered outliers.
    """
    # Initialize the EllipticEnvelope model with the specified contamination rate
    elliptic_envelope_pca = EllipticEnvelope(contamination=contamination_rate)

    # Fit the model to the data and predict outliers
    y_pred_pca = elliptic_envelope_pca.fit_predict(PCs)

    # Return a boolean array where True means the row is not an outlier
    return y_pred_pca != -1

def generate_design(data, n_samples, whitten=0):
    ccd = ccdesign(2, center=(1,1), alpha='o', face='cci')
    scaled_ccd = np.zeros_like(ccd) # Initialize scaled CCD with the correct shape
    ccd_min, ccd_max = ccd.min(axis=0), ccd.max(axis=0) # Compute scaling parameters
    data_min, data_max = np.percentile(data, [whitten, 100-whitten], axis=0)
    scaled_ccd = (ccd - ccd_min) / (ccd_max - ccd_min) * (data_max - data_min) + data_min # Scale design
    # Extract subsets
    ccd_boxes = scaled_ccd[:4]
    ccd_star = scaled_ccd[5:9]
    support_points = lambda x: np.repeat(scaled_ccd[np.newaxis, 4, :], x, axis=0) # Support points generator
    # Define the designs list with optimized operations
    designs = [
        np.vstack([ccd_boxes, support_points(1)]),
        np.vstack([ccd_star, support_points(1)]),
    ]
    # Additional designs based on manipulation of existing ones
    designs += [designs[1][:2] / 2, designs[1][2:] / 2, designs[0] / 2]
    # Determine the design closest to the target number of samples
    al_list = np.cumsum([len(d) for d in designs])
    k = np.searchsorted(al_list, n_samples)
    # Concatenate the selected designs up to the closest match
    return np.vstack(designs[:k+1]), al_list.tolist()

def iter_combinations(num_combs=np.nan, filtered_distances = None, filtered_indices = None):

    dists = filtered_distances
    idxs = filtered_indices

    total_combs = np.prod([len(row) if len(row) else 1 for row in dists])
    print(f"Total possible combinations are {total_combs}")
    num_combs = int(np.nanmin([num_combs, total_combs]))
    print(f"iterating through {num_combs} of them")


    # Generate unique combinations efficiently
    combinations = set()

    if num_combs < 4600000:
        # Generate all possible combinations systematically
        all_combinations = itertools.product(*[itertools.product(row_dist, row_idx) for row_dist, row_idx in zip(dists, idxs)])
        for comb in tqdm(itertools.islice(all_combinations, num_combs)):
            curr_comb_dist, curr_comb_idx = zip(*comb)
            combinations.add((tuple(curr_comb_dist), tuple(curr_comb_idx)))
        return combinations

    for _ in tqdm(range(num_combs)):
        curr_comb_dist = []
        curr_comb_idx = []
        for row_dist, row_idx in zip(dists, idxs):
            if len(row_dist) > 0:
                chosen = random.choice(list(zip(row_dist, row_idx)))
                curr_comb_dist.append(chosen[0])
                curr_comb_idx.append(chosen[1])
            else:
                curr_comb_dist.append(None)
                curr_comb_idx.append(None)
        combinations.add((tuple(curr_comb_dist), tuple(curr_comb_idx)))

    return combinations

#outlier technique is important, so is scaler, we need a way to analyze scatter plot distribution so the scaler and outlier don't need to be chosen by the user
#scalar_scheme can be: StandardScaler, RobustScaler, PowerTransformer
#outlier tecnhique can be: IQR Thresholding, Mahalanobis Distance, Elliptic Envelope
def select_points(df, num_samples=10, epsg_code = 32618, scalar_scheme = 'StandardScaler', outlier_technique = 'IQR Thresholding',weight = 0.5, Morgans = False, output_name = 'results'):
    Sample_IDx_FID = list(range(len(df)))
    lat = df.columns[0]
    lon = df.columns[1]
    
    chosen_cols = []
    
    for col in df.columns:
        if col not in[lat,lon]:
            chosen_cols.append(True)            
            
    mask = [i for i in chosen_cols]
    
    selected_df = df.loc[:, df.columns.drop(lat).drop(lon)[mask]]
    
    geometry = [Point(xy) for xy in df.loc[:,[lat,lon]].values]
    gdf = gpd.GeoDataFrame(selected_df, geometry=geometry)
    gdf.crs = f"EPSG:{epsg_code}"

    scaler = {
    'RobustScaler'     : RobustScaler(),
    'StandardScaler'   : StandardScaler(),
    'PowerTransformer' : PowerTransformer()
    }[scalar_scheme]
    
    X_scaled = scaler.fit_transform(selected_df.values)
    pca = PCA(n_components=X_scaled.shape[1])
    X_pca = pca.fit_transform(X_scaled)
    
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    cdf = pd.DataFrame(np.multiply(cumulative_variance, 100).round(3).reshape(1,-1).tolist(), columns=list(range(1,len(cumulative_variance)+1)), index=["% cumulative variance captured :"])

    pca_selection = 2
    
    whitten = 5
    threshold = 0.5

    pca = PCA(n_components=pca_selection)
    PCs = pca.fit_transform(X_scaled)
    
    rows_to_keep = np.ones(PCs.shape[0], dtype=bool)
    filtered_Pcs = PCs[:,:]
    preferedPCs = None
    filtered_distances = None
    filtered_indices = None
    Geo_space_XY = None
    Var_space_XY = None
    epsilion = 1e-7# change this
    NNearest_neighbour = 3 # change this

    allowed_samples = [5, 10, 12, 15, 20]# change this
    design = generate_design(filtered_Pcs, num_samples, whitten)

    if outlier_technique == 'IQR Thresholding':
        rows_to_keep = IQR_outliers(PCs, threshold)
    elif outlier_technique == 'Mahalanobis Distance':
        if threshold > 1:
            threshold = 1
        rows_to_keep = mahalanobis_outliers(PCs, threshold)
    elif outlier_technique == 'Elliptic Envelope':
        if threshold > 0.5:
            threshold = 0.5
        rows_to_keep = elliptic_envelope_outliers(PCs, threshold)
    filtered_Pcs = PCs[rows_to_keep] # Convert filtered_data list to a NumPy array
    outliers_Pcs = PCs[~rows_to_keep] # Convert filtered_data list to a NumPy array
        
    design, _ = generate_design(filtered_Pcs, num_samples, whitten)

    Geo_space_X = df.loc[rows_to_keep, lat]
    Geo_space_Y = df.loc[rows_to_keep, lon]
    Geo_space_XY = np.array([Geo_space_X, Geo_space_Y]).T
    Var_space_XY = filtered_Pcs
    max_dist = np.max(distance_matrix(Geo_space_XY,
                           Geo_space_XY))
    geo_max = max_dist
    geo_min = 0 + epsilion
    var_max = .25 # change this
    var_min = 0 + epsilion
    
    prefered_lon = None
    prefered_lat = None
    if prefered_lon:
        preferedPCs = []
        kriging_models = [OrdinaryKriging(Geo_space_Y, Geo_space_X, Var_space_XY[:, i], variogram_model='spherical') for i in range(PCs.shape[1])]
        for kriging_model in kriging_models:
            predicted_value, predicted_std = kriging_model.execute('points', prefered_lon, prefered_lat)
            preferedPCs.append(predicted_value.data)
        preferedPCs = np.vstack(preferedPCs).T

    tree = KDTree(Var_space_XY)
    distances, indices = tree.query(design, k=NNearest_neighbour)
    ind_ko = np.unique(indices)
    valid_indices = distances < var_max
    filtered_distances = [distances[i][valid_i] for i, valid_i in enumerate(valid_indices)]
    filtered_indices = [indices[i][valid_i] for i, valid_i in enumerate(valid_indices)]
    ind_ko = np.unique(indices[valid_indices])
    avg = np.average(distances, axis=1)
    
    print(filtered_indices)
    
    if (np.min(distances, axis=1) > var_max).any():
        print("\x1b[31mError: scaled design does not fit in varibale scale, please refer to graph to fit the design properly \x1b[0m")
    elif (avg > var_max).any():
        print('''\x1b[33mWarning: scaled design is not a good fit in varibale scale,
                 kindly readjust the thresholds and try again for better fit
                 or try using a different scaler to change the distribution \x1b[0m''')
    # finding unique ind across all design points
    assigned_to = {}  # Tracks which design point an index is assigned to
    point_counts = np.zeros(len(design), dtype=int)  # Tracks how many points are assigned to each design point
    for i in range(len(design)):
        for j in range(NNearest_neighbour):
            if not valid_indices[i][j]:
                break
            idx = indices[i][j]
            dist = distances[i][j]
            # Check if the index is already assigned
            if idx in assigned_to:
                # Retrieve previously assigned design point and distance
                prev_i, prev_dist = assigned_to[idx]
                if dist < prev_dist or (dist == prev_dist and point_counts[i] < point_counts[prev_i]):
                    # Update assignment if current design point is closer or equally close but has fewer points
                    point_counts[prev_i] -= 1
                    point_counts[i] += 1
                    assigned_to[idx] = (i, dist)
            else:
                # Assign index to the current design point
                assigned_to[idx] = (i, dist)
                point_counts[i] += 1
    # Construct the final list of assigned indices for each design point
    assigned_indices = [[] for _ in range(len(design))]
    for idx, (i, _) in assigned_to.items():
        assigned_indices[i].append(idx)
    # Optionally, convert each list of indices to a NumPy array
    assigned_indices = [np.array(lst) for lst in assigned_indices]
    filtered_indices = assigned_indices
    
    mat_dist = distance_matrix(Geo_space_XY,Geo_space_XY)

    np.fill_diagonal(mat_dist, np.nan)
    geo_max = np.nanmax(mat_dist)
    geo_min = np.nanmin(mat_dist) + epsilion
    var_max = .4 # change this
    var_min = 0 + epsilion

    scale_geo = lambda x: (x - geo_min)/(geo_max - geo_min)*3 #scale between 0 ~ 3
    scale_var = lambda x: (x - var_min)/(var_max - var_min)*3 #scale between 0 ~ 3

    filtered_distances = [
        distance.cdist([design[i]], Var_space_XY[e], metric='euclidean').flatten()
        if len(e) > 0 else np.array([])  # Return an empty array for empty subsets
        for i, e in enumerate(filtered_indices)
    ]

    # now check which point belong to which group or if it belongs to a group at all?? must use? create a symmetric point in XY plane? more than 2 axis???
    if prefered_lat:
        x = distance_matrix(design, preferedPCs)
        prefered_mask = (x < var_max).any(axis=0)
        _prefered_lat_lon = np.array([prefered_lat, prefered_lon]).T
        Var_space_XY = np.vstack([Var_space_XY, preferedPCs[prefered_mask]])
        Geo_space_XY = np.vstack([Geo_space_XY, _prefered_lat_lon[prefered_mask]])

        kept_prefered_points = x[x < var_max]
        c = 0
        for i,e in tqdm(enumerate((x < var_max).any(axis=1))):
            if e:
                filtered_indices[i] = np.array([len(Var_space_XY)-len(kept_prefered_points)+c])
                filtered_distances[i] =  np.array([kept_prefered_points[c]])
                c+=1
    arr = np.array(list(map(lambda x: x[0] + x[1], iter_combinations(num_combs=4600000, filtered_distances= filtered_distances, filtered_indices= filtered_indices))))
    W = weight
    final_score = float('-inf')
    threshold = geo_min
    final_result = None
    features = Var_space_XY.shape[1]
    for i in tqdm(arr):
        idx = i[design.shape[0]:].astype(int)
        dist_matrix = distance_matrix(Geo_space_XY[idx],
                                    Geo_space_XY[idx])
        np.fill_diagonal(dist_matrix, np.inf)
        dgmin = np.nanmin(dist_matrix)
        dvmax = i[:design.shape[0]].max()

        dgmin = scale_geo(dgmin)
        dvmax = scale_var(dvmax)

        score = (- W * scale_var(dvmax)) + (1-W) * scale_geo(dgmin)
        if score > final_score:
            final_score = score
            final_result = idx
    #Morgans:
    if Morgans:
        # @title Consideration of Moran's I (optional)
        # @markdown running this cell will negate the use of above algorithm and run Moran'I optimization critera instead
        arr = np.array(list(map(lambda x: x[0] + x[1], iter_combinations(num_combs=4600000))))
        W1 = 0.4  # Example weight for dgmin
        W2 = 0.3  # Example weight for dvmax
        W3 = 0.3  # Example weight for Moran's I
        final_score = float('-inf')
        final_result = None
        features = Var_space_XY.shape[1]

        for i in tqdm(arr):
            idx = i[design.shape[0]:].astype(int)
            dist_matrix = distance_matrix(Geo_space_XY[idx], Geo_space_XY[idx])
            np.fill_diagonal(dist_matrix, np.inf)
            dgmin = np.nanmin(dist_matrix)
            dvmax = i[:design.shape[0]].max()

            dgmin = scale_geo(dgmin)
            dvmax = scale_var(dvmax)

            # Calculate Moran's I for spatial spread
            weights = DistanceBand.from_array(Geo_space_XY[idx], threshold=20, binary=True, silence_warnings=True)

            # Ensure weights are valid
            if np.any(np.isnan(weights.sparse.toarray())) or np.any(np.isinf(weights.sparse.toarray())):
                continue  # Skip this iteration if weights are invalid

            mi_values = []
            for j in range(features):
                values = Var_space_XY[idx, j]
                if not np.any(np.isnan(values)) and not np.any(np.isinf(values)):
                    mi = Moran(values, weights)
                    if not np.isnan(mi.I) and not np.isinf(mi.I):
                        mi_values.append(mi.I)

            if mi_values:
                mi = np.max(mi_values)
            else:
                mi = np.inf  # Assign a high value to mi if no valid Moran's I could be calculated

            # Calculate the score
            score = (W1 * dgmin) + (W2 / dvmax) + (W3 / mi)

            if score > final_score:
                final_score = score
                final_result = idx
    ndf = pd.DataFrame(Geo_space_XY[final_result].reshape(-1, 2), columns=[lat, lon])
    ndf.to_csv(f"{output_name}.csv", index=None)
    return ndf

