from ultralytics import YOLO
import streamlit as st
import cv2
#import pafy
import folium
import settings
import numpy as np
import pandas as pd
import laspy
import plotly.express as px
import laspy
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import rvt.default
import rvt.blend
import rasterio
from rasterio.transform import from_origin
import geopandas as gpd
import os
import pyproj
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

#dictionnaries for point cloud visualization
category_name_dict = {'1' : 'unclassified',
                        '2' : 'ground',
                        '3' : 'low vegetation',
                        '4' : 'medium vegetation',
                        '5' :  'high vegetation',
                        '6' : 'building',
                        '7' : 'low noise',
                        '8' : 'model key',
                        '15' : 'transmission tower',
                        '18' : 'high noise'}
color_dict = {'unclassified': '#000000',
                'ground': '#D29B5A',
                'low vegetation': '#ACE628',
                'medium vegetation': '#70E628',
                'high vegetation': '#468420',
                'building': '#FF0000',
                'low noise': '#FF00DC',
                'model key': '#0036FF',
                'transmission tower': '#00B9FF',
                'high noise': '#6C00FF'}


def create_df_decimated(point_cloud, factor = 15):
    lidar_data = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z, point_cloud.classification)).transpose()
    df = pd.DataFrame(lidar_data, columns = ['x','y','z','category'])
    df['category']=df['category'].astype(int).astype(str)
    df['category']=[category_name_dict[i] for i in df['category']]
    df['color']= [color_dict[i] for i in df['category']]
    df_decimated = df.iloc[::factor]
    return df_decimated


#@st.cache_data
def compute_slope(df_elevation, width, height, sampling):
    dem_arr=df_elevation['z'].to_numpy().reshape(width,height)
    default = rvt.default.DefaultValues()
    default.slp_output_units = "degree"
    slope_arr = default.get_slope(dem_arr=dem_arr, resolution_x=sampling, resolution_y=sampling)
#   plt.imshow(slope_arr, cmap='gray_r')
    slope_arr = np.clip(slope_arr/(55),0,1)
    #slope_arr = (slope_arr-np.amin(slope_arr))/(np.amax(slope_arr)-np.amin(slope_arr))
#   print(f'Values of slope are between {slope_arr.min()} and {slope_arr.max()}')
    return slope_arr

#@st.cache_data
def compute_svf_opns(df_elevation, width, height,sampling):
    dem_arr=df_elevation['z'].to_numpy().reshape(width,height)
    default = rvt.default.DefaultValues()
    svf_n_dir = 16  # number of directions
    svf_r_max = int(5/sampling) # max search radius in pixels
    svf_noise = 0  # level of noise remove (0-don't remove, 1-low, 2-med, 3-high)
    svf_opns_dict = default.get_sky_view_factor(dem_arr=dem_arr, resolution=sampling,
                                                    compute_svf=True, compute_asvf=False, compute_opns=True)

    svf_arr = svf_opns_dict["svf"]
    #svf_arr = np.clip((svf_arr-0.65)/0.35,0,1)
    svf_arr = (svf_arr-np.amin(svf_arr))/(np.amax(svf_arr)-np.amin(svf_arr))
    #plt.imshow(svf_arr, cmap='gray')
    #print(f'Skyview factor values are between {svf_arr.min()} and {svf_arr.max()}')
    opns_arr = svf_opns_dict["opns"]
    opns_arr = np.clip((opns_arr-60)/35,0,1)
    #opns_arr = (opns_arr-np.amin(opns_arr))/(np.amax(opns_arr)-np.amin(opns_arr))
    #plt.imshow(opns_arr, cmap='gray')
    #print(f'Positive openness are between {opns_arr.min()} and {opns_arr.max()}')
    return svf_arr, opns_arr

#@st.cache_data
def display_VAT_HS(df_elevation,width,height, sampling):
    slope_arr = 1 - compute_slope(df_elevation,width,height, sampling)
    svf_arr, opns_arr = compute_svf_opns(df_elevation,width,height, sampling)
    VAT_HS = np.dstack((slope_arr,opns_arr, svf_arr))
    return VAT_HS

#@st.cache_data
def write_VAT_HS_tiff(df_elevation,width,height, sampling):
    slope_arr = 1 - compute_slope(df_elevation,width,height)
    svf_arr, opns_arr = compute_svf_opns(df_elevation,width,height)
    VAT_HS = np.dstack((slope_arr,opns_arr, svf_arr))
    return VAT_HS

#@st.cache_data
def create_tile_tiff(df_decimated):
    #Create df
    df_elevation_input = df_decimated[df_decimated['category']=='ground']
    df_elevation_input = df_elevation_input.drop(columns=['category', 'color'])
    #print(df_elevation_input.shape)
    #Create elevation input
    proj = "EPSG:32615"
    # Get X and Y coordinates of rainfall points
    x_coordinates_input = df_elevation_input.x
    y_coordinates_input = df_elevation_input.y
    #Create list of XY coordinate pairs
    coords_input = [list(xy) for xy in zip(x_coordinates_input, y_coordinates_input)]
    elevation_input = list(df_elevation_input.z)
    # Split data into testing and training sets
    coords_train, coords_test, elevation_train, elevation_test = train_test_split(coords_input, elevation_input, test_size = 0.20)
    # Set number of neighbors to look for
    neighbors = 100
    # Initialize KNN regressor
    knn_regressor = KNeighborsRegressor(n_neighbors = neighbors, weights = "distance")
    # Fit regressor to data
    knn_regressor.fit(coords_train, elevation_train)
    # Defining regular grid
    SAMPLING_IN_M = 1
    X_MIN, X_MAX = min(df_elevation_input.x), max(df_elevation_input.x)#
    # print('X range in m: {} - {}'.format(round(X_MIN, 0),round(X_MAX, 0)))
    Y_MIN, Y_MAX = min(df_elevation_input.y), max(df_elevation_input.y)
    # print('Y range in m: {} - {}'.format(round(Y_MIN, 0),round(Y_MAX, 0)))
    x_coordinates_output = np.arange(X_MIN,X_MAX+SAMPLING_IN_M,SAMPLING_IN_M)
    y_coordinates_output = np.arange(Y_MIN,Y_MAX+SAMPLING_IN_M,SAMPLING_IN_M)
    width = len(x_coordinates_output)
    # print('WIDTH :', width)
    height = len(y_coordinates_output)
    # print('HEIGHT :', height)
    # print('Output image dimensions: Width: {} samples - Height: {} samples '.format(width,height))
    # print('X,Y coordinates dimensions : x: {}, y: {} '.format(len(x_coordinates_output),len(y_coordinates_output)))
    coords_output = []
    for x in x_coordinates_output:
        for y in y_coordinates_output:
            coords_output.append([x,y])
    df_elevation_output = pd.DataFrame(coords_output, columns =['x', 'y'])
    # print('Elevation output x,y size', df_elevation_output.shape)
    df_elevation_output['z'] = knn_regressor.predict(coords_output)
    # print('Prediction output z size', len(df_elevation_output['z']))
    #compute layers
    slope_arr = 1 - compute_slope(df_elevation_output, width, height, SAMPLING_IN_M)
    svf_arr, opns_arr = compute_svf_opns(df_elevation_output, width, height, SAMPLING_IN_M)
    rgb_image = np.dstack((slope_arr, opns_arr, svf_arr))*255
    rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)
    rgb_image = Image.fromarray(rgb_image, 'RGB')
    rgb_image.save('images/rgb_tile.tiff', format='TIFF')
    with rasterio.open('images/rgb_tile.tiff') as src:
      tiff_image = src.read()
    transform = from_origin(X_MIN, Y_MIN, SAMPLING_IN_M, -SAMPLING_IN_M)
    crs = 'EPSG:32615'
    with rasterio.open('images/rgb_tile_georef.tiff', 'w', driver='GTiff', height=tiff_image.shape[1], width=tiff_image.shape[2], count=3, dtype=tiff_image.dtype, crs=crs, transform=transform) as dst:
      dst.write(tiff_image)

    return
    # plt.imshow(rgb_image)
    # plt.axis('off')
    # plt.show()


#@st.cache_data
def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model

#@st.cache_data
def read_contour_data(filename):
    contour_data = []
    with open(filename, 'r') as file:
        for line in file:
            #print(line)
            line = line.strip().split(' ')[1:]  # Skip the category number
            #print(line)
            latitudes = [float(lat) for lat in line[::2]]  # Extract every other value as latitudes
            longitudes = [float(lon) for lon in line[1::2]]  # Extract every other value as longitudes
            contour_data.extend([list(coord) for coord in zip(latitudes, longitudes)])
    return contour_data

def read_contour_data_no_cat(filename):
    contour_data = []
    with open(filename, 'r') as file:
        for line in file:
            #print(line)
            line = line.strip().split(' ')[0:]  # Skip the category number
            #print(line)
            latitudes = [float(lat) for lat in line[::2]]  # Extract every other value as latitudes
            longitudes = [float(lon) for lon in line[1::2]]  # Extract every other value as longitudes
            contour_data.extend([list(coord) for coord in zip(latitudes, longitudes)])
    return contour_data

#@st.cache_data
def get_latest_created_folder(directory):
    try:
        # Get a list of all folders in the specified directory
        folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

        # Sort the folders by creation time (most recent first)
        folders.sort(key=lambda f: os.path.getctime(os.path.join(directory, f)), reverse=True)

        # Choose the latest created folder (the first one in the sorted list)
        latest_folder = folders[0]

        # Get the full path to the latest created folder
        #latest_folder_path = os.path.join(directory, latest_folder)

        return latest_folder

    except (OSError, IndexError):
        # Handle exceptions if there are no folders in the directory or if there's an error
        return None

# UTM to latitude-longitude conversion function
#@st.cache_data
def utm_to_latlon(utm_x, utm_y):
    proj = pyproj.Transformer.from_crs(f"epsg:32615", f"epsg:4326", always_xy=True)
    lon, lat = proj.transform(utm_x, utm_y)
    return lat, lon


def read_coordinates_from_file(file_path):
    coordinates_list = []
    with open(file_path, 'r') as file:
        for line in file:
            coordinates_data = line.strip().split()
            for coordinate_pair in coordinates_data:
                longitude, latitude = map(float, coordinate_pair.split(','))
                coordinates_list.append([latitude, longitude])
    return coordinates_list


#@st.cache_data
def read_and_extract_coordinates(filename):
    latitudes = []
    longitudes = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 3:
                latitude = float(parts[1])
                longitude = float(parts[2])
                latitudes.append(latitude)
                longitudes.append(longitude)
    return latitudes, longitudes

#create a df with structure type, lat, lon
def extract_data_from_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                parts = line.split()
                category = int(parts[0])
                if len(parts) >= 3:
                    coordinates = [float(parts[i]) for i in range(1, 3)]
                else:
                    coordinates = [float(parts[i]) for i in range(1, len(parts))]
                    while len(coordinates) < 2:
                        coordinates.append(None)
                data.append((category, *coordinates))
    return data

# Function to map category values to structure names
def map_category_to_structure(category):
    if category == 0:
        return 'aguada'
    elif category == 1:
        return 'building'
    elif category == 2:
        return 'platform'
    else:
        return 'unknown'


def read_and_extract_coordinates(filename):
    latitudes = []
    longitudes = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 3:
                latitude = float(parts[1])
                longitude = float(parts[2])
                latitudes.append(latitude)
                longitudes.append(longitude)
    return latitudes, longitudes
