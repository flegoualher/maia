# Python In-built packages
from pathlib import Path
import PIL
import laspy
import settings
import numpy as np
import pandas as pd
import laspy
import plotly.express as px
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
import folium
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import streamlit as st
from streamlit_folium import folium_static
import csv
import pyproj


# External packages
import streamlit as st


# Local Modules
import settings
import helper


#Import functions from helper.py
from helper import create_tile_tiff, read_contour_data, get_latest_created_folder, utm_to_latlon, read_coordinates_from_file
from helper import extract_data_from_file, map_category_to_structure, read_and_extract_coordinates


# Setting page layout
st.set_page_config(
    page_title="MAIA - YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
#st.title("Let's discover the mysteries of the Maya")

# UTM zone and hemisphere information
utm_zone = 15
northern_hemisphere = True

# Sidebar
st.sidebar.image("images/31161816_transparent.png")


# Model Options
model_type = 'Segmentation'
model_path = Path(settings.SEGMENTATION_MODEL)



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

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Lidar tile upload")
#source_radio = st.sidebar.radio(
    #"Select Source", settings.SOURCES_LIST)

source_radio = 'Image'
source_img = None
point_cloud_upload = None

################################################################################
#UPLOADER
################################################################################

point_cloud_upload = st.sidebar.file_uploader(
    "Upload point cloud data here...", type=(".las", ".laz"))# , accept_multiple_files=True)

st.sidebar.header("ML Model Config")
confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100
################################################################################
#POINT CLOUD IMAGE
################################################################################
try:
    if point_cloud_upload is None:
        st.title("Nothing uploaded... yet ! ")
        st.write("To perform structure detection")
        st.write('- drag and drop your Lidar file')
        st.write("- let the magic happen !")
        st.write('''In the meantime, enjoy the view of the "Aguada Fénix" Mayan City, Mexico''')
        image = PIL.Image.open(settings.DEFAULT_POINT_CLOUD)
        st.image(image, width = 775)

    else:
        st.title("Let's go find some ancient settlements !")
        st.write("Point cloud file uploaded !")
        point_cloud = laspy.read(point_cloud_upload)
        #store coordinates in "points", and colors in "colors" variable
        lidar_data = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z, point_cloud.classification)).transpose()
        df = pd.DataFrame(lidar_data, columns = ['x','y','z','category'])
        df['category']=df['category'].astype(int).astype(str)
        df['category']=[category_name_dict[i] for i in df['category']]
        df['color']= [color_dict[i] for i in df['category']]
        factor = 15  #16
        df_decimated = df.iloc[::factor]
        # Define custom category colors using hexadecimal values
        category_colors = color_dict
        fig = px.scatter_3d(df_decimated, x='x', y='y', z='z', color='category', hover_data=['category'],  color_discrete_map=category_colors)
        fig.update_traces(marker={"size" : 1})
        fig.update_layout(scene=dict(xaxis=dict(showgrid=False),
                                    yaxis=dict(showgrid=False),
                                    zaxis=dict(showgrid=False)))
        fig.update_layout(scene=dict(xaxis=dict(showbackground=False),
                                    yaxis=dict(showbackground=False),
                                    zaxis=dict(showbackground=False)))
        fig.update_layout(scene=dict(xaxis=dict(showticklabels=False),
                                    yaxis=dict(showticklabels=False),
                                    zaxis=dict(showticklabels=False)))
        #figsize
        fig.update_layout(width=800, height=600)
        st.plotly_chart(fig)

        create_tile_tiff(df_decimated)
        #source_img = PIL.Image.open("images/rgb_tile_georef.tiff")
        source_img = "images/rgb_tile_georef.tiff"
except Exception as ex:
        st.error("Nothing.")
        st.error(ex)


#source_img = PIL.Image.open("images/rgb_tile_georef.tiff")


################################################################################
#TIFF + PREDICTION (2 columns)
################################################################################
col1, col2 = st.columns(2)
##################
#col1 : tiff Image
##################
with col1:
    try:
        if source_img is None:
            st.write(" ")
            # default_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            # default_image = PIL.Image.open(default_image_path)
            # st.image(default_image_path, caption="Lidar Image Default",
            #          use_column_width=True)
        else:
            #uploaded_image = PIL.Image.open(source_img)
            st.image(source_img, caption="Lidar Point Cloud Image",
                     use_column_width=True)
    except Exception as ex:
        st.error("Error occurred while opening the image.")
        st.error(ex)

#############################
#col2 : yolo prediction Image
#############################
with col2:
    if source_img is None:
        st.write(" ")
        # default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE_2)
        # default_detected_image = PIL.Image.open(
        #     default_detected_image_path)
        # st.image(default_detected_image_path, caption='Detected Image',
        #          use_column_width=True)
    else:
        res = model.predict(source_img,conf=confidence, save_txt = True)
        boxes = res[0].boxes
        res_plotted = res[0].plot()[:, :, ::-1]
        st.image(res_plotted, caption='Detection Results',
                     use_column_width=True)


#Create georeferenced label txt file
# Replace 'your_directory_path' with the path to your specific directory
directory_path = 'runs/segment'
latest_folder = get_latest_created_folder(directory_path)
label_path = 'runs/segment/' + latest_folder + '/labels/rgb_tile_georef.txt'

def add_marker(row):
    lat, lon, structure = row['latitude'], row['longitude'], row['structure']
    folium.Marker([lat, lon], popup=structure).add_to(m)

# Load the georeferenced TIFF file
tif_file_path = "images/rgb_tile_georef.tiff"
with rasterio.open(tif_file_path) as src:
    #print(src.crs)
    # Get the geotransform (affine transformation) of the raster
    geotransform = src.transform
    #print(geotransform)
    # Loop through each contour in the text file
    try:
        with open(label_path, "r") as file:
            # Open a new text file for writing the converted data
            with open("images/rgb_tile_georef_geoconverted.txt", "w") as output_file:
                for line in file:
                    # Split the line into individual contour coordinates
                    data = line.strip().split(",")
                    data = data[0].split()
                    #print(data)
                    #category
                    category = data[0]
                    # Skip the first value (category) and convert the rest to float
                    contour_coordinates = [float(coord) for coord in data[1:]]
                    #print(contour_coordinates)
                    # List to store the converted longitude and latitude values
                    converted_coordinates = []
                    #converted_coordinates.append(str(category))
                    # Iterate through the pairs of x, y coordinates (assumes x, y pairs)
                    for i in range(0, len(contour_coordinates), 2):
                        x, y = contour_coordinates[i-1], contour_coordinates[i]
                        # Transform pixel coordinates (x, y) to geographic coordinates (longitude, latitude)
                        X, Y = rasterio.transform.xy(geotransform, x, y)
                        lon, lat = utm_to_latlon(X, Y)
                        # convert lat and lon
                        # Append the converted longitude and latitude to the list
                        converted_coordinates.append(str(lon))
                        converted_coordinates.append(str(lat))
                # Write the converted data to the output file
                    output_file.write(category + " " + " ".join(converted_coordinates) + "\n")
        # Replace 'your_file.txt' with the actual path to your file containing the numbers
        file_path = 'images/rgb_tile_georef_geoconverted.txt'
        latitudes, longitudes = read_and_extract_coordinates(file_path)
        latitude = latitudes[0]
        longitude = longitudes[0]
        #st.write(latitude)
        #st.write(longitude)


        #create df with all detected structures
        result = extract_data_from_file(file_path)
        # Create a pandas DataFrame from the extracted data
        df = pd.DataFrame(result, columns=['category', 'latitude', 'longitude'])
        # Apply the function to create the 'structure' column
        df['structure'] = df['category'].apply(map_category_to_structure)
        aguada_number = len(df[df['category'] == 0])
        building_number = len(df[df['category'] == 1])
        platform_number = len(df[df['category'] == 2])
        total_number = aguada_number + building_number + platform_number
        map_center = [df['latitude'].iloc[0], df['longitude'].iloc[0]]

        ##############################################
        # Map
        ##############################################

        m = folium.Map(location=map_center, zoom_start=5)
        esri_satellite = folium.TileLayer(
                    tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                    attr = 'Esri',
                    name = 'Esri Satellite',
                    overlay = True,
                    control = True)

        # Apply the add_marker function to each row in the DataFrame
        df.apply(add_marker, axis=1)

        # Inomata contour
        for i in range(1,37+1):
            inomata_contour = read_coordinates_from_file(f'images/inomata_contour/inomata_{i}.txt')
            folium.PolyLine(inomata_contour,
                                    color='red',
                                    tooltip="Inomata",
                                    weight=3,  # line thickness
                                    opacity=0.8  # transparency
                                    ).add_to(m)



        esri_satellite.add_to(m)
        folium.Marker([latitude, longitude]).add_to(m)
        folium_static(m)

        st.write(f"Congratulations ! {total_number} structures detected !")
        st.write(f"Aguadas : {aguada_number} - Buildings : {building_number} - Platforms : {platform_number}")
        st.dataframe(df)

    except FileNotFoundError:
    # Custom error message for the FileNotFoundError
        st.write(" ")
