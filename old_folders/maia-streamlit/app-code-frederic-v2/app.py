import streamlit as st
import numpy as np
import pandas as pd
import laspy
import plotly.express as px
from tile_creation import create_tile
import laspy
from PIL import Image
import matplotlib.pyplot as plt
import rvt.default
import rvt.blend
import rasterio
import geopandas as gpd
import os
import pyproj
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
#from ultralytics import YOLO


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

def display_VAT_HS(df_elevation,width,height, sampling):
    slope_arr = 1 - compute_slope(df_elevation,width,height, sampling)
    svf_arr, opns_arr = compute_svf_opns(df_elevation,width,height, sampling)
    VAT_HS = np.dstack((slope_arr,opns_arr, svf_arr))
    return VAT_HS


def create_tile(df_decimated):
    #Create df
    df_elevation_input = df_decimated[df_decimated['category']=='ground']
    df_elevation_input = df_elevation_input.drop(columns=['category', 'color'])
    #print(df_elevation_input.shape)

    #Create elevation input
    #proj = "EPSG:32615"

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
    slope_arr = compute_slope(df_elevation_output, width, height, SAMPLING_IN_M)
    svf_arr, opns_arr = compute_svf_opns(df_elevation_output, width, height, SAMPLING_IN_M)

    #create rgb_image
    rgb_image = display_VAT_HS(df_elevation_output,width,height, SAMPLING_IN_M)
    # Convert the VAT_HS array to PIL Image
    image = Image.fromarray(np.uint8(rgb_image * 255))
    return image
    # plt.imshow(rgb_image)
    # plt.axis('off')
    # plt.show()



    # Save the image as a JPEG file
    #print(f"Tile {tile_name}.jpeg created ! ✅ ")
    #return image.save(f"{path}/jpeg_tiles/{tile_name}.jpeg")





#from streamlit.report_thread import get_report_ctx
#import session_state


# Create a dictionary to store page state
#session_state = st.session_state

# Page content functions
def page1():
    st.title("Page 1")
    st.write("Welcome to Page 1")
    uploaded_file = st.file_uploader("Upload a LAS/LAZ file", type=["las", "laz"])
    if uploaded_file is not None:
        point_cloud = laspy.read(uploaded_file)

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

        #store coordinates in "points", and colors in "colors" variable
        lidar_data = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z, point_cloud.classification)).transpose()
        df = pd.DataFrame(lidar_data, columns = ['x','y','z','category'])
        df['category']=df['category'].astype(int).astype(str)
        df['category']=[category_name_dict[i] for i in df['category']]
        df['color']= [color_dict[i] for i in df['category']]

        factor=5  #16
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
        #fig.show()

        # Store the DataFrame in session_state
        st.session_state.df = df_decimated



def page2():
    st.title("Page 2")
    st.write("Welcome to Page 2")

    # Check if the DataFrame exists in session_state
    if 'df' in st.session_state:
        # Access the DataFrame from session_state
        df_decimated = st.session_state.df
        # st.write(df_decimated.shape)
        #st.dataframe(data=df_decimated)
        #Display the tiff tile
        SAMPLING_IN_M =1
        tile_image = create_tile(df_decimated)
        st.image(tile_image)
    else:
        st.write("No data available. Please visit Page 1 to upload a .laz file.")

    # Store the image in session_state
    st.session_state.image = tile_image




def page3():
    st.title("Page 3")
    st.write("Welcome to Page 3")
    # Add content for page 3
    # if 'image' in st.session_state:
    #     tile_image = st.session_state.image
    #     # Access the image from session_state
    #     st.image(tile_image)
    # # Load a model
    # model = YOLO('trained_model_weights/best.pt')  # pretrained YOLOv8n model

    # # Run batched inference on a list of images
    # results = model(['im1.jpg', 'im2.jpg'])  # return a list of Results objects

    # else:
    #     st.write("No image available.")





# Create a list of page functions
pages = [page1, page2, page3]

# Initialize page_number if not already present in session_state
if 'page_number' not in st.session_state:
    st.session_state.page_number = 0

# Navigation buttons
nav_button1 = st.button("Previous")
nav_button2 = st.button("Next")

# Display the appropriate page based on the page_number
if nav_button1:
    st.session_state.page_number += -1
if nav_button2:
    st.session_state.page_number += 1

# Reset page_number if it exceeds the number of pages
if st.session_state.page_number >= len(pages):
    st.session_state.page_number = 0

# Call the page function based on the current page_number
pages[st.session_state.page_number]()
