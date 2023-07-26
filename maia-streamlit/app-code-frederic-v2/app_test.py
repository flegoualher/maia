import streamlit as st
import numpy as np
import pandas as pd
import laspy
import plotly.express as px


#st.markdown("Step 1 : point cloud visualization")


def page_upload_las():
    #Load data and check point format

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

        factor=5#16
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

        st.write("Point Cloud Data:")
        fig.show()

        if st.button("Next"):
            # Store the point cloud in session_state to pass to the next page
            st.session_state.point_cloud = df_decimated
            st.experimental_set_query_params(page="display_tiff")


def page_display_tiff():
    st.title('TIFF Image of Point Cloud')
    if 'point_cloud' not in st.session_state:
        st.warning("Please upload a LAS file first.")
        return

    point_cloud = st.session_state.point_cloud
    # Process the point cloud data and create the TIFF image (you'll need to implement this part)
    # For this example, we'll just create a dummy image for demonstration purposes
    dummy_image = Image.new('RGB', (100, 100), color='red')

    st.image(dummy_image, caption="TIFF Image", use_column_width=True)



def main():
    st.set_page_config(layout="wide")

    pages = {
        "upload_las": page_upload_las,
        "display_tiff": page_display_tiff
    }

    page = st.experimental_get_query_params().get("page", ["upload_las"])[0]

    if page in pages:
        pages[page]()
    else:
        st.error("Invalid page. Please go back to the upload page.")

if __name__ == "__main__":
    main()
