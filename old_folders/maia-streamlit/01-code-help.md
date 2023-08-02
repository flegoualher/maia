Code help for Streamlit - point cloud visualization

```python
import streamlit as st
import laspy
import plotly.express as px

def main():
    st.title("3D Scatter Plot of LAS/LAZ file with Streamlit and Plotly")

    # Create a file uploader widget
    uploaded_file = st.file_uploader("Upload a LAS/LAZ file", type=["las", "laz"])

    if uploaded_file is not None:
        # Read the uploaded LAS/LAZ file using laspy
        with laspy.open(uploaded_file) as las:
            # Access the x, y, and z coordinates from the LAS/LAZ file
            x = las.x
            y = las.y
            z = las.z

        # Create a DataFrame with the x, y, and z coordinates
        data = pd.DataFrame({'x': x, 'y': y, 'z': z})

        # Display the 3D scatter plot using Plotly Express
        fig = px.scatter_3d(data, x='x', y='y', z='z')
        st.plotly_chart(fig)


if __name__ == "__main__":
    main()
```

