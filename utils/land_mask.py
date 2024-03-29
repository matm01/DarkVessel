import geopandas as gpd
import json
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.plot import show


# Reproject the GeoDataFrame to the target CRS
def reproject_geometry(gdf, dst_crs):
    gdf_reprojected = gdf.to_crs(dst_crs)
    gdf_reprojected['area'] = gdf_reprojected['geometry'].area
    return gdf_reprojected

    
# Clip the image using the ocean mask: assign nan 
def clip_image(image_path, geojson_path, output_file=None):
    # Load the GeoJSON file
    with open(geojson_path, 'r') as file:
        geojson_data = json.load(file)
        geojson_string = json.dumps(geojson_data)  # Convert the GeoJSON data to a string
        gdf = gpd.read_file(geojson_string)  # Read into GeoPanadas DataFrame
    # Load the TIFF image
    with rasterio.open(image_path) as src:
        # Reproject the mask geometry to the target CRS
        gdf = reproject_geometry(gdf, dst_crs=src.crs)  
        # Clip the TIFF image using the MultiPolygon geometry
        out_image, out_transform = mask(src, 
                                        gdf.geometry,
                                        crop=True,
                                        nodata=np.nan)
        # Define the metadata
        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})
        
        # Save the clipped image to a new TIFF file
        if output_file:
            with rasterio.open(output_file, 'w', **out_meta) as dest:
                dest.write(out_image)
        else:
            out_image = np.squeeze(out_image, axis=0)
    return out_image, out_meta


if __name__ == '__main__':
    geojson_path = ... # Path to the GeoJSON file containing the ocean mask
    image_path = ...  # Path to the image to be clipped
    output_file = ...  # Path to the output file
    clipped_image, clipped_meta = clip_image(image_path, geojson_path, output_file)
    print(f"Clipped image saved to {output_file}")
    print(f"Clipped image metadata: {clipped_meta}")
    show(clipped_image, cmap='gray')