import geopandas as gpd
import rasterio
from rasterio.features import rasterize

class Rasterizer:
    def __init__(self, shp_path, filter_dict=None, target_crs=None):
         """
        Args:
            shp_path (str): Path to shapefile.
            filter_dict (dict, optional): e.g., {'class': 'forest', 'year': 2001}
            target_crs (str or int, optional): EPSG code or proj string for target CRS.
        """
        gdf = gpd.read_file(shp_path)
        if target_crs:
            self.gdf = gdf.to_crs(target_crs)
        if filter_dict:
            for k, v in filter_dict.items():
                gdf = gdf[gdf[k] == v]
        self.gdf = gdf

    def rasterize(self, out_raster_path, pixel_size=30):
        gdf = self.gdf
        if gdf.empty:
            raise ValueError("No matching features found for the given class/year.")
        bounds = gdf.total_bounds
        
        transform = rasterio.transform.from_origin(
            west=bounds[0],
            north=bounds[3],
            xsize=pixel_size,
            ysize=pixel_size
        )
        width = int((bounds[2] - bounds[0]) / pixel_size)
        height = int((bounds[3] - bounds[1]) / pixel_size)
        out_shape = (height, width)
        shapes = ((geom, 1) for geom in gdf.geometry)
        raster = rasterize(
            shapes=shapes,
            out_shape=out_shape,
            fill=0,
            transform=transform,
            dtype="uint8"
        )
        profile= {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "uint8",
            "crs": gdf.crs,
            "transform": transform
        }
        
        with rasterio.open(out_raster_path, "w", **profile) as dst:
            dst.write(raster, 1)
            return out_raster_path