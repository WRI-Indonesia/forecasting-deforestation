import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.mask import mask
import rasterio
import numpy as np
from scipy.ndimage import distance_transform_edt
import time

##################################
# Provide a log to trace the process into logging file
##################################
import logging

log_file = "far.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logging.getLogger().addHandler(logging.StreamHandler())
_str_decorator = "=" * 20
logging.info(f"\n{_str_decorator} BEGINNING LOG {_str_decorator}")

class ProcessingForecastDeforestation:
    def __init__(self, filter_dict=None, target_crs=None, pixel_size=30):
        """
        Args:
            filter_dict (dict, optional): e.g., {'class': 'forest', 'year': 2001}
            target_crs (str or int, optional): EPSG code or proj string for target CRS.
        """
        self.filter_dict = filter_dict
        self.pixel_size = pixel_size
        self.target_crs = target_crs

    def rasterize(self, gdf_or_path, out_raster_path, burn_value=1):
        """
        Convert AOI or polygon data into raster.
        """
        start = time.time()
        if isinstance(gdf_or_path, str):
            gdf = gpd.read_file(gdf_or_path)
        else:
            gdf = gdf_or_path.copy()
        if self.target_crs:
            gdf = gdf.to_crs(self.target_crs)
        if gdf.empty:
            raise ValueError("No features found after filtering.")
        if self.filter_dict:
            for k, v in filter_dict.items():
                gdf = gdf[gdf[k] == v]

        bounds = gdf.total_bounds
        x_min, y_min, x_max, y_max = bounds

        # Warn if CRS is geographic (degrees)
        if gdf.crs and gdf.crs.is_geographic:
            raise ValueError(
                f"GeoDataFrame is in a geographic CRS ({gdf.crs}), "
                "please project to a meter-based CRS (e.g., UTM) before rasterizing. "
                "Try passing target_crs=32749 for UTM Zone 49S."
            )
    
        width = int((x_max - x_min) / self.pixel_size)
        height = int((y_max - y_min) / self.pixel_size)
        if width <= 0 or height <= 0:
            raise ValueError(
                f"Calculated raster width/height are invalid: width={width}, height={height}. "
                "Check your pixel_size and CRS units!"
            )

        transform = rasterio.transform.from_origin(
            west=x_min,
            north=y_max,
            xsize=self.pixel_size,
            ysize=self.pixel_size
        )

        out_shape = (height, width)
        shapes = ((geom, burn_value) for geom in gdf.geometry)
        raster = rasterize(
            shapes=shapes,
            out_shape=out_shape,
            fill=0,
            transform=transform,
            dtype="uint8"
        )
        profile = {
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
        logging.info(f"Raster written: {out_raster_path} ({width}x{height}, pixel size: {self.pixel_size} units)")

        end = time.time()
        logging.info(f"Process took {end - start:.2f} seconds")
        return out_raster_path

    def clip_raster_to_aoi(self, raster_path, aoi_or_path, out_path, crop=True):
        """
        Clips raster to AOI (Area of Interest) using the AOI GeoDataFrame.
        """
        start = time.time()
        if isinstance(aoi_or_path, str):
            gdf = gpd.read_file(aoi_or_path)
        else:
            gdf = aoi_or_path.copy()
        if self.target_crs:
            gdf = gdf.to_crs(self.target_crs)
        if gdf.empty:
            raise ValueError("No features found after filtering.")
    
        # Ensure AOI and raster have the same CRS
        with rasterio.open(raster_path) as src:
            raster_crs = src.crs
            if gdf.crs != raster_crs:
                gdf = gdf.to_crs(raster_crs)
            geoms = [geom for geom in gdf.geometry]
            out_image, out_transform = mask(src, geoms, crop=crop)
            out_meta = src.meta.copy()

        # Update metadata
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
        with rasterio.open(out_path, "w", **out_meta) as dest:
            dest.write(out_image)
        logging.info(f"Clipped raster written: {out_path}")

        end = time.time()
        logging.info(f"Process took {end - start:.2f} seconds")
        return out_path

    def compute_deforestation(self, forest_t1_path, forest_t2_path, output_raster, nodata=None):
        """
        Compute deforestation: pixels that were forest in t1 but not forest in t2
    
        Args:
            forest_t1_path (str): Path to t1 forest raster
            forest_t2_path (str): Path to t2 forest raster
            output_raster (str): Output raster path
            nodata (int, float, optional): nodata value to mask (default: None)
        """
        start = time.time()
        with rasterio.open(forest_t1_path) as src1, rasterio.open(forest_t2_path) as src2:
            t1 = src1.read(1)
            t2 = src2.read(1)
            profile = src1.profile.copy()
    
            if nodata is not None:
                mask = (t1 == nodata) | (t2 == nodata)
            else:
                mask = np.zeros_like(t1, dtype=bool)
    
            # Deforestation: forest in t1 (1), non-forest in t2 (0)
            change = np.where((t1 == 1) & (t2 == 0) & ~mask, 1, 0).astype("uint8")
            if nodata is not None:
                change[mask] = nodata
                profile["nodata"] = nodata
    
            profile.update(dtype="uint8", count=1)
            with rasterio.open(output_raster, "w", **profile) as dst:
                dst.write(change, 1)
    
        logging.info(f"Deforestation raster saved to {output_raster}")
        end = time.time()
        logging.info(f"Process took {end - start:.2f} seconds")
        return output_raster
        

    def euclidean_distance(self, input_raster, output_raster, dist_meters=100):
        """
        Calculates Euclidean distance to the nearest raster pixel and outputs distance in units of dist_meters
        
        Args:
            input_raster (str): Path to geotiff
            output_raster (str): Output raster path
            dist_meters (float): Distance unit (e.g., 100 for 100m units)
        """
        start = time.time()
        with rasterio.open(input_raster) as src:
            mask = src.read(1)
            profile = src.profile.copy()
            self.pixel_size = src.transform[0]
    
        invert_mask = (mask == 0).astype(np.uint8)
        dist_pixels = distance_transform_edt(invert_mask)
        dist_in_units = (dist_pixels * self.pixel_size) / dist_meters
    
        profile.update(dtype='float32')
        with rasterio.open(output_raster, "w", **profile) as dst:
            dst.write(dist_in_units.astype("float32"), 1)
    
        logging.info(f"Euclidean distance raster saved as {output_raster}")
        end = time.time()
        logging.info(f"Process took {end - start:.2f} seconds")
        
        return output_raster