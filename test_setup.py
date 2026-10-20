import ee
import requests
import os

# --- 1. CONFIGURATION (UPDATED FOR MAX DATA) ---
PROJECT_ID = 'cspkm-474413' 
LAT = 16.3067  # Guntur Region
LON = 80.4365
SCALE_DEGREES = 0.01  # Approx 1km box
IMAGE_PIXELS = 64     # 64x64 small images for Quantum Model

# TIME TRAVEL: Look back 6 years to find plenty of clear days
YEARS = [2019, 2020, 2021, 2022, 2023, 2024]

# SEASONS (Wider range to catch more images)
# Drought: March to June (Peak Summer in AP)
DRY_MONTH_START, DRY_MONTH_END = 3, 6   

# Healthy: October to December (Post-Monsoon/Winter Crops)
# This avoids the heavy clouds of Aug/Sept but keeps the green crops.
WET_MONTH_START, WET_MONTH_END = 10, 12   

# CLOUD TOLERANCE: Increased to 50%
# We accept partly cloudy images now to ensure we get data.
CLOUD_TOLERANCE = 50 
# -----------------------------------------------

def init_earth_engine():
    """Authenticates and initializes with your specific project."""
    try:
        # Tries to initialize directly with your Project ID
        ee.Initialize(project=PROJECT_ID)
        print(f"✅ Connected successfully to Project: {PROJECT_ID}")
    except Exception as e:
        print("⚠️ Authentication required. Opening browser...")
        ee.Authenticate()
        ee.Initialize(project=PROJECT_ID)

def get_region_geometry(lat, lon, scale):
    """Creates a rectangular box around your coordinate."""
    return ee.Geometry.Rectangle([
        lon - scale, lat - scale,
        lon + scale, lat + scale
    ])

def download_images_for_season(season_name, start_month, end_month):
    """Downloads images for a specific season across all defined years."""
    
    print(f"\n--- Searching for {season_name.upper()} images ({start_month}-{end_month}) ---")
    
    # Create folder if it doesn't exist
    save_path = f"dataset/{season_name}"
    os.makedirs(save_path, exist_ok=True)
    
    region = get_region_geometry(LAT, LON, SCALE_DEGREES)
    count = 0

    for year in YEARS:
        # Define start and end dates for this year's season
        start_date = f"{year}-{start_month:02d}-01"
        # Rough end date (28th is safe for all months)
        end_date = f"{year}-{end_month:02d}-28"
        
        # SEARCH SENTINEL-2
        # We select Bands 8 (NIR), 4 (Red), 3 (Green) -> Best for Drought
        collection = (ee.ImageCollection('COPERNICUS/S2_SR')
                      .filterBounds(region)
                      .filterDate(start_date, end_date)
                      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_TOLERANCE)) 
                      .select(['B8', 'B4', 'B3'])) 

        # Get list of images found
        image_list = collection.toList(collection.size())
        list_size = image_list.size().getInfo()
        
        print(f"   Found {list_size} potential images in {year}...")

        for i in range(list_size):
            try:
                # Get the image object
                ee_image = ee.Image(image_list.get(i))
                date_taken = ee_image.date().format('YYYY-MM-dd').getInfo()
                
                # Generate Download URL
                url = ee_image.getThumbURL({
                    'min': 0,
                    'max': 3000,
                    'dimensions': IMAGE_PIXELS,
                    'region': region,
                    'format': 'png'
                })
                
                # Download and Save
                response = requests.get(url)
                file_name = f"{save_path}/{season_name}_{date_taken}_{i}.png"
                
                with open(file_name, 'wb') as f:
                    f.write(response.content)
                
                count += 1
                print(f"      Saved: {file_name}")
                
            except Exception as e:
                print(f"      Skipped image {i}: {e}")

    print(f"✅ Total {season_name} images saved: {count}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    init_earth_engine()
    
    print(f"Target Location: {LAT}, {LON}")
    print("Starting Download Process...")
    print("This may take a few minutes depending on internet speed.")
    
    # 1. Download Drought Images (Summer)
    download_images_for_season("drought", DRY_MONTH_START, DRY_MONTH_END)
    
    # 2. Download Healthy Images (Winter/Post-Monsoon)
    download_images_for_season("healthy", WET_MONTH_START, WET_MONTH_END)
    
    print("\n🎉 DONE! Check your 'dataset' folder.")