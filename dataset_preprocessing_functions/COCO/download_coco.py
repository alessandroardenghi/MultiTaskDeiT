import os
import requests
from zipfile import ZipFile
from tqdm import tqdm

# URLs for COCO dataset
coco_images_url = "http://images.cocodataset.org/zips/val2014.zip"
coco_labels_url = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"

# Directory to save the files
output_dir = "coco_data"
os.makedirs(output_dir, exist_ok=True)

def download_and_extract(url, output_dir):
    local_filename = os.path.join(output_dir, url.split('/')[-1])
    
    # Streaming download with progress bar
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        with open(local_filename, 'wb') as f, tqdm(
            desc=f"Downloading {local_filename}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))

    # Extract zip file
    print(f"Extracting {local_filename}...")
    with ZipFile(local_filename, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    print(f"Extraction complete: {local_filename}\n")


download_and_extract(coco_images_url, output_dir)
download_and_extract(coco_labels_url, output_dir)
