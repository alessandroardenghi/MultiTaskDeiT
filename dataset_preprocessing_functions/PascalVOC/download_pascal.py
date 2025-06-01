import os
import requests
from tqdm import tqdm
import tarfile

voc_url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
tar_path = "VOCtrainval_11-May-2012.tar"
extract_path = "./VOCdevkit"

def download_with_progress(url, output_path, timeout=10):
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        with open(output_path, 'wb') as f, tqdm(
            desc=output_path,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=1024):
                size = f.write(chunk)
                bar.update(size)

# DOWNLOADING
if not os.path.exists(tar_path):
    print("Downloading VOC 2012 with progress...")
    download_with_progress(voc_url, tar_path)
else:
    print("VOC tar already exists.")

# EXTRACTING
if not os.path.exists(os.path.join(extract_path, "VOC2012")):
    print("Extracting VOC 2012...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=".")
    print("Extraction complete.")
else:
    print("VOC 2012 already extracted.")