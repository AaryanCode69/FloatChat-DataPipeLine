import os
import requests
from bs4 import BeautifulSoup

# Target URL (only Jan 2020)
url = "https://www.ncei.noaa.gov/data/oceans/argo/gadr/data/indian/2020/01/"

# Local directory to save files
download_dir = "argo_data_2020_01"
os.makedirs(download_dir, exist_ok=True)

def download_nc_files(base_url, max_files=300):
    print(f"Fetching file list from {base_url} ...")
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Find all links ending with .nc
    links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].endswith('.nc')]

    print(f"Found {len(links)} files in 2020/01. Downloading first {max_files} files...")
    for i, link in enumerate(links[:max_files]):
        file_url = base_url + link
        file_path = os.path.join(download_dir, f"2020_01_{i+1:03d}.nc")

        if not os.path.exists(file_path):  # avoid re-downloading
            print(f"[{i+1}/{max_files}] Downloading {file_url}")
            r = requests.get(file_url, stream=True)
            with open(file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        else:
            print(f"File {file_path} already exists, skipping.")

    print("✅ Download complete! Files saved in 'argo_data_2020_01/' folder.")

# Run the downloader
download_nc_files(url, max_files=300)
