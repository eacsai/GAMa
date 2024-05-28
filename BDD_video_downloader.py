import os
import requests
from tqdm import tqdm

def download_file(url, save_path):
    local_filename = url.split('/')[-1]
    full_path = os.path.join(save_path, local_filename)
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kilobyte

    tqdm_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=local_filename)
    with open(full_path, 'wb') as f:
        for data in response.iter_content(block_size):
            tqdm_bar.update(len(data))
            f.write(data)
    tqdm_bar.close()
    
    if total_size != 0 and tqdm_bar.n != total_size:
        print("ERROR: Something went wrong during download")

    return full_path

def download_videos(base_url, start_index, end_index, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(start_index, end_index + 1):
        url = f"{base_url}{i:02d}.zip"
        print(f"Downloading {url} ...")
        try:
            download_file(url, save_path)
            print(f"Downloaded {url}")
        except Exception as e:
            print(f"Failed to download {url}: {e}")

if __name__ == "__main__":
    base_url = "http://dl.yf.io/bdd100k/video_parts/bdd100k_videos_test_"
    start_index = 0
    end_index = 19
    save_path = "/public/home/shiyj2-group/video_localization/BDD"

    download_videos(base_url, start_index, end_index, save_path)
    print("All downloads completed.")
