import os

import wget
import zipfile
import argparse


def download(url: str, base_path:str):
    """
    :param url: A url from where to download the data
    :param base_path: where to download the data
    :return:
    """
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    f_name = wget.download(url, f"{base_path}/data.zip")
    print(f"Successfully downloaded file  {f_name}")

    print(f"Unzipping {f_name}")

    with zipfile.ZipFile(f_name) as zipr:
        zipr.extractall(f"{base_path}")

    print("Unzipping Finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()
    download("https://www.dropbox.com/s/4si2ixdtdgabqnt/Cat_Dog.zip?dl=1", args.data_dir)