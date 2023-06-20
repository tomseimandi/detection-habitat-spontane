# This is a sample file to train s2_looking_dataset without bothering this initial implementation.
import s3fs
import zipfile 
from utils.utils import update_storage_access


def download_data_s2_looking():
    """
    Download the zip file which contain the data for the s2looking dataset
    """
    # print("Téléchargement du dataset : S2Looking")

    path_to_data = "projet-slums-detection/Donnees/PAPERS/S2Looking"
    path_to_local = "/home/onyxia/work/detection-habitat-spontane/data/"
    update_storage_access()

    fs = s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"}
    )
    fs.download(rpath=path_to_data, lpath=path_to_local, recursive=True)
    return path_to_local


def dezip_dataset(path_to_zip: str, directory_to_extract_to: str):
    with zipfile.ZipFile(path_to_zip, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)



if __name__ == "__main__":
    year = 2022
    dep = "972"
    src = "PLEIADES"
    download_data_s2_looking()