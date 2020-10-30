from google_drive_downloader import GoogleDriveDownloader as gdd
import subprocess


DATA_RAW_ZIP_LINK = 'https://drive.google.com/u/0/uc?export=download&confirm=OA3A&id=1CGs71ILc3mQLOzPUD-FcT215h2jeW7Yz'
DATA_RAW_PATH = '../../data/raw/'
FILENAME = 'raw-data.zip'



if __name__ == '__main__':
    print('Starting download...')
    gdd.download_file_from_google_drive(file_id='1CGs71ILc3mQLOzPUD-FcT215h2jeW7Yz',
                                    dest_path=DATA_RAW_PATH+FILENAME,
                                    unzip=False)
    print('Download finished! Please unzip the file at PROJECT_ROOT/data/raw/raw-data.zip')

