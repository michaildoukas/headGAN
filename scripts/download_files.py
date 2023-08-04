import zipfile
import wget
import os
import argparse

class MyProgressBar():
    def __init__(self, message):
        self.message = message

    def get_bar(self, current, total, width=80):
        print(self.message + ": %d%%" % (current / total * 100), end="\r")

def unzip_file(file_name, unzip_path):
    zip_ref = zipfile.ZipFile(file_name, 'r')
    zip_ref.extractall(unzip_path)
    zip_ref.close()
    os.remove(file_name)

def main():
    # Download urls
    download_urls = ['https://www.dropbox.com/scl/fi/u5sopigekavoqvtloz44j/files.zip?rlkey=qtog3kpbauxg7foxbjmj9jeid&dl=1',
                     'https://www.dropbox.com/scl/fi/kyiilyxnwxswzq6gpuv69/checkpoints.zip?rlkey=pqtj5xk0dkfktlbwdk1ealqou&dl=1',
                     'https://www.dropbox.com/scl/fi/g4o2t3tvr3smkw75o1njt/retinaface_r50_v1.zip?rlkey=n9nwaf6sgg3jqq8ho37n8ec1b&dl=1',
                     'https://www.dropbox.com/scl/fi/kc7zjlg4w1innmiwslk6h/resample.zip?rlkey=wuijjsgn9vn5est43akdg4x8b&dl=1']
                     
    for i, path in enumerate(download_urls):
        fname = path.split('/')[-1].split('?')[0]
        if fname == 'retinaface_r50_v1.zip':
            dir = os.path.join(os.path.expanduser('~'), '.insightface/models/')
            if not os.path.exists(dir):
                os.makedirs(dir)
        elif fname == 'resample.zip':
            dir = os.path.join(os.path.expanduser('~'), '.local/lib/python3.7/site-packages')
            if not os.path.exists(dir):
                os.makedirs(dir)
        else:
            dir = './'
        fpath = os.path.join(dir, fname)
        if not os.path.exists(fpath):
            bar = MyProgressBar('Downloading file %d/%d' % (i+1,
                                len(download_urls)))
            wget.download(path, fpath, bar=bar.get_bar)
            print('\n')
            print('Unzipping file...')
            unzip_file(fpath, dir)
    print('DONE!')

if __name__ == "__main__":
    main()
