import os
import tarfile
from six.moves import urllib
import pandas as pd


def data_fetch_fn(data_url, data_path,data_name):
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    tgz_path = os.path.join(data_path, data_name+'.tgz')
    urllib.request.urlretrieve(data_url,tgz_path)
    data_tgz= tarfile.open(tgz_path)
    data_tgz.extractall(path=data_path)
    data_tgz.close()


def load_data(data_path,data_name):
    csv_path=os.path.join(data_path,data_name+'.csv')
    return pd.read_csv(csv_path)
