"""
Downloads the following:
- Stanford parser
- Stanford POS tagger
- TREC data set(question classification task)
"""

from __future__ import print_function
import urllib2
import sys
import os
import zipfile
import gzip
import shutil


def download(url, dir_path):
    filename = url.split('/')[-1]
    file_path = os.path.join(dir_path, filename)
    try:
        u = urllib2.urlopen(url)
    except:
        print("URL %s failed to open" % url)
        raise Exception
    try:
        f = open(file_path, 'wb')
    except:
        print("Cannot write %s" % file_path)
        raise Exception
    try:
        file_size = int(u.info().getheaders("Content-Length")[0])
    except:
        print("URL %s failed to report length" % url)
        raise Exception
    print("Downloading: %s Bytes: %s" % (filename, file_size))

    downloaded = 0
    block_sz = 8192
    status_width = 70
    while True:
        buf = u.read(block_sz)
        if not buf:
            print('')
            break
        else:
            print('', end='\r')
        downloaded += len(buf)
        f.write(buf)
        status = (("[%-" + str(status_width + 1) + "s] %3.2f%%") %
                  ('=' * int(float(downloaded) / file_size * status_width) + '>', downloaded * 100. / file_size))
        print(status, end='')
        sys.stdout.flush()
    f.close()
    return file_path


def unzip(file_path):
    print("Extracting: " + file_path)
    dir_path = os.path.dirname(file_path)
    with zipfile.ZipFile(file_path) as zf:
        zf.extractall(dir_path)
    os.remove(file_path)


def ungz(file_path):
    print("Extracting: " + file_path)
    dir_path = os.path.dirname(file_path)
    with gzip.GzipFile(dir_path) as gf:
        gf.extractall(dir_path)
    os.remove(file_path)


def download_wordvecs(dirpath):
    if os.path.exists(dirpath):
        print('Found Glove vectors - skip')
        return
    else:
        os.makedirs(dirpath)
    url = 'http://www-nlp.stanford.edu/data/glove.840B.300d.zip'
    unzip(download(url, dirpath))
    

def download_trec(dir_path):
    if os.path.exists(dir_path):
        print('Found TREC set - skip')
        return
    else:
        os.makedirs(dir_path)
    train_url = 'http://cogcomp.cs.illinois.edu/Data/QA/QC/train_5500.label'
    test_url = 'http://cogcomp.cs.illinois.edu/Data/QA/QC/TREC_10.label'
    download(train_url, dir_path)
    download(test_url, dir_path)


if __name__ == '__main__':
    base_dir = os.path.dirname('.')

    # data
    data_dir = os.path.join(base_dir, 'data')
    wordvec_dir = os.path.join(data_dir, 'glove')
    trec_dir = os.path.join(data_dir, 'trec')

    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir)

    # libraries
    lib_dir = os.path.join(base_dir, 'lib')

    # download dependencies
    download_wordvecs(wordvec_dir)
    download_trec(trec_dir)
