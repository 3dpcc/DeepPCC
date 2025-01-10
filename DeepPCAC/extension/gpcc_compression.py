import os, time
import numpy as np
import subprocess
from tqdm import tqdm
rootdir = os.path.split(__file__)[0]
if rootdir == '': rootdir = '.'


def get_points_number(filedir):
    plyfile = open(filedir)
    line = plyfile.readline()
    while line.find("element vertex") == -1:
        line = plyfile.readline()
    number = int(line.split(' ')[-1][:-1])

    return number


def number_in_line(line):
    wordlist = line.split(' ')
    for _, item in enumerate(wordlist):
        try:
            number = float(item)
        except ValueError:
            continue

    return number


def gpcc_encode(filedir, bin_dir, posQuantscale=1, transformType=0, qp=22, test_time=False, show=False):
    """Compress point cloud losslessly using MPEG G-PCCv14.
    You can download and install TMC13 from https://github.com/MPEGGroup/mpeg-pcc-tmc13
    """
    config = ' --trisoupNodeSizeLog2=0' + \
             ' --neighbourAvailBoundaryLog2=8' + \
             ' --intra_pred_max_node_size_log2=6' + \
             ' --maxNumQtBtBeforeOt=4' + \
             ' --planarEnabled=1' + \
             ' --planarModeIdcmUse=0' + \
             ' --minQtbtSizeLog2=0' + \
             ' --positionQuantizationScale=' + str(posQuantscale)
    # lossless
    if posQuantscale == 1:
        config += ' --mergeDuplicatedPoints=0' + \
                  ' --inferredDirectCodingMode=1'
    else:
        config += ' --mergeDuplicatedPoints=1'
    # attr (raht)
    if qp is not None:
        config += ' --convertPlyColourspace=1'
        if transformType == 0:
            config += ' --transformType=0'
        elif transformType == 2:
            print('dbg:\t transformType=', transformType)
            config += ' --transformType=2' + \
                      ' --numberOfNearestNeighborsInPrediction=3' + \
                      ' --levelOfDetailCount=12' + \
                      ' --lodDecimator=0' + \
                      ' --adaptivePredictionThreshold=64'
        else:
            raise ValueError('transformType=' + str(transformType))
        config += ' --qp=' + str(qp) + \
                  ' --qpChromaOffset=0' + \
                  ' --bitdepth=8' + \
                  ' --attrOffset=0' + \
                  ' --attrScale=1' + \
                  ' --attribute=color'
    # headers
    headers = ['positions bitstream size', 'Total bitstream size']
    if test_time: headers += ['positions processing time (user)', 'Processing time (user)', 'Processing time (wall)']
    if qp is not None: headers += ['colors bitstream size']
    if qp is not None and test_time:  headers += ['colors processing time (user)']

    #
    subp = subprocess.Popen(rootdir + '/tmc3 --mode=0' + config + \
                            ' --uncompressedDataPath=' + filedir + \
                            ' --compressedStreamPath=' + bin_dir,
                            shell=True, stdout=subprocess.PIPE)
    results = {}
    c = subp.stdout.readline()
    while c:
        if show: print(c)
        line = c.decode(encoding='utf-8')
        for _, key in enumerate(headers):
            if line.find(key) != -1:
                value = number_in_line(line)
                results[key] = value
        c = subp.stdout.readline()

    return results


def gpcc_decode(bin_dir, rec_dir, attr=True, test_geo=False, test_attr=False, show=False):
    if attr:
        config = ' --convertPlyColourspace=1'
    else:
        config = ''
    subp = subprocess.Popen(rootdir + '/tmc3 --mode=1' + config + \
                            ' --compressedStreamPath=' + bin_dir + \
                            ' --reconstructedDataPath=' + rec_dir + \
                            ' --outputBinaryPly=0',
                            shell=True, stdout=subprocess.PIPE)
    # headers
    if test_geo: headers = ['positions bitstream size', 'positions processing time (user)',
                            'Total bitstream size', 'Processing time (user)', 'Processing time (wall)']
    if test_attr: headers += ['colors bitstream size', 'colors processing time (user)']
    headers = []
    results = {}
    c = subp.stdout.readline()
    while c:
        if show: print(c)
        line = c.decode(encoding='utf-8')
        for _, key in enumerate(headers):
            if line.find(key) != -1:
                value = number_in_line(line)
                results[key] = value
        c = subp.stdout.readline()

    return results



if __name__ == '__main__':
    import glob, argparse, os

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument("--filedir", type=str, default='tp', help="prefix of checkpoints/logger, etc.")
    parser.add_argument("--posQuantscale", type=int, default=1)
    parser.add_argument("--transformType", type=int, default=0)
    parser.add_argument("--qp", type=int, default=46)
    parser.add_argument("--gpcc_version", type=int, default=14)
    args = parser.parse_args()
    input_rootdir = '/media/ivc3090ti/新加卷/zjz/data/MPEG_Training/Partitioned/Owlii'
    bin_rootdir = '/media/ivc3090ti/新加卷/zjz/data/MPEG_Training/GPCC_Compressioned/Owlii/qp46/bin_qp46';
    os.makedirs(bin_rootdir, exist_ok=True)
    rec_rootdir = '/media/ivc3090ti/新加卷/zjz/data/MPEG_Training/GPCC_Compressioned/Owlii/qp46';
    os.makedirs(rec_rootdir, exist_ok=True)
    input_filedirs = sorted(glob.glob(os.path.join(input_rootdir, '**', f'*' + 'ply'), recursive=True))


    for i, input_filedir in enumerate(tqdm(input_filedirs)):
        bin_dir = os.path.join(bin_rootdir, input_filedir.split('/')[-1].split('.')[0] + '.bin')
        rec_dir = os.path.join(rec_rootdir, input_filedir.split('/')[-1].split('.')[0] + '.ply')
        results_enc = gpcc_encode(input_filedir, bin_dir, posQuantscale=args.posQuantscale,
                                      transformType=args.transformType, qp=args.qp, show=False)
        results_dec = gpcc_decode(bin_dir, rec_dir, show=False)

    os.system('rm -rf' + ' ' + bin_rootdir)
