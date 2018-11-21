#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import math
import requests
import pandas as pd

URL_PEAL_CALL_BED = "http://dbarchive.biosciencedbc.jp/kyushu-u/{genome}/" \
                    "eachData/bed{threshold:0>2}/{srx}.{threshold:0>2}.bed"
GENOME = 'hg19'
PEAK_CALL_THRESHOLD = 5


def get_bed_file_path(wk_dir, srx, threshold=PEAK_CALL_THRESHOLD):
    """

    :param wk_dir:
    :type wk_dir: str
    :param srx:
    :type srx: str
    :param threshold:
    :type threshold: int
    :rtype: str
    """
    if not os.path.exists(wk_dir):
        os.mkdir(wk_dir)

    url = URL_PEAL_CALL_BED.format(genome=GENOME, threshold=threshold, srx=srx)
    bed_path = os.path.join(wk_dir, os.path.basename(url))
    if not os.path.exists(bed_path):
        r = requests.get(url)
        if r.status_code == 200:
            try:
                with open(bed_path, 'wb') as f:
                    f.write(r.content)
                return bed_path
            except OSError:
                if os.path.exists(bed_path):
                    os.remove(bed_path)
                sys.stderr.write('write error:{}\n'.format(bed_path))
        else:
            sys.stderr.write('http error({}):{}\n'.format(r.status_code, url))
    else:
        return bed_path

    return None


if __name__ == "__main__":
    import argparse


    def parse_args():
        parser = argparse.ArgumentParser(
            description='extract sequence positions from BED file...')
        parser.add_argument('-v', '--version',
                            action='version',
                            version='%(prog)s 1.0.0')
        parser.add_argument('srx',
                            action='store',
                            nargs=None,
                            const=None,
                            default=None,
                            type=str,
                            choices=None,
                            help='SRX name or BED file path.',
                            metavar=None)
        parser.add_argument('seq_len',
                            action='store',
                            nargs=None,
                            const=None,
                            default=None,
                            type=int,
                            choices=None,
                            help='peak sequence length.',
                            metavar=None)
        parser.add_argument('-w', '--work-dir',
                            action='store',
                            nargs=None,
                            const=None,
                            default=None,
                            type=str,
                            choices=None,
                            help='BED file directory.',
                            metavar='DIR')
        parser.add_argument('-o', '--output-dir',
                            action='store',
                            nargs=None,
                            const=None,
                            default='.',
                            type=str,
                            choices=None,
                            help='output directory.'
                                 '(default: current directory)',
                            metavar='DIR')
        args = parser.parse_args()
        return args


    def main(args):
        wk_dir_ = args.work_dir
        srx = args.srx
        req_len = args.seq_len
        output_dir = args.output_dir

        req_len_half = math.ceil(req_len / 2)

        if wk_dir_ is not None:
            bed_path_ = get_bed_file_path(wk_dir_, srx)
        else:
            bed_path_ = srx
        if os.path.exists(bed_path_):
            srx = os.path.splitext(os.path.basename(bed_path_))[0]
            df = pd.read_table(bed_path_, header=None)
            df.drop([x for x in range(5, df.shape[1])], axis=1, inplace=True)
            df.columns = ['chr', 'start', 'end', 'name', 'score']
            df['start'] = df['start'] + 1
            df.to_csv(os.path.join(output_dir,
                                   "{}_peak_info.txt".format(srx)),
                      sep='\t', index=False)
            df['start'] = (df['end'] + df['start'] + 1
                           ).floordiv(2) - req_len_half
            df['end'] = df['start'] + req_len - 1
            df.to_csv(os.path.join(output_dir,
                                   "{}_peak_extract.txt".format(srx)),
                      sep='\t', index=False)


    main(parse_args())
