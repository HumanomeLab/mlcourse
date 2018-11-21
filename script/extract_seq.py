#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# extract sub-sequences by seq-pos listed file
#

import sys
import os
import io
import math
import collections
import pandas as pd

CHR_MAP = {
    '1': 'chr1', '2': 'chr2', '3': 'chr3', '4': 'chr4', '5': 'chr5',
    '6': 'chr6', '7': 'chr7', '8': 'chr8', '9': 'chr9', '10': 'chr10',
    '11': 'chr11', '12': 'chr12', '13': 'chr13', '14': 'chr14',
    '15': 'chr15', '16': 'chr16', '17': 'chr17', '18': 'chr18',
    '19': 'chr19', '20': 'chr20', '21': 'chr21', '22': 'chr22',
    '23': 'chrX', '24': 'chrY', '25': 'chrXY', '26': 'chrM',
    'X': 'chrX', 'x': 'chrX', 'Y': 'chrY', 'y': 'chrY',
    'XY': 'chrXY', 'xy': 'chrXY', 'MT': 'chrM', 'mt': 'chrM',
    'chr1': 'chr1', 'chr2': 'chr2', 'chr3': 'chr3', 'chr4': 'chr4',
    'chr5': 'chr5', 'chr6': 'chr6', 'chr7': 'chr7', 'chr8': 'chr8',
    'chr9': 'chr9', 'chr10': 'chr10', 'chr11': 'chr11', 'chr12': 'chr12',
    'chr13': 'chr13', 'chr14': 'chr14', 'chr15': 'chr15', 'chr16': 'chr16',
    'chr17': 'chr17', 'chr18': 'chr18', 'chr19': 'chr19', 'chr20': 'chr20',
    'chr21': 'chr21', 'chr22': 'chr22',
    'chrX': 'chrX', 'chrY': 'chrY', 'chrXY': 'chrXY',
    'chrM': 'chrM', 'chrMT': 'chrM',
}

SeqLoc = collections.namedtuple('SeqLoc', ['idx', 'chr', 'start', 'end'])
SeqText = collections.namedtuple('SeqText', ['loc', 'text'])


def _conv_chr_id(chr_):
    """ convert chromosome id

    :param chr_: chromosome id convert from
    :type chr_: str
    :return: converted chromosome id
    :rtype: str
    """
    if chr_ in CHR_MAP:
        return CHR_MAP[chr_]
    else:
        return None


def get_subsequence(seq_dir=None):
    """ co-routine of sub-sequence text

    :param seq_dir: fasta file directory
    :type seq_dir: str
    :return: co-routine[seq-text, (chr, start, end)]
    :rtype: collections.Generator[str, (str, int, int)]
    """
    def __conv_pos(x):
        """ convert sequence position to file position.

        :param x: sequence position.
        :type x: int
        :return: file seek position.
        :rtype: int
        """
        return int(math.ceil(
            float(x) * (1.0 + 1.0 / line_len) + head_len - 2))

    seq_text = None
    seq_fin = None
    last_fasta = None
    while True:
        seq_pos = (yield seq_text)
        fasta = seq_pos[0]
        if fasta != last_fasta and seq_fin is not None:
            seq_fin.close()
            seq_fin = None
        if seq_fin is None:
            if seq_dir is None:
                fasta_path = fasta
            else:
                fasta_path = os.path.join(seq_dir, fasta)
            if not os.path.exists(fasta_path):
                fasta_path = fasta_path + ".fa"
            if os.path.exists(fasta_path):
                with open(fasta_path, 'r') as fin:
                    head_len = len(fin.readline())
                    line_len = len(fin.readline()) - 1
                seq_fin = open(fasta_path, 'rb')
                last_fasta = fasta
        if seq_fin is not None:
            pos_from = __conv_pos(seq_pos[1])
            pos_end = __conv_pos(seq_pos[2])
            seq_fin.seek(pos_from)
            seq_data = seq_fin.read(pos_end - pos_from + 1).decode()
            seq_text = seq_data.replace('\n', '')
        else:
            seq_text = ''


def iter_extract_sequences(seq_dir, iter_seq_pos, from_annotation_track=False):
    """ extract ref sequences iterator.

    :param seq_dir: the directory.
                    the fasta file is contained in this directory.
    :type seq_dir: str
    :param iter_seq_pos:
    :type iter_seq_pos: collections.Generator[SeqPos]
    :param from_annotation_track:
    :type from_annotation_track: bool
    :return: generator[SeqText]
    :rtype: collections.Generator[SeqText]
    """
    get_seq = get_subsequence(seq_dir)
    get_seq.send(None)
    for seq_loc in iter_seq_pos:
        chr_ = _conv_chr_id(seq_loc.chr)
        if chr_ is not None:
            if from_annotation_track:
                start = seq_loc.start + 1
            else:
                start = seq_loc.start
            end = seq_loc.end
            seq_text = get_seq.send((chr_, start, end))
            yield SeqText(SeqLoc(seq_loc.idx, chr_, start, end), seq_text)


def extract_sequences_df(seq_loc_list_df, seq_dir, name_column=None,
                         out=sys.stdout, from_annotation_track=False):
    """ extract sub-sequences by DataFrame.

    :param seq_loc_list_df: sub-sequence location info list.
    :type seq_loc_list_df: pandas.DataFrame
    :param seq_dir: the directory.
                    the fasta file is contained in this directory.
    :type seq_dir: str
    :param name_column: id column title.
    :type name_column: str
    :param out: result output.
    :type out: io.TextIOBase
    :param from_annotation_track: True is sub-sequence
                                  annotation track type position.
    :type from_annotation_track: bool
    :return: None
    """

    def _iter_df(df_, idx):
        """

        :param df_:
        :type df_: pandas.DataFrame
        :param idx:
        :type idx: str
        :return:
        :rtype: collections.Generator[SeqLoc]
        """
        if idx is None:
            for key, row in df_.iterrows():
                yield SeqLoc(key, row['chr'], row['start'], row['end'])
        else:
            for key, row in df_.iterrows():
                yield SeqLoc(row[idx], row['chr'], row['start'], row['end'])
        pass

    out.write('id\tchr\tstart\tend\tseq\n')
    for seq_text in iter_extract_sequences(seq_dir, _iter_df(seq_loc_list_df,
                                                             name_column),
                                           from_annotation_track):
        seq_loc = seq_text.loc
        out.write('{}\t{}\t{}\t{}\t{}\n'.format(seq_loc.idx, seq_loc.chr,
                                                seq_loc.start, seq_loc.end,
                                                seq_text.text))


def extract_sequences(seq_loc_list_file, seq_dir, name_column=None,
                      out=sys.stdout, from_annotation_track=False):
    """ extract sub-sequences by seq_pos_list file.

    :param seq_loc_list_file: seq-pos list file.
    :type seq_loc_list_file: str
    :param seq_dir: the directory.
                    the fasta file is contained in this directory.
    :type seq_dir: str
    :param name_column: id column title.
    :type name_column: str
    :param out: result output.
    :type out: io.TextIOBase
    :param from_annotation_track: True is sub-sequence
                                  annotation track type position.
    :type from_annotation_track: bool
    :return: None
    """
    list_df = pd.read_table(seq_loc_list_file)
    extract_sequences_df(list_df, seq_dir, name_column, out,
                         from_annotation_track)


if __name__ == "__main__":
    import argparse


    def parse_args():
        parser = argparse.ArgumentParser(
            description='extract sub-sequences by seq-pos listed file...')
        parser.add_argument('-v', '--version',
                            action='version',
                            version='%(prog)s 1.0.0')
        parser.add_argument('seq_pos_list',
                            action='store',
                            nargs=None,
                            const=None,
                            default=None,
                            type=str,
                            choices=None,
                            help='seq-pos list file.(tab delimited text file)',
                            metavar=None)
        parser.add_argument('fasta_dir',
                            action='store',
                            nargs=None,
                            const=None,
                            default=None,
                            type=str,
                            choices=None,
                            help='reference fasta directory.',
                            metavar=None)
        parser.add_argument('--from-annotation-track',
                            action='store_true',
                            help='seq_pos_list is contained'
                                 ' annotation track type position info.')
        parser.add_argument('--header',
                            action='store',
                            nargs=None,
                            const=None,
                            default=0,
                            type=int,
                            choices=None,
                            help='row number of column header''s row',
                            metavar='POS')
        parser.add_argument('--header-def',
                            action='store',
                            nargs=None,
                            const=None,
                            default=None,
                            type=str,
                            choices=None,
                            help='columns header definitions.'
                                 '(comma separated string)',
                            metavar='DEF')
        parser.add_argument('--id-column',
                            action='store',
                            nargs=None,
                            const=None,
                            default=None,
                            type=str,
                            choices=None,
                            help='id column title string.',
                            metavar='STR')
        args = parser.parse_args()
        return args


    def main(args):
        df = pd.read_table(args.seq_pos_list,
                           header=None if args.header < 0 else args.header)
        if args.header_def is not None:
            cols = df.columns
            new_cols = []
            i = 0
            for col in args.header_def.split(','):
                col = col.strip()
                if col != '':
                    new_cols.append(col)
                else:
                    new_cols.append(cols[i])
                i += 1
            while i < df.shape[1]:
                new_cols.append(cols[i])
                i += 1
            df.columns = new_cols

        extract_sequences_df(df, args.fasta_dir, args.id_column,
                             from_annotation_track=args.from_annotation_track)

    main(parse_args())
