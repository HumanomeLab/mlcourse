#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import re
import collections
import pickle
import pandas as pd
import numpy as np

BEGIN_POS = 1
REFSEQ_P = re.compile(r'^(chr[0-9MXY]+)\.fa$')
CHR_NUM_P = re.compile(r'^chr([0-9]+)$')
REP_NSEQ_P = re.compile(r'N+')

SeqLoc = collections.namedtuple('SeqLoc', ['idx', 'chr', 'start', 'end'])


def get_chr_sort_order(ref_dir):
    """ get sorted chromosome list.

    :param ref_dir: the directory.
                    the fasta file is contained in this directory.
    :type ref_dir: str
    :return: sorted chromosome list.
    :rtype: list of str
    """
    all_nums = []
    for file in os.listdir(ref_dir):
        m = REFSEQ_P.search(file)
        if m:
            m = CHR_NUM_P.search(m.group(1))
            if m:
                id_ = m.group(1)
                all_nums.append(int(id_))
    all_nums.sort()
    all_chrs = []
    for n in all_nums:
        all_chrs.append('chr{}'.format(n))
    all_chrs.extend(['chrX', 'chrY', 'chrM'])

    return all_chrs


def extract_ranges_of_nseq_from_one(fasta_file, nmax=0):
    """ extract ranges of repeat N sequences from fasta file.

    :param fasta_file: fasta file path
    :type fasta_file: str
    :param nmax: permit max repeat count
    :type nmax: int
    :return: ranges data of repeat N sequences
    :rtype: pandas.DataFrame
    """
    df = pd.DataFrame({'a': [0] * 50, 'b': 0})
    df.columns = ['start', 'end']

    with open(fasta_file) as fin:
        fin.readline()
        line = fin.readline()
        line_len = len(line) - 1
        start = BEGIN_POS - 1
        end = BEGIN_POS - 1
        last_line = line
        count = 0
        while line:
            if line.find('N') >= 0:
                if line[-1] == '\n':
                    line = line[:-1]
                s = REP_NSEQ_P.search(line)
                if s and start >= BEGIN_POS and s.start() > 0:
                    if end - start + 1 > nmax:
                        df.loc[count] = [start, end]
                        count += 1
                    start = BEGIN_POS - 1
                off = 0
                while s:
                    if start < BEGIN_POS:
                        start = end + 1 + off + s.start()
                    off += s.end()
                    if off < line_len:
                        if end + off - start + 1 > nmax:
                            df.loc[count] = [start, end + off]
                            count += 1
                        start = BEGIN_POS - 1
                    else:
                        break
                    s = REP_NSEQ_P.search(line[off:])
            elif start >= BEGIN_POS:
                if end - start + 1 > nmax:
                    df.loc[count] = [start, end]
                    count += 1
                start = BEGIN_POS - 1
            end += line_len
            last_line = line
            line = fin.readline()
        if start >= BEGIN_POS:
            if end - start + 1 > nmax:
                df.loc[count] = [start, end]
                count += 1
    total = end - line_len + len(last_line) + 1
    if last_line[-1] == '\n':
        total -= 1
    df.loc[count] = [total, -1]
    return df.head(count + 1)


def nseq_dict_dump(nseq_dict, seq_dir, nmax):
    """

    :param nseq_dict:
    :type nseq_dict: dict of (str, pandas.DataFrame)
    :param seq_dir: the directory.
                    the fasta file is contained in this directory.
    :type seq_dir: str
    :param nmax: permit max repeat count
    :type nmax: int
    :return: None
    """
    dump_path = os.path.join(seq_dir, "nseq_dict_{}.pickle".format(nmax))
    with open(dump_path, 'wb') as fout:
        pickle.dump(nseq_dict, fout)


def nseq_dict_load(seq_dir, nmax):
    """

    :param seq_dir: the directory.
                    the fasta file is contained in this directory.
    :type seq_dir: str
    :param nmax: permit max repeat count
    :type nmax: int
    :return:
    :rtype: dict of (str, pandas.DataFrame)
    """
    dump_path = os.path.join(seq_dir, "nseq_dict_{}.pickle".format(nmax))
    if os.path.exists(dump_path):
        try:
            with open(dump_path, 'rb') as fin:
                nseq_dict = pickle.load(fin)
            nseq_dict['st_mtime'] = os.stat(dump_path).st_mtime
            return nseq_dict
        except OSError:
            pass
    return {}


def extract_ranges_of_nseq_from(seq_dir, nmax=0, log_=None):
    """  extract ranges of repeat N sequences from fasta files.

    :param seq_dir: the directory.
                    the fasta file is contained in this directory.
    :type seq_dir: str
    :param nmax: permit max repeat count
    :type nmax: int
    :param log_: callback function for progress
    :type log_: Callable[[str], None]
    :return:
    :rtype: dict of (str, pandas.DataFrame)
    """
    nseq_dict = nseq_dict_load(seq_dir, nmax)
    nseq_dict_time = -1
    if 'st_mtime' in nseq_dict:
        nseq_dict_time = nseq_dict['st_mtime']
    modified = False
    for file in os.listdir(seq_dir):
        m = REFSEQ_P.search(file)
        if m:
            if m.group(1) in nseq_dict:
                m_time = os.stat(os.path.join(seq_dir, file)).st_mtime
                if m_time < nseq_dict_time:
                    continue
            if log_ is not None:
                log_(m.group(1))
            ranges = extract_ranges_of_nseq_from_one(
                os.path.join(seq_dir, file), nmax)
            nseq_dict[m.group(1)] = ranges
            modified = True
            if log_ is not None:
                log_("")
    if modified:
        nseq_dict_dump(nseq_dict, seq_dir, nmax)

    return nseq_dict


def add_ignore_regions_from_df(nseq_dict, list_df):
    """ add tss_regions to ranges data.

    :param nseq_dict:
    :type nseq_dict: dict of (str, pandas.DataFrame)
    :param list_df: ignore sequence position list.
    :type list_df: pandas.DataFrame
    :rtype: None
    """
    for chr_, v in list_df.groupby('chr'):
        if chr_ in nseq_dict:
            df = nseq_dict[chr_]
            v = v.copy()
            pos_from = v['start']
            pos_end = v['end']
            add_df = pd.DataFrame({'a': pos_from, 'b': pos_end})
            add_df.columns = ['start', 'end']
            df = pd.concat([df, add_df],
                           ignore_index=True)  # type: pd.DataFrame
            df.sort_values(by=['start', 'end'], ascending=[True, False],
                           inplace=True)
            df.reset_index(drop=True, inplace=True)
            last_index = df[df['end'] == -1].index[0]
            seq_len = df.loc[last_index].start - 1
            if df.shape[0] > last_index + 1:
                drop_list = [x for x in range(last_index + 1, df.shape[0])]
            else:
                drop_list = []
            df[df['end'] > seq_len] = seq_len
            df_start = df['start']
            df_end = df['end']
            i = 1
            while i < last_index:
                while i < last_index and df_end[i - 1] + 1 < df_start[i]:
                    i += 1
                if i < last_index:
                    t = i - 1
                    df_end_t = df_end[t]
                    while i < last_index and df_end_t + 1 >= df_start[i]:
                        if df_end_t < df_end[i]:
                            df_end_t = df_end[i]
                        drop_list.append(i)
                        i += 1
                    df_end[t] = df_end_t
            if len(drop_list) > 0:
                df.drop(drop_list, inplace=True)
                df.reset_index(drop=True, inplace=True)
            nseq_dict[chr_] = df


def convert_ranges_to_tbl_for_random_selection(nseq_dict, req_seq_len):
    """ convert ranges data to random selecton table.

    :param nseq_dict:
    :type nseq_dict: dict of (str, pandas.DataFrame)
    :param req_seq_len:
    :type req_seq_len: int
    :return:
    :rtype: pandas.DataFrame
    """
    df = pd.DataFrame(columns=['a', 'b', 'c'])
    df['a'] = df['a'].astype(int)
    df['c'] = df['c'].astype(int)

    index_ = 0
    for chr_, ranges in nseq_dict.items():
        if isinstance(ranges, pd.DataFrame):
            starts = ranges['start'].values
            ends = np.roll(ranges['end'].values, 1)
            ends[0] = BEGIN_POS - 1
            seq_len = starts - ends - 1
            valid_lens = seq_len >= req_seq_len
            ends = ends[valid_lens]
            seq_len = seq_len[valid_lens]
            indexs = np.roll(np.cumsum(seq_len - req_seq_len + 1) + index_, 1)
            next_index = indexs[0]
            indexs[0] = index_
            df1 = pd.DataFrame({'a': indexs, 'b': chr_, 'c': ends + 1})
            df = pd.concat([df, df1], ignore_index=True)  # type: pd.DataFrame
            index_ = next_index
    df.loc[df.shape[0]] = [index_, '', req_seq_len]
    df.columns = ['idx', 'chr', 'start']

    return df


def iter_get_seq_from_random_selection_tbl(tbl, req_count):
    """ iterator of random selected sequence information.

    :param tbl:
    :type tbl: pandas.DataFrame
    :param req_count:
    :type req_count: int
    :rtype: collections.Generator[SeqLoc]
    """
    # tbl = tbl.copy()
    tbl['f'] = -1

    req_seq_len = tbl.iloc[-1].start

    n = 0
    nn = 0
    count = 0
    while req_count > 0:
        total = tbl.iloc[-1].idx
        rand_idx = np.random.randint(0, total)
        tbl_i = tbl.idx.searchsorted(rand_idx, side='right')[0] - 1
        if tbl.loc[tbl_i, 'f'] >= 0:
            nn += 1
            if nn < 10:
                continue
            for idx in reversed(tbl[tbl['f'] >= 0].index):
                tbl_row = tbl.iloc[idx]
                rand_idx = tbl_row.f
                chr_ = tbl_row.chr
                start = tbl_row.start + (rand_idx - tbl_row.idx)
                end = start + req_seq_len - 1
                idx_start = tbl_row.idx
                idx_end = tbl.iloc[idx + 1].idx
                block_len = idx_end - idx_start

                len1 = rand_idx - req_seq_len - idx_start + 1
                len2 = idx_end - (rand_idx + req_seq_len)
                if len1 >= req_seq_len:
                    if len2 > 0:
                        remove_size = block_len - len1 - len2
                        tbl.loc[idx + 0.5] = [
                            idx_start + len1 + remove_size, chr_, end + 1,
                            -1]
                    else:
                        remove_size = block_len - \
                                      (start - req_seq_len -
                                       tbl_row.start + 1)
                elif len2 > 0:
                    remove_size = end - tbl_row.start + 1
                    tbl.loc[idx] = [idx_start, chr_, end + 1, -1]
                else:
                    tbl.drop(idx, inplace=True)
                    if tbl.shape[0] <= 1:
                        return
                    idx -= 1
                    remove_size = block_len
                tbl['idx'] -= np.array([0, remove_size]).repeat(
                    [idx + 1, tbl.shape[0] - idx - 1])
            tbl.sort_index(inplace=True)
            tbl.reset_index(drop=True, inplace=True)
            tbl['f'] = -1
            n += 1
            nn = 0
            continue
        tbl_row = tbl.iloc[tbl_i]
        chr_ = tbl_row.chr
        start = tbl_row.start + (rand_idx - tbl_row.idx)
        end = start + req_seq_len - 1
        yield SeqLoc(count, chr_, start, end)
        count += 1
        tbl.loc[tbl_i, 'f'] = rand_idx
        req_count -= 1


def iter_random_selection_of_refseq_without_df(refseq_dir, nmax,
                                               ignore_list_df, req_seq_len,
                                               req_count, progress_out):
    """

    :param refseq_dir: the directory.
                       the fasta file is contained in this directory.
    :type refseq_dir: str
    :param nmax: permit max repeat count
    :type nmax: int
    :param ignore_list_df:
    :type ignore_list_df: pandas.DataFrame
    :param progress_out:
    :type progress_out: io.TextIOBase
    :param req_seq_len:
    :type req_seq_len: int
    :param req_count:
    :type req_count: int
    :rtype: collections.Generator[SeqLoc]
    """

    def __null_out(x=None):
        pass

    progress_write = __null_out
    progress_flush = __null_out
    if progress_out is not None:
        progress_write = progress_out.write
        progress_flush = progress_out.flush

        def progress_func(x):
            if x != '':
                progress_write('.')
                progress_flush()
    else:
        progress_func = None

    progress_write('refseq loading ')
    progress_flush()
    nseq_dict = extract_ranges_of_nseq_from(refseq_dir, nmax, progress_func)
    if ignore_list_df is not None:
        progress_write('\nmerge ignore list ...')
        progress_flush()
        add_ignore_regions_from_df(nseq_dict, ignore_list_df)
    progress_write('\nmake random selection table ...')
    progress_flush()
    selection_tbl = convert_ranges_to_tbl_for_random_selection(
        nseq_dict, req_seq_len)
    progress_write('\ndo random selecting ')
    progress_flush()
    progress = 0
    for selected in iter_get_seq_from_random_selection_tbl(
            selection_tbl, req_count):
        if selected.idx >= req_count * progress / 10:
            progress += 1
            progress_write('.')
            progress_flush()
        yield selected
    progress_write('\n')
    progress_flush()


def random_selection_of_refseq_without_df_sorted(refseq_dir, nmax,
                                                 ignore_list_df, req_seq_len,
                                                 req_count, progress_out):
    """

    :param refseq_dir: the directory.
                       the fasta file is contained in this directory.
    :type refseq_dir: str
    :param nmax: permit max repeat count
    :type nmax: int
    :param ignore_list_df:
    :type ignore_list_df: pandas.DataFrame
    :param progress_out:
    :type progress_out: io.TextIOBase
    :param req_seq_len:
    :type req_seq_len: int
    :param req_count:
    :type req_count: int
    :rtype: pandas.DataFrame
    """
    chr_order = get_chr_sort_order(refseq_dir)
    df = pd.DataFrame({'a': 'chr', 'b': [0] * req_count, 'c': 0})
    df.columns = ['chr', 'start', 'end']
    i = 0
    for seq in iter_random_selection_of_refseq_without_df(
            refseq_dir, nmax, ignore_list_df,
            req_seq_len, req_count, progress_out):
        df.loc[i] = [seq.chr, seq.start, seq.end]
        i += 1
    if i < req_count:
        df = df.iloc[range(i), ]
    df["chr"] = pd.Categorical(df['chr'], chr_order)
    df.sort_values(by=['chr', 'start', 'end'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


if __name__ == "__main__":
    import argparse


    def parse_args():
        parser = argparse.ArgumentParser(
            description='get random sequence locations from refseq...')
        parser.add_argument('-v', '--version',
                            action='version',
                            version='%(prog)s 1.0.0')
        parser.add_argument('fasta_dir',
                            action='store',
                            nargs=None,
                            const=None,
                            default=None,
                            type=str,
                            choices=None,
                            help='reference fasta directory.',
                            metavar=None)
        parser.add_argument('seq_len',
                            action='store',
                            nargs=None,
                            const=None,
                            default=None,
                            type=int,
                            choices=None,
                            help='subsequence length.',
                            metavar=None)
        parser.add_argument('seq_count',
                            action='store',
                            nargs=None,
                            const=None,
                            default=None,
                            type=int,
                            choices=None,
                            help='sequences count.',
                            metavar=None)
        parser.add_argument('-l', '--ignore-pos-list',
                            action='store',
                            nargs=None,
                            const=None,
                            default=None,
                            type=str,
                            choices=None,
                            help='ignore-pos list file.',
                            metavar=None)
        parser.add_argument('-a', '--from-annotation-track',
                            action='store_true',
                            help='annotation track type position info.')
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
        parser.add_argument('--nmax',
                            action='store',
                            nargs=None,
                            const=None,
                            default=50,
                            type=int,
                            choices=None,
                            help='permit max N seq repeat count.'
                                 '(bp, default:50)',
                            metavar='INT')
        parser.add_argument('-o', '--output',
                            action='store',
                            nargs=None,
                            const=None,
                            default=None,
                            type=str,
                            choices=None,
                            help='result file or directory.',
                            metavar='FILE')
        args = parser.parse_args()
        return args


    def main(args):
        if args.ignore_pos_list is not None:
            ignore_df = pd.read_table(args.ignore_pos_list,
                                      header=None
                                      if args.header < 0 else args.header)
            if args.header_def is not None:
                cols = ignore_df.columns
                new_cols = []
                i = 0
                for col in args.header_def.split(','):
                    col = col.strip()
                    if col != '':
                        new_cols.append(col)
                    else:
                        new_cols.append(cols[i])
                    i += 1
                while i < ignore_df.shape[1]:
                    new_cols.append(cols[i])
                    i += 1
                ignore_df.columns = new_cols
        else:
            ignore_df = None

        res = random_selection_of_refseq_without_df_sorted(
            args.fasta_dir,
            args.nmax,
            ignore_df, args.seq_len, args.seq_count, sys.stderr)

        if args.output is not None:
            fout = open(args.output, "w")
        else:
            fout = sys.stdout

        try:
            fout.write('chr\tstart\tend\n')
            for i, seq_loc in res.iterrows():
                fout.write('{}\t{}\t{}\n'.format(seq_loc.chr,
                                                 seq_loc.start, seq_loc.end))
        finally:
            if args.output is not None and fout is not None:
                fout.close()

    main(parse_args())
