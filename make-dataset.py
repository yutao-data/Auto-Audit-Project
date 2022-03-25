#!/usr/bin/env python

import pandas
import numpy
import tqdm
import logging
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description='Make dataset')
    parser.add_argument('--log_level', type=str, default='INFO',
                        help='Logging level')

    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format='%(asctime)s %(levelname)s %(message)s')

    logging.info('读取 firms.csv')
    firms_df = pandas.read_csv('firms.csv')

    # 股票代码
    ts_code_col = firms_df['ts_code'].drop_duplicates()

    logging.info('读取 fina_indicator.csv')
    indicator_df = pandas.read_csv('fina_indicator.csv')

    # 去除重复的股票代码
    result = []
    for ts_code in tqdm.tqdm(ts_code_col, desc='筛选年报及处理重复股票'):
        # 取出每只股票的年报，只选择ROA和ROE
        df = indicator_df[
            (indicator_df['ts_code'] == ts_code) &
            (indicator_df['end_date'].map(str).str.endswith('1231'))
        ][['ts_code', 'end_date', 'roe', 'roa']]
        if not df.all().all():
            print(df)
        result.append(df)
    logging.info('合并结果')
    roa_roe_df = pandas.concat(result, ignore_index=True)

    # 生成ID，
    def apply_to_row(s):
        return '%s-%s' % (s.ts_code[:-3], str(s.end_date)[:4])
    roa_roe_df['id'] = roa_roe_df.apply(apply_to_row, axis=1).rename('id')
    roa_roe_df.drop_duplicates(subset=['id'], keep='first', inplace=True)
    roa_roe_df = roa_roe_df.set_index('id')

    logging.info('读取 cross-tf.csv')
    cross_tf = pandas.read_csv('cross-tf.csv', index_col='id')

    # 使用次年的ROA和ROE作为target
    def map_get_target(id, label='roa'):
        code, year = id.split('-')
        year = int(year)
        target_year = year + 1
        target_id = '%s-%d' % (code, target_year)
        if not target_id in roa_roe_df.index:
            return numpy.nan
        return roa_roe_df.loc[target_id, label]
    def map_get_target_roa(id):
        return map_get_target(id, label='roa')
    def map_get_target_roe(id):
        return map_get_target(id, label='roe')
    target = pandas.DataFrame()
    target['id'] = cross_tf.index
    target.set_index('id', inplace=True)
    target['target_roe'] = target.index.map(map_get_target_roe)
    target['target_roa'] = target.index.map(map_get_target_roa)
    # 删除2020年的数据，因为没有次年的数据
    target = target[~target.index.str.endswith('2020')]
    logging.info('保存 target.csv')
    target.to_csv('target.csv')

    # 生成网络输入用的词频数据
    # 去除 2020 年的数据（因为没有次年 ROE ROA 作为target）
    # 因为 cross_tf 是所有文本的词频
    # 2020年的文本无法作为输入，没有次年数据，所以需要去除
    source = pandas.DataFrame()
    source.index=target.index
    source = source.merge(cross_tf, on='id')
    logging.info('保存 source.csv')
    source.to_csv('source.csv')
    source

if __name__ == '__main__':
    main()
