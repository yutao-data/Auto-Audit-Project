#!/usr/bin/env python3

import pandas
import numpy
import tqdm
import logging
import os
import argparse
import random


def main():
    parser = argparse.ArgumentParser(description='Make dataset')
    parser.add_argument('--log_level', type=str, default='INFO',
                        help='Logging level')

    args = parser.parse_args()

    logging.basicConfig(level=args.log_level,
                        format='%(asctime)s %(levelname)s %(message)s')

    logging.info('读取 roa.csv')
    roa_roe_df = pandas.read_csv('roa.csv', index_col='id')

    logging.info('读取 cross-tf.csv')
    cross_tf = pandas.read_csv('cross-tf.csv', index_col='id')
    logging.info('打乱 id 列表顺序')
    cross_tf_shuffle_index = cross_tf.index.tolist()
    random.shuffle(cross_tf_shuffle_index)

    # 使用次年的ROA和ROE作为target
    def map_get_target(id, label='roa'):
        code, year = id.split('-')
        year = int(year)
        target_year = year + 1
        target_id = '%s-%d' % (code, target_year)
        if target_id not in roa_roe_df.index:
            return numpy.nan
        return roa_roe_df.loc[target_id, label]

    def map_get_target_roa(id):
        return map_get_target(id, label='roa')
    target = pandas.DataFrame()
    target['id'] = cross_tf_shuffle_index
    target.set_index('id', inplace=True)
    target['target_roa'] = target.index.map(map_get_target_roa)
    # 保存
    logging.info('保存 next-year-roa.csv')
    target.to_csv('next-year-roa.csv')
    # 删除2020年的数据，因为没有次年的数据
    target = target[~target.index.str.endswith('2020')]
    # 删除任何包括空值的行
    target.dropna(how='any', inplace=True)
    logging.info('保存 target.csv')
    target.to_csv('target.csv')

    # 生成网络输入用的词频数据
    # 去除 2020 年的数据（因为没有次年 ROE ROA 作为target）
    # 因为 cross_tf 是所有文本的词频
    # 2020年的文本无法作为输入，没有次年数据，所以需要去除
    source = pandas.DataFrame()
    source.index = target.index
    source = source.merge(cross_tf, on='id')
    logging.info('保存 source.csv')
    source.to_csv('source.csv')
    source


if __name__ == '__main__':
    main()
