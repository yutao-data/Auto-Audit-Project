#!/usr/bin/env python3

import pandas
import numpy
import tqdm
import logging
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description='Make dataset')
    parser.add_argument('--log_level', type=str, default='INFO',
                        help='Logging level')

    args = parser.parse_args()

    logging.basicConfig(level=args.log_level,
                        format='%(asctime)s %(levelname)s %(message)s')

    logging.info('读取 firms.csv')
    firms_df = pandas.read_csv('firms.csv')

    # 股票代码
    ts_code_col = firms_df['ts_code'].drop_duplicates()

    logging.info('读取 fina_indicator.csv')
    indicator_df = pandas.read_csv('fina_indicator.csv')

    # 去除重复的股票代码
    result = []
    for ts_code in tqdm.tqdm(ts_code_col, desc='筛选年报及处理重复股票'):
        # 取出每只股票的年报，只选择ROA
        df = indicator_df[
            (indicator_df['ts_code'] == ts_code) &
            (indicator_df['end_date'].map(str).str.endswith('1231'))
        ][['ts_code', 'end_date', 'roa']]
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

    # 过滤异常 ROA
    roa_roe_df = roa_roe_df[(roa_roe_df['roa'] < 39) & (roa_roe_df['roa'] > -39)]

    logging.info('保存 roa.csv')
    roa_roe_df.to_csv('roa.csv')

if __name__ == '__main__':
    main()
