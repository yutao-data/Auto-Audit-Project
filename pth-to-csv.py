#!/usr/bin/env python3

import argparse
import logging
import torch
import pandas


def main():
    parser = argparse.ArgumentParser(
        description='将模型参数转换为csv文件', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str,
                        default='model.pth', help='模型参数文件')
    parser.add_argument('--csv', type=str,
                        default='model.csv', help='输出的csv文件')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logging.info(f'加载模型参数文件 {args.model}')
    data = torch.load(args.model, map_location='cpu')

    source = pandas.read_csv('source.csv', index_col='id')

    df = pandas.DataFrame()
    df['词语'] = source.keys().tolist()
    df['积极权重'] = data['fc1.weight'][0]
    df['消极权重'] = data['fc1.weight'][1]

    df.set_index('词语', inplace=True)

    total = pandas.read_csv('total-term-count.csv', index_col='term')
    total.index.rename('词语', inplace=True)
    total = total[['count']]

    df = df.merge(total, on='词语')

    logging.info(f'保存到 {args.csv}')
    df.to_csv(args.csv)


if __name__ == '__main__':
    main()
