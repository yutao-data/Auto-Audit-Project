#!/usr/bin/env python3

import os
import jieba
import pandas
import numpy
import tqdm
import argparse
import logging
import math


def main():
    parser = argparse.ArgumentParser(
        description='Term Analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--txt_dir', type=str, default='./txt',
                        help='txt directory')
    parser.add_argument('--min_count', type=int, default=8,
                        help='min term count filter (default 8)')
    parser.add_argument('--show_img', action='store_true',
                        help='生成并显示词频分析图片')
    parser.add_argument('--save_img', type=str, default='',
                        help='图片保存位置，默认不保存')
    parser.add_argument('--img_dpi', type=int, default=1200,
                        help='图片分辨率')
    parser.add_argument('--log_level', type=str, default='INFO',
                        help='日志级别')
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level,
                        format='%(asctime)s %(levelname)s %(message)s')

    logging.info('开始分析')

    txt_dir = args.txt_dir
    tmp = os.listdir(txt_dir)
    txt_file_list = [os.path.join(txt_dir, i) for i in tmp]
    logging.info('TXT 文档数量：%d' % len(txt_file_list))

    # 文档 ID 列表
    id_list = [extra_id_from_filename(i) for i in tmp]

    filepath_source_dict = {}
    # Dict: id -> filepath
    for i in range(len(id_list)):
        filepath_source_dict[id_list[i]] = txt_file_list[i]

    # 字典类型，key 为 id, value 为 list 类型，list 中每个元素是一段文本
    paragraph_dict = {}
    for id in filepath_source_dict:
        paragraph_dict[id] = get_paragraphs(filepath_source_dict[id])

    # key 为 id, value 为 字典。
    # 该字典的 key 为 词语， id 为 该词语在文档内出现的 次数 。
    # 注意 无实义词不在该字典中。无实义词应该在分析最初阶段被剔除掉（？
    # Dict id -> [term frequency count dict]
    tf_dict = {}
    for id in tqdm.tqdm(paragraph_dict, desc='分词并计算 TF'):
        tf_dict[id] = {}
        words_list = [jieba.lcut(p) for p in paragraph_dict[id]]
        for words in words_list:
            for word in words:
                add(tf_dict[id], word, 1)

    # 删除标点符号和无实义词
    del_list = [
        '，',
        ',',
        '的',
        ' ',
        '。',
        '、',
        '【',
        '】',
        '！',
        '75%',
        '70%',
        'cn',
        'Ｏ',
        'com',
        'www',
        '(',
        ')',
        '"',
        '“',
        '”',
        '）',
        '（',
        '《',
        '》',
        '：',
        ':',
        '；']
    old_dict = tf_dict
    tf_dict = {}
    for id in tqdm.tqdm(old_dict, desc='删除标点符号和无实义词'):
        tf_dict[id] = {}
        for d in old_dict[id]:
            if d in del_list:
                continue
            if isNumber(d):
                continue
            # 删掉长度为 1 的词
            if len(d) == 1:
                continue
            tf_dict[id][d] = old_dict[id][d]

    # key 为 id, value 为该文档内词数总计。用于后面计算词频作为被除数
    # Dict id -> total words count
    words_count = {}
    for id in tf_dict:
        words_count[id] = sum(tf_dict[id].values())
    words_count_df = pandas.DataFrame(words_count.items(),
                                      columns=['id', 'words_count'])
    words_count_df.set_index('id', inplace=True)
    # 过滤掉词频为 0 的文档（因为部分PDF为图片，无法转成文本）
    words_count_df = words_count_df[words_count_df['words_count'] > 0]
    logging.info('保存各个文档词数统计 document-words-count.csv ，文档数量：%d' %
                 len(words_count_df))
    words_count_df.to_csv('document-words-count.csv')

    #  所有文档合并在一起的词频统计。
    # key 是 词语，value 是该词语在所有文档内出现次数加总。
    # Dict term -> frequency count
    total_tf_dict = {}
    for id in tqdm.tqdm(tf_dict, desc='统计所有文档合并在一起的词频'):
        for term in tf_dict[id]:
            add(total_tf_dict, term, 1)

    # 上述 dataframe 版本
    # save total term frequency
    sort = sorted(total_tf_dict.items(),
                  key=lambda item: item[1], reverse=True)
    total_tf_df = pandas.DataFrame(sort, columns=['term', 'count'])
    total_tf_df.index.rename('index', inplace=True)
    logging.info('保存所有文档合并在一起的词频统计结果 total-term-count.csv')
    total_tf_df.to_csv('total-term-count.csv')

    # 筛选掉词频过低的词语
    selected_tf_df = total_tf_df[total_tf_df['count'] > args.min_count]

    # 生成词向量表 可能需要一段时间
    # 对于每个词语，返回在所有文本中出现次数的情况
    cross_tf_df = pandas.DataFrame(
        columns=['id', *selected_tf_df['term'].to_list()])
    cross_tf_df['id'] = id_list
    cross_tf_df.set_index('id', drop=True, inplace=True)

    def map_to_cross_tf(term: str):
        result = []
        for id in cross_tf_df.index:
            count = tf_dict[id].get(term)
            total = words_count.get(id)
            if count and total:
                frequency = count / total
            else:
                if count and not total:
                    print('Wrong', count, total)
                frequency = numpy.nan
            result.append(frequency)
        return result
    for term in tqdm.tqdm(
            cross_tf_df,
            total=len(
                cross_tf_df.keys()),
            desc='计算文档词频'):
        cross_tf_df[term] = map_to_cross_tf(term)
    # 删除均为nan的行，并用 0 填充缺失值
    cross_tf_df = cross_tf_df.dropna(how='all').fillna(0)

    # 计算 tf-idf
    idf = (len(cross_tf_df) / (cross_tf_df > 0).sum()).map(math.log10)
    cross_tf_df = cross_tf_df * idf

    logging.info('保存词向量表 cross-tf.csv')
    cross_tf_df.to_csv('cross-tf.csv')

    if args.show_img:
        logging.info('载入 plt')
        import matplotlib.pyplot as plt
        # 画出词频分布图
        logging.info('画出词频分布图')
        im = plt.imshow((cross_tf_df > 0).iloc[:, :])
        plt.show()
        if args.save_img:
            logging.info('保存词频分布图')
            plt.savefig('cross-tf.png', dpi=args.img_dpi)


def add(dic: dict[str], word: str, num: int):
    if dic.get(word) is None:
        dic[word] = num
    else:
        dic[word] += num


def isNumber(s: str) -> bool:
    try:
        int(s)
        return True
    except BaseException:
        try:
            float(s)
            return True
        except BaseException:
            return False


def get_paragraphs(filename: str) -> list[str]:
    with open(filename, 'r') as f:
        raw = f.read()

    # 单个文件的段落，段落以至少两个换行为分割
    paragraphs = raw.split('\n\n')

    # 去掉每个段落里的换行
    ret = [p.replace('\n', '').replace('\x0c', '') for p in paragraphs]

    # 删掉列表里所有空字符串
    while '' in ret:
        ret.remove('')

    return ret


def extra_id_from_filename(filename: str) -> str:
    if filename.endswith('.txt'):
        return filename[:-4]


if __name__ == '__main__':
    main()
