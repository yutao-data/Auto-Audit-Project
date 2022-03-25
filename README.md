# NLP For Accounting

## 目录结构

- `firms.csv` 报告列表

- `pdf` PDF 文件目录

- `convert-pdf-to-txt' 转换 PDF 到 TXT

- `txt` TXT 文件目录，由 `convert-pdf-to-txt` 和 `pdf` 构建

- `term-analysis.py` 词频分析，生成词频 `cross-tf.py` 和 总词频统计 `total-term-count.py`

- `make-dataset` 使用从财务数据 `fina_indicator.csv` 中提取次年 ROA ROE 作为 target。生成 `target.csv` `source.csv` 作为训练的 source 和 target

- `train.py` 训练 生成 model.pth 模型参数文件

- `pth-to-csv.py` 将 pth 文件转换为人类可读的 csv 格式

## Data source

<http://www.cninfo.com.cn/>

![image-target](images/target.png.jpg)

包含"*董事会 审计意见 专项说明*"关键词的搜索结果
