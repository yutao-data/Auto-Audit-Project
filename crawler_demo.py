import requests
import random
import time
import urllib


# path define

download_path = 'http://static.cninfo.com.cn/'
saving_path = './pdf/'


# Basic information

User_Agent = [
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Win64; x64; Trident/5.0; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 2.0.50727; Media Center PC 6.0)",
    "Mozilla/5.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 1.0.3705; .NET CLR 1.1.4322)",
    "Mozilla/4.0 (compatible; MSIE 7.0b; Windows NT 5.2; .NET CLR 1.1.4322; .NET CLR 2.0.50727; InfoPath.2; .NET CLR 3.0.04506.30)",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN) AppleWebKit/523.15 (KHTML, like Gecko, Safari/419.3) Arora/0.3 (Change: 287 c9dfb30)",
    "Mozilla/5.0 (X11; U; Linux; en-US) AppleWebKit/527+ (KHTML, like Gecko, Safari/419.3) Arora/0.6",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.2pre) Gecko/20070215 K-Ninja/2.1.1",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN; rv:1.9) Gecko/20080705 Firefox/3.0 Kapiko/3.0"
]

headers = {'Accept': 'application/json, text/javascript, */*; q=0.01',
           "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
           "Accept-Encoding": "gzip, deflate",
           "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7,zh-HK;q=0.6,zh-TW;q=0.5",
           'Host': 'www.cninfo.com.cn',
           'Origin': 'http://www.cninfo.com.cn',
           'Referer': 'http://www.cninfo.com.cn/new/commonUrl?url=disclosure/list/notice',
           'X-Requested-With': 'XMLHttpRequest'
           }


# Filter files
allowed_list = [
    '保留意见加事项段(更新后）',
    '保留意见加事项段',
    '无保留意见加事项段(更新后）',
    '无保留意见加事项段',
    '保留意见（更新后）',
    '保留意见',
    '无法发表意见（更新后）',
    '无法发表意见',
    '否定意见（更新后）',
    '否定意见'
]

block_list = [
    '监事会',
]

title = '宜华健康:监事会关于对2020年度带持续经营重大不确定性段落的无保留意见审计报告的专项说明'

for item in allowed_list:
    if item in title:
        allowed = True
        break
        
for item in block_list:
    if item in title:
        allowed = False
        break

print(allowed)



# 记录条目和文件
if allowed:
    name = i["secCode"] + '_' + i['secName'] + '_' + i['announcementTitle'] + '.pdf'
    if '*' in name:
        name = name.replace('*', '')
    saving_path = '/tmp/acctAnnocuncement/' + i["secCode"] + '/' + i["secYear"] + '/'
    file_path = saving_path + name
    
    
    
    time.sleep(random.random() * 2)

    headers['User-Agent'] = random.choice(User_Agent)
    r = requests.get(download)

    f = open(file_path, "wb")
    f.write(r.content)
    f.close()
else:
    continue