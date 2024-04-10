import re

import requests
from bs4 import BeautifulSoup


def save_text_from_web():
    # 指定目标网页的URL
    web_urls_path = "/target/target_spark/information/urls.txt"
    urls = []
    with open(web_urls_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        urls.append(line)
    if not urls:
        print("no urls")
    save_path = "/target/target_spark/information/"
    # 发送HTTP请求以获取页面内容
    index = 0
    for url in urls:
        index = index+1
        response = requests.get(url)

        # 检查请求是否成功
        if response.status_code == 200:
            # 使用BeautifulSoup解析HTML内容
            soup = BeautifulSoup(response.text, 'html.parser')
            # 查找网页上的文字内容
            text_content = soup.get_text()
            # with open("../doc/text.txt", "r") as f:
            #     text_content = f.read()
            text_content = re.sub(r'\n+', '\n', text_content).strip()

            #text_content = text_content.replace(" ", "")
            # 打印或处理文本内容
            doc_path = save_path+"doc"+str(index)+".txt"
            with open(doc_path, "w") as f:
                f.write(text_content)
            print("doc save succese")
        else:
            print('无法访问网页。')


def get_relevant_sections(doc_path):
    with open(doc_path, "r") as f:
        text = f.read()
    relevant_sections = []
    param_reg = r'[a-z_]+_[a-z]+'

    sections = text.split('\n')
    key = 0
    for section in sections:
        key = key + 1
        if re.findall(param_reg, section):
            relevant_sections.append(section)
            # print(str(key)+str(section))
    return relevant_sections

if __name__ == '__main__':
    save_text_from_web()
