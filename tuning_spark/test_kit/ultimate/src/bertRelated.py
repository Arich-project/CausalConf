from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
import re
import torch
import openpyxl
import pandas as pd
import openai

qa_model = "deepset/roberta-base-squad2"
use_gpt_or_bert = "gpt"


def send_to_bert(desciption):
    classifier = pipeline("zero-shot-classification",
                          model="facebook/bart-large-mnli")

    sequence_to_classify = str(desciption)
    candidate_labels = ['strongly related to performance like latency', 'not strongly related to '
                                                                        'performance like latency']
    result = classifier(sequence_to_classify, candidate_labels)
    # print(result)
    return result['scores']


def why_related_gpt(param_name, param_description):
    openai.api_key = "cd4b0343e6836a2a584c75fd80b75a85"
    openai.api_base = "http://flag.smarttrot.com/index.php/api/v1"
    prompt = "according to the text and your knowledge, why " + param_name + "affects performance ?Please " \
                                                                            "provide a concise " \
                                                                            "response. The text is " + param_name + ":" + param_description
    print(prompt)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    text = response['choices'][0]['message']['content']
    return text


def why_related_bert(question, context):
    nlp = pipeline('question-answering', model=qa_model, tokenizer=qa_model)
    QA_input = {
        'question': question,
        'context': context
    }
    res = nlp(QA_input)
    return res


def divide_sourcetxt(text):
    # 分成一段一段
    pattern = r'(\d+\.\d+\.\d+)\s+([\s\S]*?)\s+(\d+\.\d+\.\d+)'
    matches = re.findall(pattern, text)

    param_names = []
    param_descriptions = []
    whether_performance_related = []
    why_related = []
    data = {
        'param_names': param_names,
        'param_description': param_descriptions,
        'whether_performance_related': whether_performance_related,
        'why_related': why_related
    }

    for index, match1 in enumerate(matches):
        print(str(index) + " is gonging ")
        version1, content, version2 = match1

        pattern = r'^(.*?)\n.*?\n(.*?)$'
        match2 = re.search(pattern, content, re.DOTALL)

        if match2:
            param_name = match2.group(1)
            param_description = match2.group(2)
            param_names.append(param_name)
            param_descriptions.append(param_description)

            # 通过描述获得是否相关
            result = send_to_bert(param_description)
            related_score = result[0]
            whether_performance_related.append(related_score)

            # 如果相关获得原因
            if related_score > 0.55:
                if use_gpt_or_bert == "bert":
                    question = param_name + ' affect performance according to what ?'
                    context = param_name + ":" + param_description
                    res = why_related_bert(question, context)
                    why_related.append(res['answer'])
                else:
                    res = why_related_gpt(param_name, param_description)
                    why_related.append(res)
            else:
                why_related.append(None)

    df = pd.DataFrame(data)
    excel_file = 'params.xlsx'
    df.to_excel(excel_file, index=False)


def read(file_path):
    df = pd.read_excel(file_path)


if __name__ == '__main__':
    sourceWebTextSrc = r'/home/hmj/tuning_spark/target/target_spark/information/doc1.txt'
    with open(sourceWebTextSrc, "r") as f:
        sourceText = f.read()
    divide_sourcetxt(str(sourceText))
    # param_name = "spark.driver.memory"
    # des = """
    # Amount of memory to use for the driver process, i.e. where SparkContext is initialized. (e.g. 1g, 2g).
    # Note: In client mode, this config must not be set through the SparkConf directly in your application, because the driver JVM has already started at that point. Instead, please set this through the --driver-memory command line option or in your default properties file.
    # """
    # result = why_related_gpt(param_name, des)
    # print(result)
