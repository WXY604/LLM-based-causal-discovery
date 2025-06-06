import requests
import json
import threading
import os
from rich import print

# 数据集列表
# datasets = ['child','alarm','insurance','neuropathic','cancer', 'asia','water','mildew']
datasets = ['child','alarm']
# 原始 prompt 文件路径
path1 = 'prompt_design/prompt/'
# 存储结果的路径
path2 = 'LLM_query'

# 子文件夹处理顺序
folder_order = ['LLM_answer']

model="gpt-4o"

class myThread(threading.Thread):  # 继承父类 threading.Thread
    def __init__(self, threadID, name, dataset):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.dataset = dataset

    def run(self):
        print(f"Starting thread: {self.name} for dataset: {self.dataset}")

        # 按顺序处理每个文件夹
        for folder in folder_order:
            folder_path = os.path.join(path1, folder, self.dataset)
            result_folder_path = os.path.join(path2, folder, self.dataset)

            # 如果目标文件夹不存在，则创建
            if not os.path.exists(result_folder_path):
                os.makedirs(result_folder_path)

            # 遍历当前文件夹下的所有 txt 文件
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.txt'):
                    txt_file_path = os.path.join(folder_path, file_name)
                    result_file_path = os.path.join(
                        result_folder_path, model, file_name)

                    # 如果结果文件已存在，则跳过
                    if os.path.exists(result_file_path):
                        print(
                            f"Skipping {file_name} in {self.dataset}/{folder} (result already exists)")
                        continue
                    elif os.path.exists(os.path.join(result_folder_path, model))==False:
                        os.makedirs(os.path.join(result_folder_path, model))

                    # 读取文件中的 prompt
                    with open(txt_file_path, 'r', encoding='utf-8') as file:
                        prompt_text = file.read()

                    # 发起 API 请求
                    response = self.query_openai(
                        [{"role": "user", "content": prompt_text}])

                    # 处理 API 响应
                    if 'choices' in response:
                        content = response['choices'][0]['message']['content']
                        print(
                            f"Response received for {file_name} in {self.dataset}/{folder}")

                        # 将结果写入对应的文件
                        with open(result_file_path, 'w', encoding='utf-8') as result_file:
                            result_file.write(content)
                    else:
                        print(
                            f"Error for {file_name}: {response.get('error', 'Unknown error')}")

        print(f"Exiting thread: {self.name}")

    def query_openai(self, messages=[], api_key="", model=model):
        url = "https://xiaoai.plus/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        data = {
            "model": model,
            "messages": messages
        }

        response = requests.post(url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.text}


# 创建并启动线程
threads = []

for idx, dataset in enumerate(datasets):
    thread = myThread(threadID=idx, name=f"Thread-{dataset}", dataset=dataset)
    thread.start()
    threads.append(thread)

# 等待所有线程完成
for t in threads:
    t.join()

print("All threads completed.")
