import concurrent.futures
from zhipuai import ZhipuAI


class LLM_ZhiPuAI:
    api_key = "7f890ef50cfd893e7f193139c0ed2c9d.ajeSn5gOPT10H0L7"
    model_name = "glm-4-0520"

    def __init__(self, temperature=0.9, top_p=0.7, max_tokens=4095,stop=[]):
        self.client = ZhipuAI(api_key=self.api_key)  # 请填写您自己的APIKey
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stop=stop


    # 最简单的请求方法，输入prompt，返回大模型输出
    def __call__(self, prompt: str) -> str:
        response = self.client.chat.completions.create(  # 同步调用 调用后即可一次性获得最终结果
            model=self.model_name,  # 填写需要调用的模型名称
            messages=[
                {"role": "system", "content": "编写在西门子TIA Portal平台上运行的scl程序，实现相应功能。"},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stop=self.stop
        )
        return response.choices[0].message.content

    # 这个函数一般不单用，用来做并发访问的
    # 原式版本使用参数incremental=True 但是
    # 该参数的解释： 用于控制每次返回内容方式是增量还是全量，不提供此参数时默认为增量返回 - true 为增量返回 - false 为全量返回
    def llm_request(self, task):
        task_id = task[0]
        prompt = task[1]
        response = self.client.chat.completions.create(  # 同步调用 调用后即可一次性获得最终结果
            model=self.model_name,  # 填写需要调用的模型名称
            messages=[
                {"role": "system", "content": "编写在西门子TIA Portal平台上运行的scl程序，实现相应功能。"},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stop=self.stop
        )
        data = response.choices[0].message.content
        return [task_id, data]

    # 这个是做单轮的，每个task_id只生成一个答案
    # （这里的taskid是为了区分数据，因为多线程会打乱顺序）
    # 返回值是字典，key是task_id，value是大模型输出的字符串
    def submit_tasklist(self, task_list, max_workers=20):
        # 用于存放结果的列表
        result_list = []
        result_dic = {}

        # 创建一个线程池，并设置线程数
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 将任务列表中的每个任务提交给线程池执行，并将返回的Future对象存入future_list中
            future_list = [executor.submit(self.llm_request, task) for task in task_list]
            # 使用concurrent.futures.as_completed函数，等待所有Future对象完成，并获取结果
            for future in (concurrent.futures.as_completed(future_list)):
                task_id, llm_output = future.result()  # 获取返回值
                result_list.append(llm_output)  # 将结果添加到结果列表中
                result_dic[task_id] = llm_output
        return result_dic

    # 这个是做多轮的(比如代码的Pass@K)，每个task_id会重复生成多个答案
    # （这里的taskid是为了区分数据，因为多线程会打乱顺序）
    # 返回值是字典，key是task_id，value是列表，列表元素是字符串。
    def submit_tasklist_multiTurn(self, task_list, max_workers=20):
        # 用于存放结果的列表
        result_dic = {}
        for t in task_list:
            result_dic[t[0]] = []
        # 创建一个线程池，并设置线程数
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 将任务列表中的每个任务提交给线程池执行，并将返回的Future对象存入future_list中
            future_list = [executor.submit(self.llm_request, task) for task in task_list]
            # 使用concurrent.futures.as_completed函数，等待所有Future对象完成，并获取结果
            for future in (concurrent.futures.as_completed(future_list)):
                task_id, llm_output = future.result()  # 获取返回值
                result_dic[task_id].append(llm_output)
        return result_dic
