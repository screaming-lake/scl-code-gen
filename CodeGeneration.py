# -*- coding: utf-8 -*-
import sys
import warnings
import json
import os
from  LLM_ZhiPuAI import *
import os
import torch  # type: ignore
from sentence_transformers import SentenceTransformer, models
from sentence_transformers import util as st_utils
import time
import re
from nltk.corpus import stopwords
import jieba
from typing import List
from BM25Retriever import BM25Retriever
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

#参数
model_name="glm-4-0520"
temperature= 0
max_tokens=4095


#数据加载
class Question_Dataset_Instance:
    def __init__(self, ins):
        self.ins = ins

    def nl_user_query(self):
        return self.ins['description']
    
    def get_name(self):
        return self.ins['name']

    def get_json(self):
        return self.ins

def load_sample():
    # assert os.path.exists(r'./sample_pro_plus.jsonl')
    data = []
    count = 0
    with open(r'./example_pro_plus_max.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            sample['ID'] = count
            count += 1
            data.append(sample)
    return data

def load_raw_sample():
    # assert os.path.exists(r'./sample_pro_plus.jsonl')
    data = []
    count = 0
    with open(r'./sample+cot.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            sample['ID'] = count
            count += 1
            data.append(sample)
    return data

def pre_process(data):  #没有的字段填充空列表
    if "output" not in data.keys():
        data["output"]=[]
    if "in/out" not in data.keys():
        data["in/out"]=[]
    if "return_value" not in data.keys():
        data["return_value"]=[]
    return Question_Dataset_Instance(data)

def post_process(raw_code):
    code_lines = raw_code.split('\n')
    start = -1
    end = -1

    for code_num, line in enumerate(code_lines):
        if '```' in line:
            if start == -1:
                start = code_num + 1
            else:
                end = code_num
                break

    target_code = '\n'.join(code_lines[start:end])
    target_code = target_code.replace('END_TEMP', 'END_VAR')
    target_code = target_code.replace('VAR_CONSTANT', 'VAR CONSTANT')
    target_code = target_code.replace('BYTE_TO_BYTE', '')
    target_code = target_code.replace('BYTE(', '(')
    target_code = target_code.replace('ELSEIF', 'ELSIF')
    target_code = target_code.replace('END_IF\n', 'END_IF;\n')
    target_code = target_code.replace('REAL_TO_REAL\n', '')
    target_code = target_code.replace('INT_TO_', '')
    target_code = target_code.replace('%', 'MOD')
    target_code = target_code.replace('FALSE : Bool := FALSE;', '')
    target_code = target_code.replace('TRUE : Bool := TRUE;', '')
    target_code = target_code.replace('FALSE : Bool := false;', '')
    target_code = target_code.replace('TRUE : Bool := true;', '')
    target_code = target_code.replace('#TRUE', 'TRUE')
    target_code = target_code.replace('#FALSE', 'FALSE')
    # if "INCREMENT : Int := 1;" not in target_code:
    #     target_code = target_code.replace('#INCREMENT', '1')
    
    target_code = re.sub(r'#region [\w]*\n', '', target_code)
    target_code = re.sub(r'#endregion [\w]*\n', '', target_code)
    target_code = re.sub(r'FOR (#\w*) := (\w*) DOWNTO (\w*) DO', r'FOR \1 := \2 TO \3 BY -1 DO', target_code)
    return target_code


#example编码及打分
warnings.simplefilter(action='ignore', category=FutureWarning)

# 这个就是dense retrieval用到的东西
class Embedding_Example_Encoder:
    def __init__(self):
        # 这里要替换路径
        assert os.path.exists(r'./paraphrase-multilingual-MiniLM-L12-v2')
        self.device = "cpu"
        word_embedding_model = models.Transformer(r'./paraphrase-multilingual-MiniLM-L12-v2')
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        sentence_encoder = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        self.encoder = sentence_encoder.to(self.device)
        self.samples = load_sample()

        for sample in self.samples:
            # 找出一个example的description
            description = sample['description']
            sample['description_embedding'] = self.encoder.encode(description, convert_to_tensor=True, device=self.device)

    def get_semantic_similarity(self, ins_query, exp_id):
        query_embedding = self.encoder.encode(ins_query, convert_to_tensor=True, device=self.device)
        sample = self.samples[int(exp_id)]
        target_embedding = sample['description_embedding']
        cos_scores = st_utils.pytorch_cos_sim(query_embedding, target_embedding)[0].detach().cpu().numpy().item()
        return cos_scores

class BM25_Example_Encoder:
    def __init__(self):
        # nltk.download('stopwords')
        self.samples = load_sample()
        documents = []
        for sample in self.samples:
            doc = Document(text=sample['description'])
            documents.append(doc)

        splitter = SentenceSplitter(chunk_size=1024)
        nodes = splitter.get_nodes_from_documents(documents)

        self.retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=len(documents), tokenizer=self.chinese_tokenizer)
        
    def chinese_tokenizer(text: str) -> List[str]:
        tokens = jieba.lcut(text)
        return [token for token in tokens if token not in stopwords.words('chinese')]

    def get_semantic_similarity(self, ins_query):
        nodes = self.retriever.retrieve(ins_query)
        texts = []
        for node in nodes:
            texts.append(node.text)
        return texts

# 根据ins给出评分最高的N个例子
def get_examples_by_ins_embedding(ins, expEncoder):
    assert isinstance(ins, Question_Dataset_Instance)

    samples = load_sample()
    # 通过ins特有的打分函数，这里的prompt要多次计算，每次都一样
    semantic_similarity = lambda sample: expEncoder.get_semantic_similarity(ins.nl_user_query(), sample['ID'])

    # 排序函数，简单一点
    sort_functions = [
        # 语义相似度越高越好
        lambda sample: -semantic_similarity(sample) * 12e2,
    ]

    sorted_samples = sorted(samples, key=lambda exp: sum([sort_fn(exp) for sort_fn in sort_functions]), reverse= not True)
    
    return sorted_samples


def get_examples_by_ins_BM25(ins, expEncoder):
    assert isinstance(ins, Question_Dataset_Instance)

    samples = load_sample()
    examples_desription = expEncoder.get_semantic_similarity(ins.nl_user_query())
    examples = []
    for description in examples_desription:
        for sample in samples:
            if sample['description'] == description:
                examples.append(sample)
                break
    
    return examples

def merge_examples(example1, example2, name, topN=3):
    samples = load_sample()
    raw_samples=load_raw_sample()
    # 选择'FloatingAverage','SearchMinMax_DInt','StackMin' 三个作为固定example
    result_samples=[]
    result_samples1=[]
    result_samples2=[]
    filtered_sorted_samples1=[s for s in example1 if not s['name'] in ['FloatingAverage','SearchMinMax_DInt','StackMin', name]]
    filtered_sorted_samples2=[s for s in example2 if not s['name'] in ['FloatingAverage','SearchMinMax_DInt','StackMin', name]]
    result_names = set()
    count = 0
    
    for i in range(len(filtered_sorted_samples1)):
        if filtered_sorted_samples1[i]['name'] not in result_names:
            result_samples1.append(filtered_sorted_samples1[i])
            result_names.add(filtered_sorted_samples1[i]['name'])
            count += 1
            if count == topN:
                break
    count = 0
    for i in range(len(filtered_sorted_samples2)):
        if filtered_sorted_samples2[i]['name'] not in result_names:
            result_samples2.append(filtered_sorted_samples2[i])
            result_names.add(filtered_sorted_samples2[i]['name']) 
            count += 1
            if count == topN:
                break
    for i in range(topN):
        # result_samples.append(result_samples1[i])

        name=result_samples1[i]['name']
        raw_sample=[s for s in raw_samples if s['name']==name][0]
        result_samples.append(raw_sample)

        name=result_samples2[i]['name']
        raw_sample=[s for s in raw_samples if s['name']==name][0]
        result_samples.append(raw_sample)
    
    FloatingAverage_sample=[s for s in samples if s['name']=='FloatingAverage'][0]
    SearchMinMax_DInt=[s for s in samples if s['name']=='SearchMinMax_DInt'][0]
    StackMin=[s for s in samples if s['name']=='StackMin'][0]
    result_samples.append(SearchMinMax_DInt)
    result_samples.append(StackMin)
    result_samples.append(FloatingAverage_sample)
    # result_samples.extend([filtered_sorted_samples[:topN-3]])
    return result_samples

#prompt构造
def build_question_and_sample(json_obj, is_sample=False):
    prompt_format = "题目：{title}\n\n功能描述：{description}\n\n输出类型：{type}\n\n名称：{name}\n\n"
    # 构建接口信息
    # api_info = ""
    # inp = json_obj["input"]
    # outp = json_obj["output"]
    # in_out = json_obj["in/out"]
    # return_value = json_obj["return_value"]
    if is_sample:
        answer = json_obj["answer"]
    else:
        answer = None
    
    api_info="程序框架：\n"
    api_info+=build_prompt_code(json_obj)

    if is_sample:
        #COT
        api_info+="\n逻辑步骤：\n"+json_obj["cot"]
        api_info += "\n答案：\n"+"```SCL\n"
        api_info += answer.strip() + "\n```\n"
    else:
        api_info += "\n逻辑步骤：\n"
    prompt = prompt_format.format(**json_obj) + api_info
    return prompt

def build_prompt(ins):
    FINAL_PROMPT = '''根据参考程序案例，完成代码填空任务，编写在西门子TIA Portal平台上运行的scl程序。你需要在【】内填入正确的代码，实现相应功能。
参考程序案例:
<public example>
{public_example}
</public example>
以下是可以直接调用的API，你生成的程序中可直接调用，不得使用除下列API以外的任何API，不得编造API。
<API>
**位操作函数**
SHR：
右移函数，使用“右移”指令，可以将参数 IN 的内容逐位向右移动，并将结果作为函数值返回。参数 N 用于指定应将特定值移位的位数。
使用示例为:
```scl
#Result:=SHR(IN := #Value, N := 1);
```
SHL：
左移函数，使用“左移”指令，可以将参数 IN 的内容逐位向左移动，并将结果作为函数值返回。参数 N 用于指定应将特定值移位的位数。
使用示例为:
```scl
#Result:=SHL(IN := #Value, N := 1);
```
</API>
以下是需要生成的代码，你需要在【】内填入正确的代码，实现相应功能。
注意：所生成的代码的格式均需要与示例程序相同，不得使用未在示例程序中的函数和语法。
注意：不得使用任何未定义的变量和函数，所有变量必须先定义再使用，包括循环中所用的循环变量和代表错误数字的常量。
注意：不得使用示例程序中未使用过的数据类型，你生成的代码中所有运算符两边的变量数据类型均需要在示例程序中出现过。
注意：scl代码与其它所有的代码语言均不兼容，不得使用任何未在示例程序中的语法格式，不得使用C语言的选择表达式、问号表达式和其它C语言特定的操作符和表达式。
注意：若你需要使用循环语句，不得使用输入变量作为循环变量，需多定义一个循环变量，而这个循环变量也必须先定义再使用。
注意：不得直接使用在参考程序案例中出现的变量，必须提前定义后才能使用。
注意：不得将IF语句或者CASE语句赋值给变量,这种格式scl不接受。若你要使用IF语句或者CASE语句，则代码块中必须为完整语句，不能只有一个变量。
注意：你生成的程序中不需要使用Region块。
注意：TRUE和FALSE为内置Bool类型变量的值，不需要定义。
注意：Int数值类型变量不得与Bool类型变量进行逻辑运算。
注意：Int数值类型变量不得与Bool类型变量进行算术运算。
注意：FOR循环中的循环变量必须为Int数值类型，不得为其它类型。
注意：定义变量赋值的时候只能赋一个值，不得通过表达式赋值。
{complete_user_query}

 '''
    
    examples1 = get_examples_by_ins_embedding(ins, Embedding_Example_Encoder())
    examples2 = get_examples_by_ins_BM25(ins, BM25_Example_Encoder())
    examples = merge_examples(examples1, examples2, ins.get_name(), topN=3)
    examples = examples[::-1]
    public_example = ''
    for example in examples:
        public_example += build_question_and_sample(example, is_sample=True)+"\n-------------------------------------------------\n"
    complete_user_query = build_question_and_sample(ins.get_json(), is_sample=False)

    variables = {
        "complete_user_query": complete_user_query,
        "public_example": public_example,
    }
    final_prompt = FINAL_PROMPT.format(**variables)
    return final_prompt

def build_prompt_code(json_obj):
    #参数信息
    json_obj["var"]=""
    if json_obj["input"]:
        json_obj["var"]+="\n    VAR_INPUT"
        for i in json_obj["input"]:
            i["description"]=i["description"].replace("\n",' ')
            if "fields" not in i.keys():
                json_obj["var"]+="\n        "+i["name"]+" : "+i["type"]+";"+"    //"+i["description"]
            else:
                json_obj["var"]+="\n        "+i["name"]+" : "+i["type"]+"    //"+i["description"]
                if type(i["fields"])==list:
                    for j in i["fields"]:
                        j["description"]=j["description"].replace("\n",' ')
                        json_obj["var"]+="\n            "+j["name"]+" : "+j["type"]+";"+"    //"+j["description"]
                else:
                    # print(type(i["fields"]))
                    if "description" in i["fields"].keys():
                        i["fields"]["description"]=i["fields"]["description"].replace("\n",' ')
                        json_obj["var"]+="\n            "+i["fields"]["name"]+" : "+i["fields"]["type"]+";"+"    //"+i["fields"]["description"]
                    else:
                        for k in i["fields"].keys():
                            i["fields"][k]["description"]=i["fields"][k]["description"].replace("\n",' ')
                            json_obj["var"]+="\n            "+i["fields"][k].get("name","name")+" : "+i["fields"][k].get("type","type")+";"+"    //"+i["fields"][k].get("description","description")
                json_obj["var"]+="\n        "+"END_STRUCT;"
        json_obj["var"]+="\n    END_VAR\n"
    if json_obj["output"]:
        json_obj["var"]+="\n    VAR_OUTPUT"
        for i in json_obj["output"]:
            i["description"]=i["description"].replace("\n",' ')
            if "fields" not in i.keys():
                json_obj["var"]+="\n        "+i["name"]+" : "+i["type"]+";"+"    //"+i["description"]
            else:
                json_obj["var"]+="\n        "+i["name"]+" : "+i["type"]+"    //"+i["description"]
                if type(i["fields"])==list:
                    for j in i["fields"]:
                        j["description"]=j["description"].replace("\n",' ')
                        json_obj["var"]+="\n            "+j["name"]+" : "+j["type"]+";"+"    //"+j["description"]
                else:
                   # print(type(i["fields"]))
                    if "description" in i["fields"].keys():
                        i["fields"]["description"]=i["fields"]["description"].replace("\n",' ')
                        json_obj["var"]+="\n            "+i["fields"]["name"]+" : "+i["fields"]["type"]+";"+"    //"+i["fields"]["description"]
                    else:
                        for k in i["fields"].keys():
                            i["fields"][k]["description"]=i["fields"][k]["description"].replace("\n",' ')
                            json_obj["var"]+="\n            "+i["fields"][k].get("name","name")+" : "+i["fields"][k].get("type","type")+";"+"    //"+i["fields"][k].get("description","description")
                json_obj["var"]+="\n        "+"END_STRUCT;"
        json_obj["var"]+="\n    END_VAR\n"
    if json_obj["in/out"]:
        json_obj["var"]+="\n    VAR_IN_OUT"
        for i in json_obj["in/out"]:
            i["description"]=i["description"].replace("\n",' ')
            if "fields" not in i.keys():
                json_obj["var"]+="\n        "+i["name"]+" : "+i["type"]+";"+"    //"+i["description"]
            else:
                json_obj["var"]+="\n        "+i["name"]+" : "+i["type"]+"    //"+i["description"]
                if type(i["fields"])==list:
                    for j in i["fields"]:
                        j["description"]=j["description"].replace("\n",' ')
                        json_obj["var"]+="\n            "+j["name"]+" : "+j["type"]+";"+"    //"+j["description"]
                else:
                    # print(type(i["fields"]))
                    if "description" in i["fields"].keys():
                        i["fields"]["description"]=i["fields"]["description"].replace("\n",' ')
                        json_obj["var"]+="\n            "+i["fields"]["name"]+" : "+i["fields"]["type"]+";"+"    //"+i["fields"]["description"]
                    else:
                        for k in i["fields"].keys():
                            i["fields"][k]["description"]=i["fields"][k]["description"].replace("\n",' ')
                            json_obj["var"]+="\n            "+i["fields"][k].get("name","name")+" : "+i["fields"][k].get("type","type")+";"+"    //"+i["fields"][k].get("description","description")
                json_obj["var"]+="\n        "+"END_STRUCT;"
        json_obj["var"]+="\n    END_VAR\n"
    json_obj["var"]+='''
    VAR 
        【】
    END_VAR
    
    VAR_TEMP
        【】
    END_VAR
    
    VAR CONSTANT
        【】
    END_VAR'''
    
    if json_obj["type"]=="FUNCTION_BLOCK":
        p_format='''{type} "{name}"
{{ S7_Optimized_Access := 'TRUE' }}
    {var}

BEGIN
    【】
END_{type}
    '''
    elif json_obj["type"]=="FUNCTION":
        #类型为函数且没有返回值，return_value置为Void
        if not json_obj["return_value"]:
            json_obj["return_value"].append({"type":"Void","description":""})
        p_format='''{type} "{name}" : {return_value[0][type]}
{{ S7_Optimized_Access := 'TRUE' }}
    {var}

BEGIN
    【】
END_{type}
''' 
    return p_format.format(**json_obj)

def get_chunk(index, code):
    chunk = []
    flag = False
    while index < len(code):
        line = code[index]
        chunk.append(line)
        index += 1
        if "IF " in line:
            count = 1
            while (index < len(code)) and (count != 0):
                line = code[index]
                chunk.append(line)
                index += 1
                if "END_IF;" in line:
                    count -= 1
                elif "IF " in line:
                    count += 1
            flag = True
            break
        elif "FOR " in line:
            count = 1
            while (index < len(code)) and (count != 0):
                line = code[index]
                chunk.append(line)
                index += 1
                if "END_FOR;" in line:
                    count -= 1
                elif "FOR " in line:
                    count += 1
            flag = True
            break
        elif ';' in line:
            break
    return index, chunk, flag

known_apis=['ASM', 'A0', 'A1', 'AB', 'ABS', 'ABSTRACT', 'ACOS', 'ACTION', 'AD', 'ADD', 'ALTERNATIVE_BRANCH', 'AND', 'Any_Array', 'Any_BCD', 'Any_Bit', 'Any_Block', 'Any_Char', 'Any_Chars', 'Any_CodeBlock', 'Any_DataBlock', 'Any_Date', 'Any_Duration', 'Any_Elementary', 'Any_Int', 'Any_Magnitude', 'Any_Num', 'Any_Ordered', 'Any_Pointer', 'Any_Real', 'Any_Reference', 'Any_Signed', 'Any_String', 'Any_Struct', 'Any_Structured', 'Type', 'Any_TypeBlock', 'Any_TypedReference', 'Any_UnOrdered', 'Any_UnSigned', 'Any_UnTypedRe', 'ference', 'AR1', 'AR2', 'ASIN', 'AT', 'ATAN', 'AUTHOR', 'AW', 'BEGIN', 'BIE', 'BR', 'BROWSERINFO', 'BR', 'BY', 'CALL', 'CASE', 'CAUSE', 'CAUSE_GROUP', 'CC0', 'CC1', 'CEIL', 'CLASS', 'CODE_VERSION1', 'COMM_BLOCK', 'CONCAT', 'CONFIGURATION', 'CONST', 'CONSTANT', 'CONTINUE', 'COS', 'DATA_BLOCK', 'DATATYPE', 'DB', 'DB_SPECIFIC', 'DBB', 'DBD', 'DBLG', 'DBNO', 'DBW', 'DBX', 'DCHAR', 'DELETE', 'DI', 'DIB', 'DID', 'DILG', 'DINO', 'DIV', 'DIW', 'DIX', 'DO', 'DT', 'EB', 'ED', 'EFFECT', 'EFFECT_GROUP', 'ELEMENT', 'ELSE', 'ELSIF', 'EN', 'END_POST_OPERATION', 'END_PRE_OPERATION', 'END_ACTION', 'END_ALTERNATIVE_BRANCH', 'END_BROWSERINFO', 'END_CASE', 'END_CAUSE', 'END_CAUSE_GROUP', 'END_CLASS', 'END_CONFIGURATION', 'END_CONST', 'END_DATA_BLOCK', 'END_EFFECT', 'END_EFFECT_GROUP', 'END_ELEMENT', 'END_FOR', 'END_FOREACH', 'END_FUNCTION', 'END_FUNCTION_BLOCK', 'END_IF', 'END_INTERFACE', 'END_INTERLOCK', 'END_INTERSECTIONS', 'END_LIBRARY', 'END_NAMESPACE', 'END_NETWORK', 'END_NAMESPACE', 'END_NETWORK', 'END_NODE', 'END_ORGANIZATION_BLOCK', 'END_PROGRAM', 'END_REGION', 'END_REPEAT', 'END_REQUIRE', 'END_RESOURCE', 'END_RUNG', 'END_SELECTION', 'END_SEQUENCE', 'END_SIMULTAN', 'EOUS_BRANCH', 'END_STEP', 'END_STRUCT', 'END_SUPERVISION', 'END_SYSTEM_FUNCTION', 'END_SYSTEM_FUNCTION_BLOCK', 'END_TRANSITION', 'END_TYPE', 'END_VAR', 'END_WHILE', 'END_WIRE', 'ENO', 'ENTRY', 'EQ', 'EW', 'EXIT', 'EXPT', 'EXTENDS', 'F_EDGE', 'FALSE', 'FAMILY', 'FB', 'FC', 'FINAL', 'FIND', 'FLOOR', 'FOR', 'FOREACH', 'FUNCTION', 'FUNCTION_BLOCK', 'GE', 'GOTO', 'GT', 'IB', 'ID', 'IF', 'IMPLEMENTATION', 'IMPLEMENTS', 'INSERT', 'INSIDE', 'INTERFACE', 'INTERLOCK', 'INTERNAL', 'INTERSECTIONS', 'INTERVAL', 'IW', 'KNOW_HOW_PROTECT', 'LABEL', 'LB', 'LD', 'LDATE', 'LE', 'LDATE_AND_TIME', 'LEFT', 'LEN', 'LIBRARY', 'LIMIT', 'LN', 'LOG', 'LOWER_BOUND', 'LT', 'LTIME_OF_DAY', 'LW', 'MAX', 'MB', 'MD', 'MDD_CHECK', 'METHOD', 'MID', 'MIN', 'MOD', 'MOVE', 'MUL', 'MUX', 'MW', 'NAME', 'NAME_OF', 'NAMESPACE', 'NE', 'NETWORK', 'NODE', 'NON_RETAIN', 'NOP', 'NOT', 'NU', 'NULL', 'OB', 'OF', 'ON', 'OR', 'ORGANIZATION_BLOCK', 'OS', 'OV', 'OVERLAP', 'OVERRIDE', 'PA', 'PAB', 'PACKED', 'PAD', 'PAW', 'PB', 'PE', 'PD', 'PEB', 'PED', 'PEW', 'PI', 'PIB', 'PID', 'PIW', 'POST_OPERATION', 'PQ', 'PQB', 'PQD', 'PQW', 'PRAGMA_BEGIN', 'PRAGMA_END', 'PRE_OPERATION', 'PRIORITY', 'PRIVATE', 'PROGRAM', 'PROTECTED', 'PUBLIC', 'PW', 'QB', 'QD', 'QW', 'R_EDGE', 'READ_ONLY', 'READ_WRITE', 'REF', 'REF_TO', 'REGION', 'RELATION', 'REPEAT', 'REPLACE', 'REQUIRE', 'RESOURCE', 'RET_VAL', 'RETAIN', 'RETURN', 'RIGHT', 'ROL', 'ROR', 'RUNG', 'S5T', 'SDB', 'SEL', 'SELECTION', 'SEQUENCE', 'SFB', 'SFC', 'SHL', 'SHR', 'SIMULTANEOUS_BRANCH', 'SIN', 'SINGLE', 'SQRT', 'STANDARD', 'STEP', 'STW', 'SUB', 'SUBSET', 'SUPER', 'SUPERVISION', 'SYSTEM_FUNCTION', 'SYSTEM_FUNCTION_BLOCK', 'TAN', 'TASK', 'THEN', 'THIS', 'TITLE', 'TO', 'TO_BOOL', 'TO_BYTE', 'TO_CHAR', 'TO_DATE', 'TO_DINT', 'TO_DT', 'TO_DWORD', 'TO_INT', 'TO_LDATE', 'TO_LDT', 'TO_LINT', 'TO_LREAL', 'TO_LTIME', 'TO_LTOD', 'TO_LWORD', 'TO_REAL', 'TO_SINT', 'TO_TIME', 'TO_TOD', 'TO_UDINT', 'TO_UINT', 'TO_ULINT', 'TO_USINT', 'TO_WCHAR', 'TO_WORD', 'TOD', 'TRANSITION', 'TRUE', 'TRUNC', 'TYPE', 'UDT', 'UNLINKED', 'UNTIL', 'UO', 'UPPER_BOUND', 'USING', 'USTRING', 'VAR', 'VAR_ACCESS', 'VAR_CONFIG', 'VAR_EXTERNAL', 'VAR_GLOBAL', 'VAR_IN_OUT', 'VAR_INPUT', 'VAR_OUTPUT', 'VAR_TEMP', 'VERSION', 'WHILE', 'WIRE', 'WITH', 'XOR']
def get_apis4(code):
    pattern = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b')
    all_keywords = pattern.finditer(code)
    function_names=[]
    # 遍历匹配项及其在原文中的位置
    last_name=''
    for match in all_keywords:
        next_char=code[min(match.end(),len(code)-1)]
        previous_char=code[max(match.start()-1,0)]
        # 一个name的右边是左括号而且不是关键词
        if  next_char=='(' :
            function_names.append(match.group())
        last_name=match.group()
    return function_names

def get_chunk(index, code):
    chunk = []
    flag = False
    while index < len(code):
        line = code[index]
        chunk.append(line)
        index += 1
        if "IF " in line:
            count = 1
            while (index < len(code)) and (count != 0):
                line = code[index]
                chunk.append(line)
                index += 1
                if "END_IF;" in line:
                    count -= 1
                elif "IF " in line:
                    count += 1
            flag = True
            break
        elif "FOR " in line:
            count = 1
            while (index < len(code)) and (count != 0):
                line = code[index]
                chunk.append(line)
                index += 1
                if "END_FOR;" in line:
                    count -= 1
                elif "FOR " in line:
                    count += 1
            flag = True
            break
        elif ';' in line:
            break
    return index, chunk, flag

def delete_chunk(code):

    pattern = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b')
    all_keywords = pattern.finditer(code)
    to_be_deleted=[]

    # 遍历匹配项及其在原文中的位置

    new_code=''
    for match in all_keywords:
        next_char=code[min(match.end(),len(code)-1)]

        if  not next_char=='(' :
            continue

        if match.group() in known_apis:
            continue
        # print(f'! {match.group()} 函数未定义')

        # 先找到右边的分号
        for i in range(match.start(),len(code)-1):
            to_be_deleted.append(i)
            if code[i]==';':
                break

        # 再找到左边的分号
        i=match.start()
        new_line = False

        while i >= 0:

            if code[i]==';':
                break
            elif code[i-4:i]=='THEN' or code[i-2:i]=='DO':
                break
            elif code[i]=='\n':
                new_line = True
            elif (code[i-2:i]=='IF' or code[i-3:i]=='FOR') and (not new_line):
                if code[i-2:i]=='IF':
                    index = i - 2 + 1
                    to_be_deleted.append(i-2)
                    to_be_deleted.append(index)
                    count = 1
                    while (index < len(code)) and (count != 0):
                        if code[index:index+3] == 'IF ':
                            count = 1
                        elif code[index-7:index] == 'END_IF;':
                            count -= 1
                        to_be_deleted.append(index)
                        index += 1
                    break
                else:
                    index = i - 3 + 1
                    to_be_deleted.append(i-3)
                    to_be_deleted.append(index)
                    count = 1
                    while (index < len(code)) and (count != 0):
                        if code[index:index+4] == 'FOR ':
                            count = 1
                        elif code[index-8:index] == 'END_FOR;':
                            count -= 1
                        to_be_deleted.append(index)
                        index += 1
                    break
            to_be_deleted.append(i)
            i-=1
        
    for i in range(len(code)):
        if not i in to_be_deleted:
            new_code+=code[i]
    return new_code.split('\n')

# 先删除行注释
def check_var_undefined(var_definition,code):
    cleaned_code = re.sub(r'//.*?\n', '', code)  
    # 然后再找到所有的# 开头变量
    def is_digit(s):  
        return bool(re.search(r'\d', s))  
    pattern = re.compile(r'#\w.*?(?=\W)')  
    matches = pattern.finditer(cleaned_code)  
    for match in matches:  
        var=match.group()
        if is_digit(var):
            continue
        var_=var.replace('#','')
        if not var_ in var_definition:
            # print(f'变量未定义：{var}')
            return True
        
    # pattern = re.compile(r'\.\w.*?(?=\W)') 
    pattern = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b') 
    
    matches = pattern.finditer(cleaned_code)  
    for match in matches:  
        start=match.start()
        if  start==0 or not code[start-1]=='.':
            continue
        var=match.group()
        if is_digit(var):
            continue
        var_=var.replace('.','')
        if not var_ in var_definition:
            # print(f'变量未定义：{var}')
            return True

    return False

def remove_unknown_apis(ans):
    # 先拿到所有变量定义的块
    pattern = re.compile(r'VAR.*?\n.*?END_VAR', re.DOTALL)  # 使用 re.DOTALL 标志  
    matches = pattern.findall(ans)  
    var_definition=''
    for match in matches:  
        var_definition+=match

    new_code=[]
    begin = False
    codes = ans.split('\n')
    index = 0
    change = False
    while (index < len(codes)) and (not begin):
        new_code.append(codes[index])
        if "BEGIN" in codes[index]:
            begin = True
        index += 1
        
    while index < len(codes):
        index, chunk, flag = get_chunk(index, codes)
        apis=get_apis4('\n'.join(chunk))
        add = True
        for api in apis:
            # print(api)
            if not api in known_apis:
                add = False
                break
        
        if check_var_undefined(var_definition,'\n'.join(chunk)):
            # print('#==================================')
            # print('该块被弃用')
            # print('\n'.join(chunk))
            # print('==================================#')
            add = False

        if add:
            new_code.extend(chunk)
        elif flag:
            change = True
            chunk = delete_chunk('\n'.join(chunk))
            if chunk is not None:
                new_code.extend(chunk)
        else:
            change = True
        # print(index)
        # print('\n'.join(chunk))
        # print('==================================')
    answer = '\n'.join(new_code)
    if change:
        answer = supplement(answer)
    return answer

def supplement(code):
    codes = code.split('\n')
    index = 0
    while(index < len(codes)):
        line = codes[index]
        if "IF " in line:
            start = index
            index += 1
            while (index < len(codes)):
                line = codes[index]
                if "IF " in line:
                    start = index
                elif "END_IF;" in line:
                    end = index
                    break
                index += 1
            add = True

            for i in range(start+1, end):
                line = codes[i]
                if not line.replace('\n', '').replace(' ', '') == '':
                    add = False
            if add:
                codes.insert(start+1, ";\n")
                index += 1
        elif "FOR " in line:
            start = index
            index += 1
            while (index < len(codes)):
                line = codes[index]
                if "FOR " in line:
                    start = index
                elif "END_FOR;" in line:
                    end = index
                    break
                index += 1
            add = False
            for i in range(start+1, end):
                line = codes[i]
                if not line.replace('\n', '').replace(' ', '') == '':
                    add = True
            if add:
                codes.insert(start+1, ";\n")
                index += 1
        else:
            index += 1
    return '\n'.join(codes)
#运行
def code_generation(data):
 
    qus=pre_process(data)

    llm=LLM_ZhiPuAI(temperature,max_tokens=max_tokens,stop=["```\n"])
    llm.model_name=model_name
    
    prompt=build_prompt(qus)
    # print(prompt)
    # sys.exit(0)
    #start_time = time.time()
    predict_res=llm(prompt)
    #end_time = time.time()
    
    #elapsed_time = end_time - start_time
    # print(f'程序生成时间{elapsed_time}')
    code=post_process(predict_res)
    code2=remove_unknown_apis(code)
    
    #with open('./prompt.txt', 'w', encoding='utf-8') as f:
    #    f.write(prompt)
    #with open('./answer.scl', 'w', encoding='utf-8') as f:
    #    f.write(code)
    
    return code2

if __name__ == '__main__':
    # assert os.path.exists(r'./question.jsonl')
    llm=LLM_ZhiPuAI(temperature,max_tokens=max_tokens,stop=["```\n"])
    llm.model_name=model_name
    predict_res={}
    task_list=[]
    with open(r'./sample.jsonl', 'r', encoding='utf-8') as f:
        os.makedirs('./answer', exist_ok=True)
        for line in f:
            data = json.loads(line.strip())
            qus=pre_process(data)
            prompt=build_prompt(qus)
            task_id=data['name']
            predict_res[task_id]=[]
            task_list.append([task_id,prompt])

        predict_res=llm.submit_tasklist(task_list,max_workers=15)
        for key, value in predict_res.items(): 
            path = './answer/' + key + '.scl'
            with open(path, 'w', encoding='utf-8') as f:
                f.write(remove_unknown_apis(post_process(value)))

    