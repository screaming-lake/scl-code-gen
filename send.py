import os  
import json  
import requests  

# 接口地址和端口范围  
base_url = 'http://localhost:'  
port = 8000


def send_request(data, port):  
    """  
    发送POST请求并处理响应。  
    """  
    url = f"{base_url}{port}"  
    try:  
        headers = {'Content-Type': 'application/json'}  
        response = requests.post(url, json=data, headers=headers)
        print(response.elapsed.seconds)

        # 处理响应  
        if response.status_code == 200:  
            content = response.json()
            success = True
            if content in [None, [], {}]:  
                content = {"error": "Empty result"}  
                print(f"error: Empty result")
                success = False
            if success:
                print('Code Generation success')
            return content
   
    except Exception as e:  
        print(f"Unexpected error processing {data['name']}: {str(e)}")   
  
if __name__ == "__main__":  
    assert os.path.exists(r'./question.jsonl')
    with open(r'./question.jsonl', 'r', encoding='utf-8') as f:
        os.makedirs('./answer', exist_ok=True)
        for line in f:
            data = json.loads(line.strip())
            if data['name'] == 'ExtractStringFromCharArray':
                response = send_request(data, port)
            
                path = './answer/' + data['name'] + '.scl'
                with open(path, 'w', encoding='utf-8-sig') as f:
                    f.write(response['code'])
