from flask import Flask, request, jsonify
from CodeGeneration import *

app = Flask(__name__)

@app.route('/', methods=['POST'])
def generate_code():
    data = request.get_json()
    # 模拟代码⽣成逻辑
    code = code_generation(data=data)
    response = {
        "name": data.get("name"),
        "code": code
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)