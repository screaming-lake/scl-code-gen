# 基础镜像
FROM python:3.10-slim
# 设置⼯作⽬录
WORKDIR /app
# 复制依赖⽂件到⼯作⽬录
COPY requirements.txt /app/requirements.txt
COPY PyStemmer-2.2.0.1-cp310-cp310-linux_x86_64.whl /app/PyStemmer-2.2.0.1-cp310-cp310-linux_x86_64.whl
# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install ./PyStemmer-2.2.0.1-cp310-cp310-linux_x86_64.whl
RUN python -c "import nltk; nltk.download('stopwords')"
# 复制项⽬⽂件到⼯作⽬录
COPY . /app
# 暴露端⼝
EXPOSE 8000
# 启动服务
CMD ["python", "app.py"]