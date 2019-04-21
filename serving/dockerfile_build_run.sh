docker build -t segment-service:0.1.0 .

docker run -p 8500:8500 -p 8501:8501 -e MODEL_NAME='segment' -e MODEL_BASE_PATH='hdfs://hdfs-bizaistca.corp.microsoft.com:8020/user/hadoop/fanyuguang/output/saved-model-data' -t segment-service:0.1.0
