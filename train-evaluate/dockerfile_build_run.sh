docker build -t segment-train:0.1.0 .

docker run -e HDFS_HOST='hdfs-bizaistca.corp.microsoft.com' -e HDFS_PORT='8020' -e HDFS_USER='hadoop' -e INPUT_PATH='/user/hadoop/fanyuguang/input/' -e OUTPUT_PATH='/user/hadoop/fanyuguang/output/' -t segment-train:0.1.0
