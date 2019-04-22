#!/usr/bin/python3
#  -*- coding: utf-8 -*-

# hdfs = HDFileSystem(host='hdfs-bizaistca.corp.microsoft.com', port=8020, user='hadoop')
from hdfs.client import Client

hdfs_path = '/user/hadoop/fanyuguang/input/'
local_path = '.'

client = Client("hdfs-bizaistca.corp.microsoft.com:8020/", root="/", timeout=10000, session=False)
result = client.list(hdfs_path, status=False)
print(result)
# client.download(hdfs_path, local_path, overwrite=False)