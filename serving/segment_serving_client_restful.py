#!/usr/bin/python3
# -*- coding: utf-8 -*-

import json
import requests

# curl -d '{"signature_name":"predict_segment", "instances": [{"input_sentences": "明 天 去 上 海"}]}' -X POST http://localhost:8501/v1/models/segment:predict
segment_url = 'http://localhost:8501/v1/models/segment:predict'
data = {"signature_name":"predict_segment", "instances": [{"input_sentences": "明 天 去 上 海"}]}
response = requests.post(segment_url, json=data)
print(response.text)
