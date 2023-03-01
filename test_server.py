import requests


print(requests.post('http://127.0.0.1:8081/predict?url=1').json())
