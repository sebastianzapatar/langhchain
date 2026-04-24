import requests
import json

res = requests.post("http://localhost:8000/chat", json={"mensaje": "De que tecnologías hablo en mis documentos?"})
print(json.dumps(res.json(), indent=2))
