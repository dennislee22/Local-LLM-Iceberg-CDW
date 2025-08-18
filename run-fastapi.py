import os
CDSW_APP_PORT=os.environ['CDSW_APP_PORT'] 
os.system("python fastapi-llm.py --server-name=127.0.0.1 --checkpoint-path=Qwen2.5-14B-Instruct-1M --server-port=$CDSW_APP_PORT > fastapi.log 2>&1")
