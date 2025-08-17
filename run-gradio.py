import os
CDSW_APP_PORT=os.environ['CDSW_APP_PORT'] 
os.system("python app-gradio.py --server-name=127.0.0.1 --server-port=$CDSW_APP_PORT > gradio.log 2>&1")