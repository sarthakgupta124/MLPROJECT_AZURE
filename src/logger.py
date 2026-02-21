import logging 
import os 
from datetime import datetime

stamp=datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
logs_path=os.path.join(os.getcwd(),"logs",f"{stamp}")
os.makedirs(logs_path,exist_ok=True)

LOG_file=f"{stamp}.log"
LOG_file_path=os.path.join(logs_path,LOG_file)

logging.basicConfig(
    filename=LOG_file_path,
    format="[%(asctime)s] %(lineno)d %(name)s-%(levelname)s-%(message)s",
    level=logging.INFO,
)
