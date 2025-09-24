import debugpy

# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

import sys
sys.path.append('/home/huangwenke/Wenke_Project/LLaVAPro')

from llava.train.train import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
