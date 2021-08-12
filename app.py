from argparse import ArgumentParser
import base64
import datetime
import pytz
from pathlib import Path
import hashlib
import io

from flask import Flask
from flask import request
from flask import jsonify
import numpy as np

# 自製模型預測模組
from predict import predict

TZ = pytz.timezone('Asia/Taipei')
WAV_FILES_PATH = Path("./wav_files/")

app = Flask(__name__)

####### PUT YOUR INFORMATION HERE #######
CAPTAIN_EMAIL = ''  # 隊長信箱
SALT = 'StarRingChild'  # 加密用字串 隨意輸入即可 此處輸入隊名
API_KEY = ''
#########################################

@app.route('/classify', methods=['POST'])
def classify():
    """ API that return your model predictions when TomoFun calls this API. """
    # 接收 TomoFun 發來的Request
    if request.headers['x-api-key'] != API_KEY:
        return jsonify({'message': 'unknown api key'})

    # 存成 wav file
    ts_now = datetime.datetime.now(TZ).strftime("%Y%m%d%H%M%S")
    file_name = ts_now + '.wav'
    file_path = WAV_FILES_PATH/file_name
    with open(file_path, 'bx') as f:
        f.write(request.data)

    # 模型預測
    try:
        answer = predict(file_path)
    except TypeError as type_error:
        raise type_error
    except Exception as e:
        raise e
    
    # 提交API預測結果
    result = {'label': answer[0],  
                    'probability': answer[1]}
    with open("log.txt", "a+") as f:
        f.write(ts_now + " " + str(result) + "\n")
    return jsonify(result)

if __name__ == "__main__":
    arg_parser = ArgumentParser(
        usage='Usage: python ' + __file__ + ' [--port <port>] [--help]'
    )
    arg_parser.add_argument('-p', '--port', default=8080, help='port')
    arg_parser.add_argument('-d', '--debug', default=False, help='debug')
    options = arg_parser.parse_args()

    app.run(host='0.0.0.0', debug=options.debug, port=options.port)
