from flask import request, Flask, stream_with_context
from flask_cors import CORS
import logging
import time
import json

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
logging.getLogger('flask_cors').level = logging.DEBUG

response, response_time = "", -1

@app.post("/process")
def ask():
    data = request.get_json()
    def generate_response():
        yield json.dumps({"information": "Started processing information!"})

    return app.response_class(stream_with_context(generate_response()))

@app.post("/send_response")
def send_response():
    global response, response_time
    data = request.get_json()
    response = data["message"]
    response_time = time.time()
    return {'ok': True}

def wait_for_response(request_time):
    global response, response_time
    while request_time < response_time:
        return response
    
