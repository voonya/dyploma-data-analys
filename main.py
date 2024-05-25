from flask import Flask, jsonify, request
from flask_cors import CORS
from check_propaganda import is_propaganda
from get_topic import get_topics, get_vis

app = Flask(__name__)
CORS(app)

@app.route('/is-propaganda', methods=['POST'])
def check_is_propaganda():
    data = request.json
    
    print(data)

    result = is_propaganda(data['msgs'])

    return jsonify(result), 200

@app.route('/topics', methods=['POST'])
def get_topic():
    data = request.json
    
    print(data)

    result = get_topics(data['msgs'])

    return jsonify(result), 200

@app.route('/vis', methods=['GET'])
def get_vis_re():
    vis = get_vis()
    print(jsonify({'template': vis}))
    return jsonify({'template': vis}), 200

if __name__ == '__main__':
    app.run(debug=True)
