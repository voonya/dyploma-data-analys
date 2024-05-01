from flask import Flask, jsonify, request
from check_propaganda import is_propaganda
from get_topic import get_topics

app = Flask(__name__)

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

if __name__ == '__main__':
    app.run(debug=True)
