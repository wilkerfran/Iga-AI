from flask import Flask, request, render_template, send_file
import pandas as pd
import matplotlib.pyplot as plt
import os
from beemo import TalkingLLM  # Atualize o nome do arquivo aqui

app = Flask(__name__)
talking_llm = TalkingLLM()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_graph', methods=['POST'])
def generate_graph():
    user_input = request.form['text']
    response = talking_llm.handle_input(user_input)
    
    if os.path.exists('static/price_variation.png'):
        return render_template('index.html', response=response, image='static/price_variation.png')
    else:
        return render_template('index.html', response=response)

if __name__ == '__main__':
    app.run(debug=True)
