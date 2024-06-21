from flask import Flask, request, jsonify
import pandas as pd
import os
from dotenv import load_dotenv, find_dotenv
import openai
import numpy as np
import whisper
import statsmodels.api as sm
import matplotlib.pyplot as plt
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain.agents import AgentExecutor
from queue import Queue
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load environment variables
load_dotenv(find_dotenv())
client = openai.Client()

app = Flask(__name__)
CORS(app)

class TalkingLLM:
    def __init__(self, model="gpt-3.5-turbo-0613", whisper_size="small"):
        self.selected_dataframe = None
        self.voice_enabled = True

        self.whisper = whisper.load_model(whisper_size)
        self.llm = ChatOpenAI(model=model)
        self.llm_queue = Queue()

        self.csv_files = self.get_csv_files()
        self.df = None  # Inicialmente nenhum DataFrame selecionado

    def get_csv_files(self):
        datasets_path = "datasets"
        return [f for f in os.listdir(datasets_path) if f.endswith('.csv')]

    def select_dataframe(self, file_index):
        selected_file = self.csv_files[file_index]
        self.df = pd.read_csv(os.path.join("datasets", selected_file))
        self.create_agent()

    def create_agent(self):
        if self.df is None:
            raise ValueError("DataFrame não está definido.")

        agent_prompt_prefix = """
            Você se chama Beemo, e está trabalhando com dataframe pandas no Python. O nome do Dataframe é `df`.
            Você sabe como realizar análise de regressão utilizando `statsmodels` e pode executar código Python diretamente para realizar análises de dados.
        """
        self.agent = create_pandas_dataframe_agent(
            self.llm,
            self.df,
            prefix=agent_prompt_prefix,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
        )
        self.agent_executor = AgentExecutor(
            agent=self.agent.agent,
            tools=self.agent.tools,
            verbose=True,
            handle_parsing_errors=True
        )

    def handle_input(self, user_input):
        if "gerar gráfico" in user_input.lower():
            return self.generate_graph(user_input)
        else:
            try:
                response = self.agent_executor.invoke({"input": user_input})
                if 'execute' in response['output']:
                    exec_globals = {}
                    exec(response['output'], {}, exec_globals)
                    return exec_globals['results_summary']
                else:
                    return response["output"]
            except ValueError as e:
                return "Erro de parsing: " + str(e)
            except json.JSONDecodeError as e:
                return "Erro ao decodificar JSON: " + str(e)

    def generate_graph(self, user_input):
        try:
            if "price" in user_input.lower():
                plt.figure(figsize=(10, 6))
                plt.plot(self.df['Price'])
                plt.xlabel('Index')
                plt.ylabel('Price')
                plt.title('Price Variation')
                plt.savefig('static/price_variation.png')
                plt.close()
                return "Gráfico salvo como 'static/price_variation.png'."
            else:
                return "Desculpe, não consegui identificar a variável para o gráfico."
        except Exception as e:
            return f"Erro ao gerar o gráfico: {e}"

llm = TalkingLLM()

@app.route('/datasets', methods=['GET'])
def list_datasets():
    csv_files = llm.get_csv_files()
    return jsonify({"datasets": csv_files})

@app.route('/select_dataset', methods=['POST'])
def select_dataset():
    data = request.json
    file_index = data['file_index']
    llm.select_dataframe(file_index)
    return jsonify({"message": "Dataset selected successfully."})

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    user_input = data['input']
    response = llm.handle_input(user_input)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
