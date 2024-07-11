import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents import AgentExecutor, AgentType
from dotenv import load_dotenv, find_dotenv
import openai

# Load environment variables
load_dotenv(find_dotenv())

app = Flask(__name__)
CORS(app)

class TalkingLLM:
    def __init__(self, model="gpt-3.5-turbo-0613"):
        self.selected_dataframe = None
        self.llm = ChatOpenAI(model=model)
        self.csv_files = self.get_csv_files()
        self.df = None
        self.agent_executor = None

    def get_csv_files(self):
        datasets_path = "datasets"
        return [f for f in os.listdir(datasets_path) if f.endswith('.csv')]

    def select_dataframe(self, file_index):
        file_index = int(file_index)
        selected_file = self.csv_files[file_index]
        print(f"Selecionando o dataset: {selected_file}")  # Log para verificar o arquivo selecionado
        self.df = pd.read_csv(os.path.join("datasets", selected_file))
        self.columns = self.df.columns.tolist()
        self.create_agent()

    def create_agent(self):
        if self.df is None:
            raise ValueError("DataFrame não está definido.")
        
        agent_prompt_prefix = """
            Você se chama Beemo, e está trabalhando com dataframe pandas no Python. O nome do Dataframe é df.
            Você sabe como realizar análise de regressão utilizando statsmodels e pode executar código Python diretamente para realizar análises de dados.
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
        if any(term in user_input.lower() for term in ["gráfico", "gere um gráfico", "crie um gráfico", "construa um gráfico"]):
            return self.generate_graph(user_input)
        else:
            try:
                response = self.agent_executor.invoke({"input": user_input})
                return response['output']
            except Exception as e:
                return str(e)

    def generate_graph(self, user_input):
        try:
            plt.clf()
            # Identificar o tipo de gráfico solicitado
            if "barras" in user_input.lower() or "barra" in user_input.lower():
                graph_type = "bar"
            elif "linha" in user_input.lower():
                graph_type = "line"
            elif "scatter" in user_input.lower() or "dispersão" in user_input.lower():
                graph_type = "scatter"
            else:
                return {"error": "Tipo de gráfico não suportado."}

            # Extrair as colunas mencionadas pelo usuário
            columns = self.extract_columns(user_input)
            if len(columns) != 2:
                return {"error": "Precisamos de exatamente duas colunas para gerar um gráfico."}

            if graph_type == "bar":
                plt.bar(self.df[columns[0]], self.df[columns[1]])
            elif graph_type == "line":
                plt.plot(self.df[columns[0]], self.df[columns[1]])
            elif graph_type == "scatter":
                plt.scatter(self.df[columns[0]], self.df[columns[1]])

            plt.xlabel(columns[0])
            plt.ylabel(columns[1])
            plt.title(f'{columns[0]} vs {columns[1]}')

            img_path = os.path.join("static", "chart.png")
            plt.savefig(img_path)
            
            # Gerar texto explicativo com base no gráfico gerado
            text = f"Aqui está um gráfico {graph_type} mostrando a relação entre {columns[0]} e {columns[1]}."

            result = {"img_url": f"/{img_path}", "text": text}
            self.save_output_to_file(user_input, json.dumps(result))
            return result
        except Exception as e:
            error_message = {"error": f"Erro ao gerar o gráfico: {e}"}
            self.save_output_to_file(user_input, json.dumps(error_message))
            return error_message

    def extract_columns(self, user_input):
        columns = []
        for col in self.columns:
            if col.lower() in user_input.lower():
                columns.append(col)
        return columns

    def save_output_to_file(self, user_input, output):
        if not os.path.exists("output"):
            os.makedirs("output")
        filename = f"output/{len(os.listdir('output')) + 1}_output.txt"
        with open(filename, "w") as file:
            file.write(f"Input: {user_input}\n\nOutput: {output}")

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

@app.route('/static/<path:filename>', methods=['GET'])
def get_static_file(filename):
    return send_file(os.path.join('static', filename))

if __name__ == '__main__':
    app.run(debug=True)
