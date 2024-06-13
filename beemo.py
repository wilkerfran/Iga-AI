import openai
from dotenv import load_dotenv, find_dotenv
import sounddevice as sd
import wave
import os
import numpy as np
import whisper
from langchain_openai import ChatOpenAI
from queue import Queue
import io
import soundfile as sf
import threading
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain.agents import AgentExecutor
import statsmodels.api as sm
import matplotlib.pyplot as plt

load_dotenv(find_dotenv())
client = openai.Client()

class TalkingLLM():
    def __init__(self, model="gpt-3.5-turbo-0613", whisper_size="small"):
        self.is_recording = False
        self.audio_data = []
        self.samplerate = 44100
        self.channels = 1
        self.dtype = 'int16'
        self.selected_dataframe = None
        self.voice_enabled = True

        self.whisper = whisper.load_model(whisper_size)
        self.llm = ChatOpenAI(model=model)
        self.llm_queue = Queue()

        self.csv_files = self.get_csv_files()
        self.df = self.select_dataframe()
        self.create_agent()

    def get_csv_files(self):
        datasets_path = "datasets"
        return [f for f in os.listdir(datasets_path) if f.endswith('.csv')]

    def select_dataframe(self):
        print("Arquivos CSV disponíveis:")
        for idx, file in enumerate(self.csv_files):
            print(f"{idx + 1}: {file}")
        
        file_index = int(input("Escolha o número do arquivo que deseja usar: ")) - 1
        selected_file = self.csv_files[file_index]
        print(f"Arquivo selecionado: {selected_file}")
        
        return pd.read_csv(os.path.join("datasets", selected_file))

    def create_agent(self):
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

        # Adicionando a configuração para lidar com erros de parsing
        self.agent_executor = AgentExecutor(
            agent=self.agent.agent,
            tools=self.agent.tools,
            verbose=True,
            handle_parsing_errors=True
        )

    def save_and_transcribe(self):
        print("Saving the recording...")
        if "temp.wav" in os.listdir():
            os.remove("temp.wav")
        wav_file = wave.open("test.wav", 'wb')
        wav_file.setnchannels(self.channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(self.samplerate)
        wav_file.writeframes(np.array(self.audio_data, dtype=self.dtype))
        wav_file.close()

        result = self.whisper.transcribe("test.wav", fp16=False)
        print("Usuário:", result["text"])

        response_text = self.handle_input(result["text"])
        self.llm_queue.put(response_text)

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

    def convert_and_play(self):
        while True:
            tts_text = self.llm_queue.get()
            if tts_text and self.voice_enabled:
                print("TTS Text:", tts_text)
                
                spoken_response = client.audio.speech.create(
                    model="tts-1",
                    voice='echo',
                    response_format="opus",
                    input=tts_text
                )

                buffer = io.BytesIO()
                for chunk in spoken_response.iter_bytes(chunk_size=4096):
                    buffer.write(chunk)
                buffer.seek(0)

                with sf.SoundFile(buffer, 'r') as sound_file:
                    data = sound_file.read(dtype='int16')
                    sd.play(data, sound_file.samplerate)
                    sd.wait()

    def input_text(self):
        user_input = input("Digite seu texto: ")
        response_text = self.handle_input(user_input)
        self.llm_queue.put(response_text)

    def audio_callback(self, indata, frame_count, time_info, status):
        if self.is_recording:
            self.audio_data.extend(indata.copy())

    def toggle_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.save_and_transcribe()
        else:
            print("Starting recording...")
            self.is_recording = True
            self.audio_data = []

    def run(self):
        t1 = threading.Thread(target=self.convert_and_play)
        t1.start()

        with sd.InputStream(samplerate=self.samplerate, channels=self.channels, dtype=self.dtype, callback=self.audio_callback):
            while True:
                print("\nOpções:")
                print("1: Pressione 'r' para gravar")
                print("2: Pressione 't' para digitar")
                print("3: Pressione 'd' para selecionar um novo banco de dados")
                print("4: Pressione 'v' para ativar/desativar a voz")
                print("5: Pressione 'q' para sair")
                
                user_choice = input("Escolha uma opção: ")
                if user_choice.lower() == 'r':
                    self.toggle_recording()
                elif user_choice.lower() == 't':
                    self.input_text()
                elif user_choice.lower() == 'd':
                    self.df = self.select_dataframe()
                    self.create_agent()
                elif user_choice.lower() == 'v':
                    self.voice_enabled = not self.voice_enabled
                    status = "ativada" if self.voice_enabled else "desativada"
                    print(f"Voz {status}.")
                elif user_choice.lower() == 'q':
                    print("Saindo...")
                    break
                else:
                    print("Opção inválida.")

        t1.join()

if __name__ == "__main__":
    talking_llm = TalkingLLM()
    talking_llm.run()
