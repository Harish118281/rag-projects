from dotenv import load_dotenv
from embedchain import App

load_dotenv()

app = App.from_config(config_path="config.yaml")

app.add("documents/Control_system.pdf")

while True:
    question = input("\nAsk a question (type 'exit' to quit): ")

    if question.lower() == "exit":
        break

    answer = app.query(question)

    print("\nAnswer:")
    print(answer)

