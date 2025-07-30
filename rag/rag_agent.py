# rag_agent.py

class RagAgent:
    def __init__(self, model):
        self.model = model

    def generate_response(self, input_text):
        return self.model.predict(input_text)

    def process_input(self, input_text):
        cleaned_text = self.clean_input(input_text)
        return self.generate_response(cleaned_text)

    def clean_input(self, input_text):
        return input_text.strip().lower()  # Simplified cleaning

# Example usage
if __name__ == '__main__':
    agent = RagAgent(model='your_model_here')
    response = agent.process_input('  Hello World!  ')
    print(response)