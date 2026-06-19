import ollama

stream = ollama.chat(
    model='hf.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF:latest',
    messages=[{'role': 'user', 'content': 'What is a fact about xbox 360'}],
    stream=True
)
answer = ''
for chunk in stream:
    temp_answer = chunk['message']['content']
    answer = answer + temp_answer
print(answer)
