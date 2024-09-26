import openai

# Replace 'your-api-key' with your actual OpenAI API key
openai.api_key = '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'

# List of possible GPT models
models = [
    "gpt-4",
    "gpt-4-0613",
    "gpt-4-32k",
    "gpt-4-32k-0613",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k-0613"
]

# Prompt to test with the models
prompt = "Test prompt to check the model."

# Loop through models and find the first one that works
for model in models:
    try:
        print(f"Trying model: {model}")
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        print(f"Model '{model}' works and gives the following output:\n")
        print(response.choices[0].message["content"])
        break
    except Exception as e:
        print(f"Model '{model}' did not work. Error: {str(e)}")
