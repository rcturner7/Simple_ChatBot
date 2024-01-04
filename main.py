import nltk
nltk.download('vader_lexicon')
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import requests
from nltk.sentiment import SentimentIntensityAnalyzer
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Initialize model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("./fine_tuned_model")

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()


# Function to get weather data
def get_weather(city, api_key="YOUR_API_KEY_HERE"):
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}appid={api_key}&q={city}"
    response = requests.get(complete_url)
    return response.json()


def get_bot_response(user_input):
    global chat_history_ids

    # Sentiment analysis
    sentiment = sia.polarity_scores(user_input)
    if sentiment['compound'] < -0.5:
        return "You seem upset. How can I help you better?"

    # Check for weather query
    if "weather" in user_input.lower():
        city = user_input.split()[-1]  # simplistic way to get city name
        weather_data = get_weather(city)
        temp = weather_data['main']['temp'] - 273.15  # Convert Kelvin to Celsius
        return f"The current temperature in {city.title()} is {temp:.2f}°C."

    # Tokenize and encode the input phrase
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Append the new user input tokens to the chat history
    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
    else:
        bot_input_ids = new_user_input_ids

    # Generate a response
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode and return the model's output
    return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)


def main():
    print("Welcome to the Chatbot! Type 'bye' to exit.")
    global chat_history_ids
    chat_history_ids = None

    while True:
        user_input = input("You: ")
        if "bye" in user_input.lower():
            print("Bot: Goodbye! Have a great day!")
            break
        response = get_bot_response(user_input)
        print("Bot:", response)


if __name__ == "__main__":
    main()
