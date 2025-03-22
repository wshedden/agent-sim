import requests
from pathlib import Path

def read_api_key():
    try:
        # Determine the path to the 'API_KEY' file in the same directory as this script
        script_dir = Path(__file__).resolve().parent
        api_key_path = script_dir / 'API_KEY'

        # Read the API key from the file
        with open(api_key_path, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        raise Exception(f"API key file '{api_key_path}' not found.")
    except Exception as e:
        raise Exception(f"An error occurred while reading the API key: {e}")

def generate_content(prompt):
    api_key = read_api_key()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    headers = {
        'Content-Type': 'application/json'
    }
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "maxOutputTokens": 50
        }
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

if __name__ == "__main__":
    prompt = "Explain different cigarette rolling techniques"
    try:
        response = generate_content(prompt)
        print(response)
    except Exception as e:
        print(e)