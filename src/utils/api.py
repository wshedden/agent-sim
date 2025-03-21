from google import genai
from pathlib import Path

# Function to read the API key from the 'API_KEY' file
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

# Read the API key
api_key = read_api_key()

# Initialize the client with the API key
client = genai.Client(api_key=api_key)

# Define your prompt
prompt = "The quick brown fox jumps over the lazy dog."

# Count tokens in the prompt
token_count_response = client.models.count_tokens(
    model="gemini-2.0-flash",
    contents=prompt
)
total_tokens = token_count_response.total_tokens
print("Total tokens in prompt:", total_tokens)

# Generate content based on the prompt
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt
)

# Access usage metadata to get detailed token counts
usage_metadata = response.usage_metadata
print("Prompt token count:", usage_metadata.prompt_token_count)
print("Candidates token count:", usage_metadata.candidates_token_count)
print("Total token count:", usage_metadata.total_token_count)

# Print the generated response
print("Generated response:", response.text)
