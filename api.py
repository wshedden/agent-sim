from google import genai

# Initialize the client with your API key
client = genai.Client(api_key="YOUR_API_KEY")

# Define your prompt
prompt = "The quick brown fox jumps over the lazy dog."

# Count tokens using the client's count_tokens method
token_count_response = client.models.count_tokens(
    model="gemini-2.0-flash",
    contents=prompt
)

# Extract the total token count from the response
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
