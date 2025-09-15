# from daytona import Daytona, DaytonaConfig
  
# # Define the configuration
# config = DaytonaConfig(api_key="dtn_319beb76f2a0d90a1b59115617e2018780eb4ccf773688c12acd3cee3b53102d")

# # Initialize the Daytona client
# daytona = Daytona(config)

# # Create the Sandbox instance
# sandbox = daytona.create()

# # Run the code securely inside the Sandbox
# response = sandbox.process.code_run('print("Hello World from code!")')
# if response.exit_code != 0:
#   print(f"Error: {response.exit_code} {response.result}")
# else:
#     print(response.result)
  
from dotenv import load_dotenv

load_dotenv()

from google import genai

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash", contents="Explain how AI works in a few words"
)
print(response.text)