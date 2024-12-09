from langchain_core.prompts import ChatPromptTemplate
from nested import *
from graph import *
from langchain_google_genai import ChatGoogleGenerativeAI
from userTranscript import formatted
import getpass
import os
from dotenv import load_dotenv

load_dotenv()


# os.environ[GOOGLE_API_KEY] = getpass.getpass({GOOGLE_API_KEY}: os.loadenv())

# def _set_env(var: str):
#     if not os.environ.get(var):
#         os.environ[var] = getpass.getpass(f"{var}: ")


# _set_env("GOOGLE_API_KEY")


# print()
def invokeExtraction():
    tools = [TranscriptSummary]


    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", api_key=os.getenv("GOOGLE_API_KEY"))

    bound_llm = bind_validator_with_retries(
        llm,
        tools=tools,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Respond directly using the TranscriptSummary function."),
            ("placeholder", "{messages}"),
        ]
    )

    chain = prompt | bound_llm

    try:
        results = chain.invoke(
            {
                "messages": [
                    (
                        "user",
                        f"Extract the summary from the following conversation:\n\n<convo>\n{formatted}\n</convo>"
                        "\n\nRemember to respond using the TranscriptSummary function.",
                    )
                ]
            },
        )
        results.pretty_print()
    except ValueError as e:
        print(repr(e))


invokeExtraction()