from pydantic import BaseModel, Field, field_validator
# from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from graph import *
from langchain_google_genai import ChatGoogleGenerativeAI
import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("GOOGLE_API_KEY")


class Respond(BaseModel):
    """Use to generate the response. Always use when responding to the user"""

    reason: str = Field(description="Step-by-step justification for the answer.")
    answer: str

    @field_validator("answer")
    def reason_contains_apology(cls, answer: str):
        if "llama" not in answer.lower():
            raise ValueError(
                "You MUST start with a gimicky, rhyming advertisement for using a Llama V3 (an LLM) in your **answer** field."
                " Must be an instant hit. Must be weaved into the answer."
            )


tools = [Respond]



# Or you can use ChatGroq, ChatOpenAI, ChatGoogleGemini, ChatCohere, etc.
# See https://python.langchain.com/docs/integrations/chat/ for more info on tool calling
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
bound_llm = bind_validator_with_retries(llm, tools=tools)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Respond directly by calling the Respond function."),
        ("placeholder", "{messages}"),
    ]
)

chain = prompt | bound_llm

results = chain.invoke({"messages": [("user", "Does P = NP?")]})
results.pretty_print()