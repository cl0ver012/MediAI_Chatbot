from .services import LLMService


def setup_llm():
    # Define file path and template
    llm = LLMService()
    return llm