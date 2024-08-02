from enum import Enum
from glob import glob
from logging import getLogger
from logging.config import fileConfig
from os.path import join, isfile
from os import environ
from dotenv import load_dotenv

PROJECT_ROOT = ""
LOG_INI = join(PROJECT_ROOT, 'log.ini')
FIXTURES = join(PROJECT_ROOT, 'fixtures')

def load_env():
    load_dotenv(join(PROJECT_ROOT, ".env"))

class PromptTemplate(Enum):
    SYSTEM_PROMPT = "system.txt"
    
class ModelType(str, Enum):
    gpt4o = 'gpt-4o'
    gpt4o_mini = 'gpt-4o-mini'
    embedding = "text-embedding-3-large"


def get_prompt_template(prompt_template: PromptTemplate):
    with open(join(PROJECT_ROOT, "prompt_templates", prompt_template.value), "rt") as f:
        return f.read()


def configure_logging(get_logger=False):
    global logging_configured
    if not logging_configured:
        fileConfig(LOG_INI)
        logging_configured = True
    if get_logger:
        logger = getLogger('freedo')
        return logger
