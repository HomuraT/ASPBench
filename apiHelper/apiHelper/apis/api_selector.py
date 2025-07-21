from apiHelper.apis.anthropic_api import call_anthropic_api
from apiHelper.apis.openai_api import call_openai_api


def select_call_api(model_name):
    # if 'claude' in model_name:
    #     return call_anthropic_api
    # else:
    return call_openai_api