# import litellm
# import openai
from openai import OpenAI
from litellm import completion
import os
import json
import time
from transformers import pipelines
# import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# from trl import get_kbit_device_map

environment = ""
role = ""

candidates = []

init_prompt = f"""
    # Task
    {f'You are in a {environment}.' if environment else ''}Your task is to select a person to be {role} with.

    # Input
    The input is a list of dictionaries. 

    The profiles are given below after chevrons:

    <PROFILES>
    {json.dumps(candidates)}
    </PROFILES>

    # Output
    The output should be given in JSON format with the following structure

    {{
        "name" : name of the person you selected,
        "reason" : reason for selecting the person
    }}

    # Notes

    * The name of the person you selected must be one of the names in the input.
    * Your output must be JSON only.

    ```json
    """


class Chatbot():
    def __init__(self, model_name, model_path=''):
        # client = litellm.Client()
        self.last_chat_time = None
        print("Now using " + model_name)
        if model_name.startswith("groq"):
            with open('groq_api_key.txt', 'r') as file:
                api_key = file.read().strip()
            os.environ['GROQ_API_KEY'] = api_key
            if api_key is not None:
                print("API key set")
            else:
                print("API key not set")
            pass
        elif model_name.startswith("gpt"):
            with open('openai_api_key.txt', 'r') as file:
                api_key = file.read().strip()
            os.environ['OPENAI_API_KEY'] = api_key
            if api_key is not None:
                print("API key set")
            else:
                print("API key not set")
            pass
        elif model_name.startswith("deepseek"):
            with open("api-keys/DeepSeek", 'r') as file:
                api_key = file.read().strip()
            os.environ['DEEPSEEK_API_KEY'] = api_key
            if api_key is not None:
                print("API key set")
            else:
                print("API key not set")
            pass

        # else:
        #     model_name = model_path + model_name
        #     with open('huggingface_api_key.txt', 'r') as file:
        #         api_key = file.read().strip()
        #     torch.cuda.empty_cache()
        #     self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=api_key, max_new_token=1024, use_fast=True)
        #     self.model_name = AutoModelForCausalLM.from_pretrained(model_name, token=api_key, torch_dtype=torch.float32,
        #                                                       device_map=get_kbit_device_map())
        #     self.tokenizer.padding_side = "left"

    def _get_response(self, message, model_name, system_message='', temperature=0.5):
        if model_name.startswith("gpt"):
            client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
            )
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": message,
                    }
                ],
                model=model_name,
                temperature=temperature,
            )
        elif model_name.startswith("groq"):
            response = completion(
                model=model_name,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": message}
                ],
            )
        elif model_name.startswith("deepseek"):
            client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": message},
                    {"role": "user", "content": message},
                ],
                stream=True
            )
            # print(response.choices[0].message.content)
        else:
            messages = [[{"role": "user", "content": message}]]
            input_ids = self.tokenizer.apply_chat_template(messages, padding=True, return_tensors='pt',
                                                           add_generation_prompt=True).to(self.model.device)
            output_ids = self.model.generate(input_ids, max_new_tokens=1024, temperature=temperature)
            response = self.tokenizer.decode(output_ids[0]).replace("<pad>", "").replace("<eos>", "")
            start = response.find("<end_of_turn>\n<start_of_turn>model_name\n") + len(
                "<end_of_turn>\n<start_of_turn>model_name\n")
            response = response[start:]
            end = response.find("<end_of_turn>")
            response = response[:end]
            # print(response)
        return response

    def chat(self, message, model_name="groq/llama3-8b-8192", system_message='', tempurature=0.5):
        return self._get_response(message, model_name, system_message, tempurature)

    def chat_limit(self, message, tempurature=0.5, model_name="groq/llama3-8b-8192"):
        wait_time = 300
        if self.last_chat_time is not None:
            elapsed_time = time.time() - self.last_chat_time
            if elapsed_time < wait_time:
                wait_time = wait_time - elapsed_time
                time.sleep(wait_time)
        self.last_chat_time = time.time()
        return self._get_response(message, tempurature, model_name)


class OpenAIChatbot():
    def __init__(self, model_name, model_path=''):
        # client = litellm.Client()
        self.last_chat_time = None
        self.model_name = model_name
        print("Now using " + model_name)
        if model_name.startswith("gpt"):
            with open('openai_api_key.txt', 'r') as file:
                api_key = file.read().strip()
            os.environ['OPENAI_API_KEY'] = api_key
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), )
            if api_key is not None:
                print("API key set")
            else:
                print("API key not set")
            pass
        elif model_name.startswith("deepseek"):
            with open("api-keys/DeepSeek", 'r') as file:
                api_key = file.read().strip()
            os.environ['DEEPSEEK_API_KEY'] = api_key
            self.client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
            if api_key is not None:
                print("API key set")
            else:
                print("API key not set")
            pass

    def chat(self, messages, user_input, system_input='', tempurature=0.5):
        if system_input != '':
            messages.append({"role": "system", "content": system_input})
        messages.append({"role": "user", "content": user_input})
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=tempurature,
            stream=False
        )
        # assistant_reply = ""
        # for chunk in response:
        #     if chunk.choices:
        #         delta = chunk.choices[0].delta
        #         content = getattr(delta, "content", None)  # 获取 content，默认值为 None
        #         if content:  # 只有在 content 不是 None 时才拼接
        #             print("------")
        #             print(content)
        #             print("------")
        #             assistant_reply += content
        # return assistant_reply
        assistant_reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_reply})  # 记录 AI 回复

        return assistant_reply,messages
