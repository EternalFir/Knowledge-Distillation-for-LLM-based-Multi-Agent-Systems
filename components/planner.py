from data_gen.Chatbot import OpenAIChatbot

import json
import re

# self.planner_init_promot = f"""
#             # Role
#             You are now a planner. You will be given a big task and you are required to break it down into smaller tasks.
#             # Task
#             The task is {task_name}. Here is the task description:
#
#             {task_description}
#
#             The environment around is {observation}
#             {f'You are in a {environment}.' if environment else ''}Your task is to select a person to be {role} with.
#
#             # Input
#             The input is a list of dictionaries.
#
#             The profiles are given below after chevrons:
#
#             <PROFILES>
#             {json.dumps(candidates)}
#             </PROFILES>
#
#             # Output
#             The output should be given in JSON format with the following structure
#
#             {{
#                 "name" : name of the person you selected,
#                 "reason" : reason for selecting the person
#             }}
#
#             # Notes
#
#             * The name of the person you selected must be one of the names in the input.
#             * Your output must be JSON only.
#
#             ```json
#             """

class Planner():

    def __init__(self,model_name,args):
        self.model_name=model_name
        self.model=OpenAIChatbot(model_name)
        self.args=args
        self.messages=[]



    def init_planning(self,task_name,task_description,observation):
        self.planner_init_promot=f"""
            # Role
            You are now a planner. You will be given a big task and you are required to break it down into smaller sub-tasks.
            # Task
            The task is {task_name}. Here is the task description:

            {task_description}

            When looking around, you can see
            {observation}

            # Output
            Each sub-task should be given in JSON format with the following structure

            {{
                "order" : The order of the sub-task, must be a number,
                "sub-task" : detailed description of the sub-task
            }}

            # Notes
            * In the last step, you should focus on the item you created or found to submit it to the environment. Remember, you should focus on the item you want itself, not its container or its location or something else.
            * Your output must be JSON only.

            ```json
            """
        response,self.messages=self.model.chat(messages=self.messages,user_input="Please give out your plan.",system_input=self.planner_init_promot,tempurature=0.0)
        try:
            sub_tasks = json.loads(re.sub(r'^```json\n|\n```$', '', response))
            return sub_tasks
        except json.JSONDecodeError as e:
            print("Failed to parse JSON:", e)
            print("Response content:", repr(response))
            exit(1)

    def replanning(self,advice):
        response, self.messages = self.model.chat(messages=self.messages, user_input=(advice+"Please plan again."))
        try:
            sub_tasks = json.loads(re.sub(r'^```json\n|\n```$', '', response))
            return sub_tasks
        except json.JSONDecodeError as e:
            print("Failed to parse JSON:", e)
            print("Response content:", repr(response))
            exit(1)


