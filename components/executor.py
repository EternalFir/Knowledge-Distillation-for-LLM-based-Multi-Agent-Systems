from data_gen.Chatbot import OpenAIChatbot

import json
import re


class Executor():

    def __init__(self, model_name, args):
        self.model_name = model_name
        self.model = OpenAIChatbot(model_name)
        self.args = args
        self.messages = []
        self.system_prompt = f"""
        # Role
        You are now an Executor. You will be given a task, a list of objects and a list of actions you can do.
        Your task is to give out a list of actions to act.
        """

    def get_area(self, areas):
        self.get_area_promot = f"""
        Here are some different kinds of tasks:
        {areas}
        
        Please give out the one that you think you are good at.
        Remember, you can only give out one area choice.
        
        ** Your output has to be one of the areas I give you.
        ** Do not output anything more than the name of the area you choose.
        """
        response, self.messages = self.model.chat(messages=self.messages, user_input=self.get_area_promot,system_input=self.system_prompt, tempurature=0.0)
        return response

    def init_execute(self, sub_task, objects_list, actions_list,prev_sub_task_conclusion):
        self.init_execute_promot = f"""
            # Task
            Your task is:
            {sub_task}
            
            # Previous Conclusion
            If you are not the first one to execute the task, here is the previous conclusion:
            {prev_sub_task_conclusion} 
            
            # Possible Actions
            Here are all actions you can do:
            {actions_list}
            
            # Objects
            Here are all objects you have:
            {objects_list}
            
            # Hints
            ** If the environment gives you instructs about actions formats, please follow that. 
            ** Remember, you can only give out one action each time.
            ** If the environment told you "Unknown action" that means your output is not in the action list I gave you.
            ** If you forget about legal actions, you can just output "help".
            ** If the environment told you "Ambiguous request" that means your action is not clear enough. You can just choose one number the environment give to you for the intended action. But that's only usable for one step.
            
            
            Please give out the next action you want to do.
            """

        response, self.messages = self.model.chat(messages=self.messages, user_input=self.init_execute_promot,
                                                  tempurature=0.0)
        return response

    def execute(self, env_response):
        self.execute_promot = f"""
            The environment response for your last action is:
            {env_response}
            Please give out the next action you want to do.
            You only need to give out one action each time and DO not give out anything more than the action.
            Do not give out any notes or captions.
            If you think you've finished the task, you should firstly give out "I'm done". Then give out a brief conclusion of what you'd done and what you'd got.
            Your conslusion should directly start with "I'm done. " and then followed by your conclusion. Do not have any caption before "I'm done. ".
            """

        response, self.messages = self.model.chat(messages=self.messages, user_input=self.execute_promot,
                                                 tempurature=0.0)
        return response


