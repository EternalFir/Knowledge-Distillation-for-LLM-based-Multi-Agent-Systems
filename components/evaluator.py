from data_gen.Chatbot import OpenAIChatbot

import json
import re


class Evaluator():

    def __init__(self, model_name, args):
        self.model_name = model_name
        self.model = OpenAIChatbot(model_name)
        self.args = args
        self.messages = []

    def init_evaluate(self, task_name, task_description, observation, possible_action,sub_tasks):
        self.evaluate_promot = f"""
                    # Role
                    You are now an evaluator. You will be given a big task and someone else had break it down into smaller sub-tasks.
                    Your job is to evaluate that if the sub-tasks are reasonable or not. For example, if the task are too detailed.
                    If not, you need to give out your advice to the planner.

                    # Task
                    The task is {task_name}. Here is the task description:

                    {task_description}

                    # Observation
                    When looking around, you can see
                    {observation}
                    
                    # Actions
                    Here are all the possible actions
                    {possible_action}

                    # Output
                    Your output should be a list with two elements:
                    First element is "Yes" or "No", which means the sub-tasks are reasonable or not.
                    Second element is your advice to the planner. If you think the sub-tasks are reasonable, you can give out "Good job!".
                    ** Your output should be in only one line.

                    """
        self.user_message = f"""
            # sub-tasks
            The sub-tasks are given below:
            {sub_tasks}
            
            Please give out your evaluation.
            ** You should still follow the format requirement.
            """
        response, self.messages = self.model.chat(messages=self.messages, user_input=self.user_message,
                                                  system_input=self.evaluate_promot, tempurature=0.0)
        return response

    def reevaluate(self, sub_tasks):
        self.user_message = f"""
                    # sub-tasks
                    The sub-tasks are now given below:
                    {sub_tasks}

                    Please give out your evaluation again.
                    
                    ** You should still follow the format requirement.
                    """
        response, self.messages = self.model.chat(messages=self.messages, user_input=self.user_message, tempurature=0.0)
        return response
