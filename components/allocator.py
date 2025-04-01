from data_gen.Chatbot import OpenAIChatbot

import json
import re


class Allocator():

    def __init__(self, model_name, args):
        self.model_name = model_name
        self.model = OpenAIChatbot(model_name)
        self.args = args
        self.messages = []
        self.system_prompt = f"""
        # Role
        You are now an allocator. You will be given a series of small tasks and some agents.
        Each agent have a area that they are good at. For example, move or search. 
        Your job is to allocate the tasks to the agents. 
        """
        response, self.messages = self.model.chat(messages=self.messages, user_input="Lets' start.",
                                                  system_input=self.system_prompt, tempurature=0.0)

    def area_des(self, sub_tasks):
        self.area_des_promot = f"""
            # Tasks
            All the tasks are given as follow:
            {sub_tasks}
            
            Please give out a few words, each one best descript some tasks.
            The words you give out needs to follow the following format:
            {["word1", "word2", "word3", "..."]}
            The number of the words should be no more than 3.
            ** Remember, you only need to give out a list of areas. Do not give out any explanation.
            """
        response, self.messages = self.model.chat(messages=self.messages, user_input=self.area_des_promot,
                                                  tempurature=0.0)
        return response

    def allocate(self, sub_tasks, agents_areas):
        self.allocate_promot = f"""
            # Agents
            Here are some agents with their areas which they best at:
            {agents_areas}
            
            # Tasks
            Here are some tasks:
            {sub_tasks}
            
            You are supposed to allocate each task to the agent who is best at it.
            
            You should give out the allocation in the following format:
            ["agent1","agent3","agent2","agent1","..."]
            where the n-value of the list is the agent you allocate the n-th task to.
            ** Remember, you only need to give out a list of agents. Do not give out any explanation.
            """
        response, self.messages = self.model.chat(messages=self.messages, user_input=self.allocate_promot,
                                                  tempurature=0.0)
        return response
