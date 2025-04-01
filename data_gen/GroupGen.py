import json

from scienceworld import ScienceWorldEnv
from Chatbot import Chatbot
from components import planner
from components import evaluator
from components import allocator
from components import executor

import argparse
import time
import os
import random
import ast

exitCommands = ["quit", "exit"]


def SingleTaskGen(args):
    env = ScienceWorldEnv("", args.jar_path, envStepLimit=args.env_step_limit)

    taskNames = env.get_task_names()
    task_name = taskNames[args.task_ID]
    for var_id in range(args.var_num):
        for episode in range(args.num_episodes):
            output_dir = "./data/" + args.model_name + "/task_" + str(args.task_ID) + "/var_" + str(var_id) + "/"+str(episode)+"/"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            summary_name = output_dir + "running_log.txt"
            summary_file = open(summary_name, "w")
            plan_agent = planner.Planner(args.model_name, args, )
            evaluate_agent = evaluator.Evaluator(args.model_name, args, )
            allocate_agent = allocator.Allocator(args.model_name, args, )
            executor_agents = [executor.Executor(args.model_name, args, ) for _ in range(3)]
            env_running_msg = []
            total_reward = 0
            env.load(task_name, var_id, "", generateGoldPath=True)
            print(env.get_task_description())
            print(env.get_gold_action_sequence())
            # observation=env.look()
            # print("look: " + str(observation))
            # possible_actions=env.get_possible_actions()
            # print("Possible actions: " + str(possible_actions))
            sub_tasks = plan_agent.init_planning(task_name, env.get_task_description(), env.look())
            evaluation = evaluate_agent.init_evaluate(task_name, env.get_task_description(), env.look(),
                                                      env.get_possible_actions(), sub_tasks)
            print("evaluation: " + evaluation, file=summary_file)
            evaluation = ast.literal_eval(evaluation)
            while evaluation[0] != "Yes":
                advice = evaluation[1]
                sub_tasks = plan_agent.replanning(advice)
                evaluation = evaluate_agent.reevaluate(sub_tasks)
                print("evaluation: " + evaluation, file=summary_file)
                evaluation = ast.literal_eval(evaluation)
            print("sub_tasks", file=summary_file)
            for sub_task in sub_tasks:
                print(sub_task, file=summary_file)
            areas = allocate_agent.area_des(sub_tasks)
            print("areas: " + areas, file=summary_file)
            areas = ast.literal_eval(areas)
            prefer_area = []
            print("find area", file=summary_file)
            for agent in executor_agents:
                area = agent.get_area(areas)
                print(area, file=summary_file)
                areas.remove(area)
                prefer_area.append(area)
            print(prefer_area, file=summary_file)
            agent_areas = {}
            for i in range(len(prefer_area)):
                agent_areas["agent " + str(i)] = prefer_area[i]
            allocation = allocate_agent.allocate(sub_tasks, agent_areas)
            print(allocation, file=summary_file)
            allocation = ast.literal_eval(allocation)
            prev_sub_task_conclusion = ""
            env_step_cnt = 0
            for i in range(len(sub_tasks)):
                print("i: ", i, file=summary_file, flush=True)
                sub_task = sub_tasks[i]
                print("sub_task: ", sub_task, file=summary_file)
                allocation_agent = executor_agents[int(allocation[i][-1])]
                objects_list = env.get_possible_objects()
                actions_list = env.get_possible_actions()
                response = allocation_agent.init_execute(sub_task, objects_list, actions_list, prev_sub_task_conclusion)
                print("response: ", response, file=summary_file, flush=True)
                while not response.startswith("I'm done"):
                    env_step_cnt += 1
                    if env_step_cnt > args.env_step_limit:
                        break
                    env_response, single_step_reward, terminated, look = env.step(response)
                    total_reward += single_step_reward
                    env_running_msg.append([response, [env_response, single_step_reward, terminated, look]])
                    print("env_response: ", env_response, file=summary_file)
                    print("reward: ", single_step_reward, file=summary_file, flush=True)
                    # print(env_response)
                    # print(len(env_response))
                    # exit(0)
                    response = allocation_agent.execute(env_response)
                    print("response: ", response, file=summary_file, flush=True)
                if env_step_cnt > args.env_step_limit:
                    break
                print(response)
                if response.startswith("I'm done. "):
                    prev_sub_task_conclusion = response[len("I'm done. "):]
                else:
                    prev_sub_task_conclusion = response[len("I'm done"):]
            plan_log = open(output_dir + "plan_log.txt", "w")
            evaluate_log = open(output_dir + "evaluate_log.txt", "w")
            allocate_log = open(output_dir + "allocate_log.txt", "w")
            execute_logs = [open(output_dir + "execute_log_" + str(i) + ".txt", "w") for i in range(3)]
            env_running_log = open(output_dir + "env_running_log.txt", "w")
            json.dump(plan_agent.messages, plan_log)
            json.dump(evaluate_agent.messages, evaluate_log)
            json.dump(allocate_agent.messages, allocate_log)
            for i in range(3):
                json.dump(executor_agents[i].messages, execute_logs[i])
            json.dump(env_running_msg, env_running_log)
            print("total_reward: ", total_reward, file=summary_file,flush=True)
            plan_log.close()
            evaluate_log.close()
            allocate_log.close()
            for log in execute_logs:
                log.close()
            env_running_log.close()
            env.reset()

    # for var_id in range(args.var_num):
    #     prev_episode_summary_str = ""
    #     for episodeIdx in range(0, args.num_episodes):
    #         print(env.get_task_description())
    #         env.load(task_name, var_id, args.simplification_str,generateGoldPath=True)
    #         print(env.get_task_description())
    #         score = 0.0
    #         score_positive = 0.0
    #         is_complete = False
    #         main_step = 0
    #         rawActionHistory = [""]
    #         rationaleHistory = [""]
    #         observationHistory = [""]
    #         subgoalHistory = [env.get_goal_progress()]
    #

    return


def parse_args():
    parser = argparse.ArgumentParser(description='SingleTaskGen')
    parser.add_argument('--task_ID', type=int, default=0)
    parser.add_argument('--task_num', type=int, default=0)
    parser.add_argument('--var_num', type=int, default=1)
    parser.add_argument('--simplification_str', type=str, default="")
    parser.add_argument('--num_episodes', type=int, default=1)
    parser.add_argument('--jar_path', type=str, default="")
    parser.add_argument('--env_step_limit', type=int, default=100)
    parser.add_argument('--model_name', type=str, default="deepseek-reasoner")
    parser.add_argument('--log_dir', type=str, default="./data/api_logs/")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    SingleTaskGen(args)
