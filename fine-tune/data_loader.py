import json
import argparse

def single_data_load(model_name, task_ID, var_id,episode):
    log_dir="./data/"+model_name+"/task_"+str(task_ID)+"/var_"+str(var_id)+"/"+str(episode)+"/"
    with open(log_dir+"plan_log.txt", "r") as f:
        plan_log=json.load(f)
    with open(log_dir+"evaluate_log.txt", "r") as f:
        evaluate_log=json.load(f)
    with open(log_dir+"allocate_log.txt", "r") as f:
        allocate_log=json.load(f)
    with open(log_dir+"env_running_log.txt", "r") as f:
        env_running_log=json.load(f)
    execute_logs=[]
    for i in range(3):
        with open(log_dir+"execute_log_"+str(i)+".txt", "r") as f:
            execute_logs.append(json.load(f))
    return [plan_log, evaluate_log, allocate_log, env_running_log, execute_logs]


def data_loader(args):
    task_history = []
    for var_id in range(args.var_num):
        one_task_history=[]
        for episode in range(args.num_episodes):
            one_task_history.append(single_data_load(args.model_name, args.task_ID, var_id, episode))
        task_history.append(one_task_history)
    return task_history

def parse_args():
    parser=argparse.ArgumentParser(description='DataLoader')
    parser.add_argument('--task_ID',type=int,default=0)
    parser.add_argument('--var_num',type=int,default=0)
    parser.add_argument('--model_name',type=str,default="deepseek-reasoner")
    parser.add_argument('--num_episodes',type=int,default=1)
    return parser.parse_args()

if __name__ == '__main__':
    args=parse_args()
    task_history=data_loader(args)
    print(task_history[0])