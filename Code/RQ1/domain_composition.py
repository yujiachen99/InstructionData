import random
import json

code_file = "../../Datas/data-oss_instruct-decontaminated.jsonl"
math_file = "../../Datas/MathInstruct.json"

code_data = []
math_data = []
with open(code_file,"r",encoding='utf-8') as f_code, open(math_file,"r",encoding='utf-8') as f_math:
    for line in f_code:
        data = json.loads(line)
        code_data.append(data)

    math_data = json.load(f_math)

code_samples = random.sample(code_data, int(0.9*len(code_data))) # problem / solution
math_samples = random.sample(math_data, int(0.1*len(code_data))) # instruction / output

all_samples = []

for data in code_samples:
    all_samples.append({"instruction":data["problem"],"input":"","output":data["solution"]})

for data in math_samples:
    all_samples.append({"instruction":data["instruction"],"input":"","output":data["output"]})

print(len(all_samples))

json.dump(all_samples,open("OSS/RQ1_OSS_MATH_9_1.json","w",encoding="utf-8"),indent=4,ensure_ascii=False)

general_file = "../../Datas/alpaca.jsonl"

general_data = []
with open(general_file,"r",encoding='utf-8') as f_general:
    for line in f_general:
        data = json.loads(line) 
        general_data.append(data)

general_samples = random.sample(general_data,int(0.1*len(code_data))) # instruction / output

all_samples = []

for data in code_samples:
    all_samples.append({"instruction":data["problem"],"input":"","output":data["solution"]})

for data in general_samples:
    all_samples.append({"instruction":data["instruction"],"input":"","output":data["output"]})

print(len(all_samples))

json.dump(all_samples,open("OSS/RQ1_OSS_GENERAL_9_1.json","w",encoding="utf-8"),indent=4,ensure_ascii=False)
