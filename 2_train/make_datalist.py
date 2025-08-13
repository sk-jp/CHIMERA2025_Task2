import glob
import json

phase = "train" 
#phase = "valid"

train_ratio = 0.8

topdir = "/data/MICCAI2025_CHIMERA/task2/data"
subdirs = sorted(glob.glob(f"{topdir}/2*"))

print("case,age,sex,smoking,tumor,stage,substage,grade,reTUR,LVI,variant,EORTC,no_instillations,BRS")

keys = ["age", "sex", "smoking", "tumor",
        "stage", "substage", "grade", "reTUR", "LVI",
        "variant", "EORTC", "no_instillations", "BRS"]

cd_datas = [[], [], []]
case_names = [[], [], []]
for subdir in subdirs:
    case_name = subdir.split("/")[-1]
    cd_file = f"{subdir}/{case_name}_CD.json"
    cd_data = json.load(open(cd_file, "r"))

    if cd_data["BRS"] == "BRS1":
        cd_datas[0].append(cd_data)
        case_names[0].append(case_name)
    elif cd_data["BRS"] == "BRS2":
        cd_datas[1].append(cd_data)
        case_names[1].append(case_name)
    elif cd_data["BRS"] == "BRS3":
        cd_datas[2].append(cd_data)
        case_names[2].append(case_name)

# count the number of cases in each BRS
num_cases = [len(case_name) for case_name in case_names]

# print the list of cases
for i in range(3):
    num_train_cases = int(len(cd_datas[i]) * train_ratio)
    if phase == "train":
        ids = list(range(num_train_cases))
    elif phase == "valid":
        ids = list(range(num_train_cases, len(cd_datas[i])))

    for id in ids:
        case_name = case_names[i][id]
        s = f"{case_name}"
        cd_data = cd_datas[i][id]
        for key in keys:
            if key in cd_data:
                s += f",{cd_data[key]}"
            else:
                s += ","
        print(s)

