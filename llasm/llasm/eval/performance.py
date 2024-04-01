import json
def eval_classification(predict, label, boundary = 91):
    tp = 0
    fp = 0
    fn = 0
    for item in label:
        if item < boundary:
            fn += 1
        else:
            tp += 1

    for item in predict:
        if item < boundary:
            fp += 1

    P = tp / (tp + fp)
    R = tp / (tp + fn)
    F1 = 2 * P * R / (P + R)
    return P, R, F1

def check_is_equal(OA_names, OB_names):
    count = 0
    for item_A in OA_names:
        for item_B in OB_names:
            if item_A in item_B or item_B in item_A:
                count += 1
                continue
    if count >= min(len(OA_names), len(OB_names)) / 3:
        return True
    else:
        return False

def eval_optimization(OA, OB):
    pool = []
    for name in OA:
        if name in OB:
            pool.append(name) 
    match = 0
    for name in pool:
        OA_names = OA[name].split("_")
        OB_names = OB[name].split("_")
        if check_is_equal(OA_names, OB_names):
            match += 1
    return match / len(pool)

if __name__ == "__main__":
    print("performance in test set.")
    path = "./save/dataset/"       
    for file in ["overall", "eval_O0", "eval_O1", "eval_O2", "eval_O3"]:
        label = []
        predict = []
        lines = open(path + file + ".json").readlines()
        for line in lines:
            data = json.loads(line)
            label.extend(data["label_list"])
            predict.extend(data["predict_list"])
        for boundary in [100, 99, 98, 96, 91]:  
            P, R, F1 = eval_classification(predict, label, boundary)
            print(f"{file} in rank={101-boundary}: Precision:{P}, Recall:{R}, F1 score:{F1}")


    print("performance in malware analysis.")
    path = "./save/mirai/"       
    for file in ["mirai"]:
        label = []
        predict = []
        lines = open(path + file + ".json").readlines()[:60]
        for line in lines:
            data = json.loads(line)
            label.extend(data["label_list"])
            predict.extend(data["predict_list"])
        for boundary in [100, 99, 98, 96, 91]:  
            P, R, F1 = eval_classification(predict, label, boundary)
            print(f"{file} in rank={101-boundary}: Precision:{P}, Recall:{R}, F1 score:{F1}")

    print("performance in cross optimization.")
    path = "./save/dataset/"
    for score in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        eval_O0 = {}
        eval_O1 = {}
        eval_O2 = {}
        eval_O3 = {}
        for file in ["eval_O0", "eval_O1", "eval_O2", "eval_O3"]:
            label = []
            predict = []
            lines = open(path + file + ".json").readlines()         
            for line in lines:
                data = json.loads(line)
                if file == "eval_O0" and data["score"] > score:
                    eval_O0[data["name"]] = data["predict"].lower()
                if file == "eval_O1" and data["score"] > score:
                    eval_O1[data["name"]] = data["predict"].lower()
                if file == "eval_O2" and data["score"] > score:
                    eval_O2[data["name"]] = data["predict"].lower()
                if file == "eval_O3" and data["score"] > score:
                    eval_O3[data["name"]] = data["predict"].lower()

        match = eval_optimization(eval_O0, eval_O1)
        print(f"{match} functions are matched cross O0 and O1 above score = {score}")
        match = eval_optimization(eval_O0, eval_O2)
        print(f"{match} functions are matched cross O0 and O2 above score = {score}")
        match = eval_optimization(eval_O0, eval_O3)
        print(f"{match} functions are matched cross O0 and O3 above score = {score}")
        match = eval_optimization(eval_O1, eval_O2)
        print(f"{match} functions are matched cross O1 and O2 above score = {score}")
        match = eval_optimization(eval_O1, eval_O3)
        print(f"{match} functions are matched cross O1 and O3 above score = {score}")
        match = eval_optimization(eval_O2, eval_O3)
        print(f"{match} functions are matched cross O2 and O3 above score = {score}")
        print()

