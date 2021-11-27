attr_names = ["id", "ccf", "age", "sex", "painloc", "painexer", "relrest", "pncaden", "cp", "trestbps", "htn", "chol", "smoke", "cigs", "years", "fbs", "dm", "famhist", "restecg", "ekgmo", "ekgday", "ekgyr", "dig", "prop", "nitr", "pro", "diuretic", "proto", "thaldur", "thaltime", "met", "thalach", "thalrest", "tpeakbps", "tpeakbpd", "trestbpd", "dummy", "exang",
              "xhypo", "oldpeak", "slope", "rldv5", "rldv5e", "ca", "restckm", "exerckm", "restef", "restwm", "exeref", "exerwm", "thal", "thalsev", "thalpul", "earlobe", "cmo", "cday", "cyr", "num", "lmt", "ladprox", "laddist", "diag", "cxmain", "ramus", "om1", "om2", "rcaprox", "rcadist", "lvx1", "lvx2", "lvx3", "lvx4", "lvf", "cathef", "junk", "name"]
print(len(attr_names))
files = ["datasets/cleveland.data", "datasets/hungarian.data",
         "datasets/long-beach-va.data", "datasets/switzerland.data"]
for file in files:
    with open (file, errors="ignore") as data:
        newFileName = file.split(".")[0] + "_prep.data"
        newFile = open (newFileName, "w")
        newFile.write(" ".join(attr_names) +"\n")
        current_text = ""
        for line in data:
            current_text+= line
            if "name" in current_text:
                newFile.write(current_text)
                current_text = ""
            else:
                current_text = current_text.strip() + " "
        newFile.close()
