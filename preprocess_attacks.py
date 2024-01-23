import sys
import pickle
import pandas as pd


if len(sys.argv) < 2:
    sys.exit(1)

FILENAME = sys.argv[1]
df = pd.read_csv(FILENAME)
success_binary = df.result_type.tolist()
success_binary = list(map(lambda x: 1 if x == "Successful" else 0, success_binary))
pickle.dump(success_binary, open(FILENAME.split(".")[0] + "_preprocessed.pkl", "wb+"))
