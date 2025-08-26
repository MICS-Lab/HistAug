import pandas as pd

train = pd.read_csv("train.csv")
val = pd.read_csv("val.csv")
test = pd.read_csv("test.csv")

train_cases = set(train["case_id"])
val_cases = set(val["case_id"])
test_cases = set(test["case_id"])

# Find overlaps
overlap_train_val = train_cases & val_cases
overlap_train_test = train_cases & test_cases
overlap_val_test = val_cases & test_cases

if overlap_train_val or overlap_train_test or overlap_val_test:
    print("Data leakage detected!")
    if overlap_train_val:
        print("Train-Val overlap:", overlap_train_val)
    if overlap_train_test:
        print("Train-Test overlap:", overlap_train_test)
    if overlap_val_test:
        print("Val-Test overlap:", overlap_val_test)
else:
    print("No data leakage: case_ids are unique across splits.")
