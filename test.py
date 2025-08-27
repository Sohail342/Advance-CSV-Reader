import pandas as pd
import requests

df = pd.read_csv("JXxWxW.csv")

# get gl_code and branch_code columns
gl_branch = df[["sub_gl_code", "branch_code"]]

for row in gl_branch.itertuples():
    response = requests.get(
        "http://localhost:8000/search_existing",
        params={"sub_gl_codes": row.sub_gl_code, "branch_codes": row.branch_code},
    )
    print(response.json())
