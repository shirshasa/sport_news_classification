import pandas as pd
import io


def df2str(df):
    header = ",".join(df.columns.tolist()) + "\n"
    s = [header, ]
    for index, oid, text in df.itertuples():
        s.append(f"{index},{oid},{text}\n")
    return "".join(s)


def str2df(s) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(s), sep=",", index_col=0)


def list2df(d: list) -> pd.DataFrame:
    return pd.DataFrame(d)
