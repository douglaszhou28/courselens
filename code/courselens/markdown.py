import pandas as pd


def df_to_markdown(df: pd.DataFrame, index: bool = False) -> str:
    if df is None or len(df) == 0:
        return ""
    work = df.copy()
    if index:
        work = work.reset_index()
    cols = [str(c) for c in work.columns]
    rows = []
    for _, row in work.iterrows():
        vals = []
        for v in row.tolist():
            if pd.isna(v):
                vals.append("")
            elif isinstance(v, float):
                vals.append(f"{v:.4g}")
            else:
                vals.append(str(v))
        rows.append(vals)
    def esc(x):
        return str(x).replace("|", "\\|")
    out = ["| " + " | ".join(map(esc, cols)) + " |", "|" + "|".join(["---" for _ in cols]) + "|"]
    out += ["| " + " | ".join(map(esc, r)) + " |" for r in rows]
    return "\n".join(out)
