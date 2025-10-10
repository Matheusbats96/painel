# plot_utils.py
# Gráficos do dashboard (retornam JSON string para o front)

from __future__ import annotations
import json
import pandas as pd

GREEN = ["#004C3F", "#007A63", "#0A8F79", "#13A98F", "#4EC7AB", "#8EDFCC", "#C9EFE4"]
BG = "#FFFFFF"

def _empty():
    return json.dumps({})

def _base_layout_no_title():
    # layout enxuto (sem título interno; usamos o título do card)
    return {
        "paper_bgcolor": BG,
        "plot_bgcolor": BG,
        "margin": {"l": 12, "r": 12, "t": 8, "b": 8},
        "autosize": True,
    }

def create_score_gauge(df: pd.DataFrame) -> str:
    """Gauge meia-lua de 1 a 10 mostrando a média do Score_Previsto."""
    col = "Score_Previsto"
    if col not in df.columns or df[col].dropna().empty:
        return _empty()

    avg = float(df[col].mean())

    data = [{
        "type": "indicator",
        "mode": "gauge+number",
        "value": round(avg, 2),
        "number": {"font": {"size": 34, "color": "#004C3F"}},
        "gauge": {
            "shape": "angular",               # meia-lua
            "axis": {"range": [1, 10], "dtick": 1, "tickwidth": 1},
            "bar": {"color": "#007A63"},
            "steps": [
                {"range": [1, 3],  "color": "#E6F2F0"},
                {"range": [3, 5],  "color": "#C9EFE4"},
                {"range": [5, 7],  "color": "#8EDFCC"},
                {"range": [7, 8.5], "color": "#4EC7AB"},
                {"range": [8.5, 10], "color": "#13A98F"},
            ],
        },
        "domain": {"x": [0, 1], "y": [0, 1]},
    }]

    layout = _base_layout_no_title()
    # margem top um pouco maior para evitar qualquer clip
    layout["margin"] = {"l": 12, "r": 12, "t": 16, "b": 0}

    return json.dumps({"data": data, "layout": layout})

def create_finalidade_pie(df: pd.DataFrame) -> str:
    if df.empty or "Finalidade" not in df.columns:
        return _empty()

    grp = (
        df.groupby("Finalidade", dropna=True)
          .size()
          .reset_index(name="qtd")
          .sort_values("qtd", ascending=False)
    )

    values = [int(v) for v in grp["qtd"].tolist()]

    data = [{
        "type": "pie",
        "labels": grp["Finalidade"].astype(str).tolist(),
        "values": values,
        "textinfo": "label+percent",
        "textposition": "inside",
        "hovertemplate": "%{label}: %{value} operações<extra></extra>",
        "marker": {"colors": GREEN},
        "hole": 0.15
    }]

    layout = _base_layout_no_title()
    # legenda embaixo, para nunca cortar à direita
    layout["legend"] = {"orientation": "h", "y": -0.08, "x": 0.5, "xanchor": "center"}

    return json.dumps({"data": data, "layout": layout})

def create_treemap_cnae_setor(df: pd.DataFrame) -> str:
    """Treemap SEM subprograma: raiz 'Setores' -> filhos = CNAE(Setor) com contagem."""
    if df.empty:
        return _empty()

    for c in ["CNAE do Projeto (Setor)", "CNAE Projeto (Setor)", "CNAE da Empresa (Setor)", "CNAE Empresa (Setor)"]:
        if c in df.columns:
            cnae_col = c
            break
    else:
        return _empty()

    base = df[[cnae_col]].dropna()
    if base.empty:
        return _empty()

    grp = (
        base.groupby(cnae_col, dropna=True)
            .size()
            .reset_index(name="qtd")
            .sort_values("qtd", ascending=False)
    )

    labels = ["Setores"] + grp[cnae_col].astype(str).tolist()
    parents = [""] + ["Setores"] * len(grp)

    total = int(grp["qtd"].sum())
    children = [int(v) for v in grp["qtd"].tolist()]
    values = [total] + children

    data = [{
        "type": "treemap",
        "labels": labels,
        "parents": parents,
        "values": values,
        "branchvalues": "total",
        "marker": {"colors": GREEN},
        "hovertemplate": "%{label}: %{value}<extra></extra>"
    }]

    layout = _base_layout_no_title()
    layout["margin"] = {"l": 6, "r": 6, "t": 6, "b": 6}

    return json.dumps({"data": data, "layout": layout})
