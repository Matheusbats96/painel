# app.py
# Flask: páginas HTML + API + modo diagnóstico.
# Mantém telemetria, cache, limites da tabela e entrega os NOVOS gráficos:
# - score_gauge (igual)
# - finalidade_pie_chart (sobe para o topo no lugar do "Distribuição por Porte")
# - treemap_cnae_setor (sem Subprograma; ocupa a LINHA INFERIOR toda)
#
# Observações:
# - Mantém suporte ao filtro "Porte" na SIDEBAR (não há mais o gráfico de Porte).
# - Adiciona filtro de "Gerência" no dashboard e na tabela.
# - NÃO removeu nada crítico que você já tinha.

from __future__ import annotations

from flask import Flask, jsonify, request, render_template
import pandas as pd
import logging
from functools import lru_cache
from pathlib import Path
from datetime import datetime
from collections import deque
import math
import json
import os

import plot_utils  # suas funções de gráficos

# --------------------------------
# App + logging
# --------------------------------
BASE_DIR = Path(__file__).resolve().parent
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static")
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("app")

DATA_PATH = BASE_DIR / "data" / "gerop_processed.parquet"

# Limite padrão de linhas renderizadas na Tabela (evita travar o DOM)
TABLE_MAX_ROWS = int(os.environ.get("TABLE_MAX_ROWS", "3000"))

# --------------------------------
# Debug/telemetria de UI (em memória)
# --------------------------------
DEBUG_UI = True  # deixe True até fecharmos; depois pode desativar
_TELEMETRY_BUFFER = deque(maxlen=300)

def _telemetry_push(kind: str, payload: dict):
    try:
        rec = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "kind": kind,
            "remote_addr": request.headers.get("X-Forwarded-For") or request.remote_addr,
            "path": request.path,
            "user_agent": request.headers.get("User-Agent"),
            "payload": payload,
        }
        _TELEMETRY_BUFFER.append(rec)
        log.info("[UI-TELEMETRY] %s", json.dumps(rec, ensure_ascii=False))
    except Exception as e:
        log.exception("Falha ao registrar telemetria: %s", e)

@app.post("/__telemetry")
def ui_telemetry():
    if not DEBUG_UI:
        return jsonify({"ok": False, "msg": "DEBUG_UI desativado"}), 403
    data = request.get_json(silent=True) or {}
    kind = str(data.get("type") or "event")
    _telemetry_push(kind, data)
    return jsonify({"ok": True, "stored": len(_TELEMETRY_BUFFER)})

@app.get("/__telemetry_dump")
def ui_telemetry_dump():
    if not DEBUG_UI:
        return jsonify({"ok": False, "msg": "DEBUG_UI desativado"}), 403
    return jsonify({"ok": True, "count": len(_TELEMETRY_BUFFER), "events": list(_TELEMETRY_BUFFER)})

@app.get("/__debug_on")
def debug_on():
    global DEBUG_UI
    DEBUG_UI = True
    _telemetry_push("server", {"note": "DEBUG_UI ativado"})
    return jsonify({"ok": True, "DEBUG_UI": DEBUG_UI})

@app.get("/__debug_off")
def debug_off():
    global DEBUG_UI
    DEBUG_UI = False
    _telemetry_push("server", {"note": "DEBUG_UI desativado"})
    return jsonify({"ok": True, "DEBUG_UI": DEBUG_UI})

# --------------------------------
# Utils de moeda/data
# --------------------------------
def _is_nan_or_inf(v: float) -> bool:
    return isinstance(v, float) and (math.isnan(v) or math.isinf(v))

def format_brl(value) -> str:
    try:
        v = float(value)
    except Exception:
        v = 0.0
    if _is_nan_or_inf(v):
        v = 0.0
    inteiro = int(abs(v))
    centavos = int(round((abs(v) - inteiro) * 100))
    s = f"{inteiro:,}".replace(",", ".")
    return ("-R$ " if v < 0 else "R$ ") + f"{s},{centavos:02d}"

def parse_brl(v) -> float:
    if v is None:
        return float("nan")
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).replace("R$", "").replace("r$", "").strip()
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return float("nan")

def to_datetime_safe(x):
    return pd.to_datetime(x, errors="coerce")

# --------------------------------
# Carregamento de dados (cache)
# --------------------------------
@lru_cache(maxsize=1)
def _load_data_cached() -> pd.DataFrame:
    if not DATA_PATH.exists():
        log.error("Parquet não encontrado em %s. Rode o data_processor.py.", DATA_PATH)
        return pd.DataFrame()
    try:
        df = pd.read_parquet(DATA_PATH)
        if "Dt. Contrato" in df.columns:
            df["Dt. Contrato"] = pd.to_datetime(df["Dt. Contrato"], errors="coerce")
        return df
    except Exception as e:
        log.exception("Erro ao ler Parquet: %s", e)
        return pd.DataFrame()

def get_data() -> pd.DataFrame:
    return _load_data_cached().copy()

# --------------------------------
# Helpers
# --------------------------------
POSSIBLE_CNAE_SETOR_COLS = [
    "CNAE do Projeto (Setor)",
    "CNAE Projeto (Setor)",
    "CNAE da Empresa (Setor)",
    "CNAE Empresa (Setor)",
]

def get_cnae_setor_col(df: pd.DataFrame) -> str | None:
    for c in POSSIBLE_CNAE_SETOR_COLS:
        if c in df.columns:
            return c
    return None

def _find_col(df: pd.DataFrame, candidates):
    """Retorna o nome da coluna real que casa com algum dos candidates (case-insensitive, substring)."""
    low = [c.lower() for c in candidates]
    # match exato (case-insensitive)
    for col in df.columns:
        if col.lower() in low:
            return col
    # match por substring: cobre "Gerência", "Gerencia", "Gerenc..."
    for col in df.columns:
        if any(token in col.lower() for token in ["gerência", "gerencia", "gerenc"]):
            return col
    return None

# --------------------------------
# KPIs / filtros / graphs
# --------------------------------
def _compute_kpis(df: pd.DataFrame) -> dict:
    if df.empty:
        return dict(
            total_operacoes="0",
            total_liberado=format_brl(0),
            ticket_medio=format_brl(0),
            perc_atraso_carencia="0,0%",
            media_score="0,00",
        )

    total_ops = len(df)
    total_liberado = df["Valor Liberado"].sum() if "Valor Liberado" in df.columns else 0.0
    ticket_medio = (total_liberado / total_ops) if total_ops > 0 else 0.0

    atraso_col = "Atrasou na Carência?" if "Atrasou na Carência?" in df.columns else None
    if atraso_col and pd.api.types.is_numeric_dtype(df[atraso_col]):
        perc_atraso = 100.0 * df[atraso_col].mean(skipna=True)
    else:
        perc_atraso = 0.0

    score_col = "Score_Previsto" if "Score_Previsto" in df.columns else None
    media_score = float(df[score_col].mean(skipna=True)) if (score_col and pd.api.types.is_numeric_dtype(df[score_col])) else float("nan")

    return dict(
        total_operacoes=f"{total_ops:,}".replace(",", "."),
        total_liberado=format_brl(total_liberado),
        ticket_medio=format_brl(ticket_medio),
        perc_atraso_carencia=f"{perc_atraso:,.1f}%".replace(",", "X").replace(".", ",").replace("X", "."),
        media_score=(f"{media_score:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") if not math.isnan(media_score) else "0,00"),
    )

def _initial_filters_for_front(df: pd.DataFrame) -> dict:
    if df.empty:
        return dict(
            min_date=None, max_date=None,
            min_valor=0, max_valor=0,
            portes=[], subprogramas=[], finalidades=[], cnae_setores=[], gerencias=[]
        )

    # Datas
    if "Dt. Contrato" in df.columns:
        dt = pd.to_datetime(df["Dt. Contrato"], errors="coerce")
        dt_min, dt_max = dt.min(), dt.max()
        min_date = dt_min.strftime("%Y-%m-%d") if pd.notna(dt_min) else None
        max_date = dt_max.strftime("%Y-%m-%d") if pd.notna(dt_max) else None
    else:
        min_date = max_date = None

    # Valor
    if "Valor Liberado" in df.columns and pd.api.types.is_numeric_dtype(df["Valor Liberado"]):
        min_valor = float(df["Valor Liberado"].min())
        max_valor = float(df["Valor Liberado"].max())
    else:
        min_valor = 0.0
        max_valor = 0.0

    portes = sorted([x for x in df.get("Porte", pd.Series(dtype=object)).dropna().unique().tolist() if str(x).strip()])
    subprogramas = sorted([x for x in df.get("Subprograma_1", pd.Series(dtype=object)).dropna().unique().tolist() if str(x).strip()])
    finalidades = sorted([x for x in df.get("Finalidade", pd.Series(dtype=object)).dropna().unique().tolist() if str(x).strip()])

    cnae_col = get_cnae_setor_col(df)
    if cnae_col:
        cnae_setores = sorted([x for x in df.get(cnae_col, pd.Series(dtype=object)).dropna().unique().tolist() if str(x).strip()])
    else:
        cnae_setores = []

    ger_col = _find_col(df, ["Gerência", "Gerencia"])
    gerencias = sorted(df[ger_col].dropna().astype(str).unique().tolist()) if ger_col else []

    return dict(
        min_date=min_date, max_date=max_date,
        min_valor=min_valor, max_valor=max_valor,
        portes=portes, subprogramas=subprogramas, finalidades=finalidades,
        cnae_setores=cnae_setores, gerencias=gerencias
    )

def _build_graphs(df: pd.DataFrame) -> dict:
    # 3 gráficos: gauge, pizza finalidades (topo), treemap CNAE Setor (base inteira)
    return {
        "score_gauge": plot_utils.create_score_gauge(df),
        "finalidade_pie_chart": plot_utils.create_finalidade_pie(df),
        "treemap_cnae_setor": plot_utils.create_treemap_cnae_setor(df),
    }

# --------------------------------
# Rotas HTML
# --------------------------------
@app.route("/", methods=["GET"])
def pagina_tabela():
    if DEBUG_UI:
        _telemetry_push("server", {"note": "GET / (tabela) render"})
    df_full = get_data()

    total_rows = len(df_full)
    show_all = request.args.get("all") == "1"
    if show_all or total_rows <= TABLE_MAX_ROWS:
        df = df_full
        limited = False
        display_rows = total_rows
    else:
        df = df_full.head(TABLE_MAX_ROWS).copy()
        limited = True
        display_rows = len(df)

    if df.empty:
        cabecalho, dados = [], []
        tot, media = "0", "0,00"
    else:
        cabecalho = df.columns.tolist()

        def _fmt_cell(col, v):
            if col == "Valor Liberado":
                return format_brl(v)
            if col == "Dt. Contrato":
                try:
                    d = pd.to_datetime(v, errors="coerce")
                    return d.strftime("%d/%m/%Y") if pd.notna(d) else ""
                except Exception:
                    return str(v)
            if v is None or isinstance(v, str):
                return v
            if isinstance(v, (int, float)):
                return f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            return str(v)

        dados = [{col: _fmt_cell(col, row[col]) for col in cabecalho} for _, row in df.iterrows()]
        kpis = _compute_kpis(df_full)  # KPIs calculados no conjunto completo
        tot, media = kpis["total_operacoes"], kpis["media_score"]

    # >>> lista de GERÊNCIAS para o select da tabela
    ger_col = _find_col(df_full, ["Gerência", "Gerencia"])
    gerencias = sorted(df_full[ger_col].dropna().astype(str).unique().tolist()) if ger_col else []

    return render_template(
        "tabela.html",
        active_page="tabela",
        cabecalho=cabecalho,
        dados=dados,
        total_operacoes=tot,
        media_score=media,
        limited=limited,
        display_rows=display_rows,
        total_rows=total_rows,
        table_max_rows=TABLE_MAX_ROWS,
        gerencias=gerencias,  # <--- NOVO
    )

@app.route("/dashboard", methods=["GET"])
def pagina_dashboard():
    if DEBUG_UI:
        _telemetry_push("server", {"note": "GET /dashboard render"})
    return render_template("dashboard.html", active_page="dashboard")


# --------------------------------
# API
# --------------------------------
@app.route("/api/dashboard_data", methods=["POST"])
def api_dashboard_data():
    payload = request.get_json(silent=True) or {}
    is_initial = (len(payload) == 0) or bool(payload.get("initial_load", False))
    if DEBUG_UI:
        _telemetry_push("api_call", {"endpoint": "/api/dashboard_data", "payload": payload})

    df = get_data()

    # filtros por período
    start_date = payload.get("start_date")
    end_date = payload.get("end_date")
    if start_date and end_date and "Dt. Contrato" in df.columns:
        s = to_datetime_safe(start_date); e = to_datetime_safe(end_date)
        if pd.notna(s) and pd.notna(e):
            df = df[df["Dt. Contrato"].between(s, e)]

    # filtro por valor liberado
    valor_range = payload.get("valor_range")
    if valor_range and len(valor_range) == 2 and "Valor Liberado" in df.columns:
        try:
            vmin = parse_brl(valor_range[0]); vmax = parse_brl(valor_range[1])
            if not math.isnan(vmin) and not math.isnan(vmax):
                df = df[df["Valor Liberado"].between(vmin, vmax)]
        except Exception as ex:
            if DEBUG_UI:
                _telemetry_push("warn", {"msg": "falha ao aplicar valor_range", "ex": str(ex)})

    def _as_list(x):
        if x is None: return None
        return x if isinstance(x, list) else [x]

    # filtros de sidebar
    portes        = _as_list(payload.get("portes"))
    subs          = _as_list(payload.get("subprogramas"))
    finalidades   = _as_list(payload.get("finalidades"))
    cnae_setores  = _as_list(payload.get("cnae_setores"))
    gerencias     = _as_list(payload.get("gerencias"))

    if portes and "Porte" in df.columns:
        df = df[df["Porte"].isin([p for p in portes if p is not None])]
    if subs and "Subprograma_1" in df.columns:
        df = df[df["Subprograma_1"].isin([s for s in subs if s is not None])]
    if finalidades and "Finalidade" in df.columns:
        df = df[df["Finalidade"].isin([f for f in finalidades if f is not None])]

    cnae_col = get_cnae_setor_col(df)
    if cnae_setores and cnae_col:
        df = df[df[cnae_col].isin([c for c in cnae_setores if c is not None])]

    ger_col = _find_col(df, ["Gerência", "Gerencia"])
    if ger_col and gerencias:
        df = df[df[ger_col].astype(str).isin([g for g in gerencias if g is not None])]

    # filtros por clique (gráficos ativos)
    clicked_finalidade = payload.get("clicked_finalidade")
    if clicked_finalidade and "Finalidade" in df.columns:
        df = df[df["Finalidade"] == clicked_finalidade]

    clicked_cnae = payload.get("clicked_cnae_setor")
    cnae_col = get_cnae_setor_col(df)
    if clicked_cnae and cnae_col:
        df = df[df[cnae_col] == clicked_cnae]

    # compat: caso volte a usar gráfico de Porte no futuro
    clicked_porte = payload.get("clicked_porte")
    if clicked_porte and "Porte" in df.columns:
        df = df[df["Porte"] == clicked_porte]

    kpis = _compute_kpis(df)
    graphs = _build_graphs(df)
    filters = _initial_filters_for_front(get_data()) if is_initial else {}
    resp = dict(filters=filters, kpis=kpis, graphs=graphs)

    if DEBUG_UI:
        _telemetry_push("api_resp_summary", {
            "kpis_keys": list(kpis.keys()),
            "graphs_keys": list(graphs.keys()),
            "filters_present": bool(filters)
        })

    return jsonify(resp)

@app.post("/api/refresh_cache")
def refresh_cache():
    _load_data_cached.cache_clear()
    _ = get_data()
    if DEBUG_UI:
        _telemetry_push("server", {"note": "cache invalidado"})
    return jsonify({"status": "ok", "message": "Cache limpo e recarregado."})

# --------------------------------
# Health
# --------------------------------
@app.get("/health")
def health():
    return jsonify({"status": "ok", "ts": datetime.utcnow().isoformat()})

if __name__ == "__main__":
    app.run(debug=True)
