from __future__ import annotations

import os
from pathlib import Path
import logging
import re
from typing import Optional, Tuple

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


BASE_DIR = Path(__file__).resolve().parent
DATA_SOURCE_PATH = BASE_DIR / "data" / "gerop.xlsx"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "gerop_processed.parquet"
MODEL_OUTPUT_PATH = BASE_DIR / "models" / "score_model.joblib"


REPLACE_CNAE = os.environ.get("REPLACE_CNAE", "0") == "1"

os.makedirs(PROCESSED_DATA_PATH.parent, exist_ok=True)
os.makedirs(MODEL_OUTPUT_PATH.parent, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("data_processor")



def parse_brl_to_float(v) -> float:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return np.nan
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).replace("R$", "").replace("r$", "").strip()
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan


def ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].map(parse_brl_to_float), errors="coerce")
    return df


def map_binary_sim_nao(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].map({"Sim": 1, "Não": 0, "Nao": 0, "NAO": 0}).fillna(0).astype(int)
    return df


def choose_encoder():
    """Cria OneHotEncoder compatível com múltiplas versões do sklearn."""
    try:
        # sklearn >= 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # versões mais antigas
        return OneHotEncoder(handle_unknown="ignore", sparse=False)



CNAE_SECTIONS = [
    ("A", (1, 3),  "AGRICULTURA, PECUÁRIA, PRODUÇÃO FLORESTAL, PESCA E AQÜICULTURA"),
    ("B", (5, 9),  "INDÚSTRIAS EXTRATIVAS"),
    ("C", (10, 33),"INDÚSTRIAS DE TRANSFORMAÇÃO"),
    ("D", (35, 35),"ELETRICIDADE E GÁS"),
    ("E", (36, 39),"ÁGUA, ESGOTO, ATIVIDADES DE GESTÃO DE RESÍDUOS E DESCONTAMINAÇÃO"),
    ("F", (41, 43),"CONSTRUÇÃO"),
    ("G", (45, 47),"COMÉRCIO; REPARAÇÃO DE VEÍCULOS AUTOMOTORES E MOTOCICLETAS"),
    ("H", (49, 53),"TRANSPORTE, ARMAZENAGEM E CORREIO"),
    ("I", (55, 56),"ALOJAMENTO E ALIMENTAÇÃO"),
    ("J", (58, 63),"INFORMAÇÃO E COMUNICAÇÃO"),
    ("K", (64, 66),"ATIVIDADES FINANCEIRAS, DE SEGUROS E SERVIÇOS RELACIONADOS"),
    ("L", (68, 68),"ATIVIDADES IMOBILIÁRIAS"),
    ("M", (69, 75),"ATIVIDADES PROFISSIONAIS, CIENTÍFICAS E TÉCNICAS"),
    ("N", (77, 82),"ATIVIDADES ADMINISTRATIVAS E SERVIÇOS COMPLEMENTARES"),
    ("O", (84, 84),"ADMINISTRAÇÃO PÚBLICA, DEFESA E SEGURIDADE SOCIAL"),
    ("P", (85, 85),"EDUCAÇÃO"),
    ("Q", (86, 88),"SAÚDE HUMANA E SERVIÇOS SOCIAIS"),
    ("R", (90, 93),"ARTES, CULTURA, ESPORTE E RECREAÇÃO"),
    ("S", (94, 96),"OUTRAS ATIVIDADES DE SERVIÇOS"),
    ("T", (97, 97),"SERVIÇOS DOMÉSTICOS"),
    ("U", (99, 99),"ORGANISMOS INTERNACIONAIS E OUTRAS INSTITUIÇÕES EXTRATERRITORIAIS"),
]
LETTER_TO_NAME = {sec: nome for sec, _, nome in CNAE_SECTIONS}
_division_re = re.compile(r"(\d{2})")

def _extract_leading_letter(value: str) -> Optional[str]:
    if not isinstance(value, str):
        value = str(value) if value is not None else ""
    value = value.strip()
    if not value:
        return None
    first = value[0].upper()
    return first if first in LETTER_TO_NAME else None

def _extract_numeric_division(value: str) -> Optional[int]:
    if not isinstance(value, str):
        value = str(value) if value is not None else ""
    m = _division_re.search(value)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def _division_to_section(div: int) -> Optional[Tuple[str, str]]:
    for sec, (start, end), nome in CNAE_SECTIONS:
        if start <= div <= end:
            return sec, nome
    return None

def cnae_to_setor(value) -> Optional[str]:
    """1) Se começar com letra A..U → mapeia; 2) senão, usa divisão 2 dígitos."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    letter = _extract_leading_letter(value)
    if letter:
        return LETTER_TO_NAME.get(letter)
    div = _extract_numeric_division(value)
    if div is not None:
        sec = _division_to_section(div)
        if sec:
            _, nome = sec
            return nome
    return None

def normalize_cnae_columns(df: pd.DataFrame, replace: bool = False) -> pd.DataFrame:
    """Cria '<col> (Setor)' para cada coluna com 'CNAE' no nome. Se replace=True, substitui original."""
    if df.empty:
        return df
    cnae_cols = [c for c in df.columns if "CNAE" in str(c).upper()]
    if not cnae_cols:
        return df
    for col in cnae_cols:
        setor_col = f"{col} (Setor)"
        setor_values = df[col].apply(cnae_to_setor)
        if replace:
            backup_col = f"{col} (Original)"
            if backup_col not in df.columns:
                df[backup_col] = df[col]
            df[col] = setor_values
        else:
            df[setor_col] = setor_values
    return df

PLACEHOLDERS = {"", "-", "--", "—", "N/A", "NA", "na", "n/a", "None", "none"}

def _clean_placeholders_to_na(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip()
    s = s.replace(list(PLACEHOLDERS), pd.NA)
    return s

def _to_float_brl(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.upper().str.replace("R$", "", regex=False).str.strip()
    s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

def _to_float_ptbr(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

def _to_float_percent(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.replace("%", "", regex=False).str.strip()
    s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

def sanitize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Converte colunas mistas para tipos aceitos pelo pyarrow sem quebrar suas lógicas acima."""
    if df.empty:
        return df
    for col in df.columns:
        ser = df[col]
        if pd.api.types.is_numeric_dtype(ser) or pd.api.types.is_bool_dtype(ser) or pd.api.types.is_datetime64_any_dtype(ser):
            continue
        if not (pd.api.types.is_object_dtype(ser) or pd.api.types.is_string_dtype(ser)):
            continue

        s = _clean_placeholders_to_na(ser)
        s_non = s.dropna()
        if s_non.empty:
            df[col] = s
            continue

        frac_pct = s_non.str.contains("%", regex=False).mean()
        frac_brl = s_non.str.contains("R$", regex=False).mean()

        if frac_pct >= 0.5:
            df[col] = _to_float_percent(s)
            continue
        if frac_brl >= 0.5:
            df[col] = _to_float_brl(s)
            continue

        as_num = _to_float_ptbr(s)
        if as_num.notna().mean() >= 0.6:
            df[col] = as_num
        else:
            df[col] = s

    return df.convert_dtypes()


def main():
    if not DATA_SOURCE_PATH.exists():
        log.error("Fonte %s não encontrada.", DATA_SOURCE_PATH)
        raise SystemExit(1)

    log.info("Lendo Excel...")
    try:
        df = pd.read_excel(DATA_SOURCE_PATH, engine="openpyxl")
    except Exception:
    
        df = pd.read_excel(DATA_SOURCE_PATH)

    if "Dt. Contrato" in df.columns:
        df["Dt. Contrato"] = pd.to_datetime(df["Dt. Contrato"], errors="coerce")


    candidates_brl = [c for c in df.columns if "Valor" in c or "Montante" in c]
    for specific in ["Valor Liberado"]:
        if specific in df.columns and specific not in candidates_brl:
            candidates_brl.append(specific)
    df = ensure_numeric(df, candidates_brl)


    df = map_binary_sim_nao(
        df,
        ["Atrasou na Carência?", "Possui FAE?", "Possui Honra no Fundo?", "Liquidado?"],
    )


    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in obj_cols:
        df[col] = df[col].fillna("N/A").astype(str)


    df = normalize_cnae_columns(df, replace=REPLACE_CNAE)


    if "Score" in df.columns:
        log.info("Treinando modelo para Score_Previsto...")
        y = pd.to_numeric(df["Score"], errors="coerce")


        drop_cols = {"Score", "Score_Previsto"}
        drop_cols |= {c for c in df.columns if c.lower() in {"cpf", "cnpj"}}
        drop_cols |= {c for c in df.columns if c.lower().startswith("id")}
        X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")


        if "Dt. Contrato" in X.columns:
            X = X.drop(columns=["Dt. Contrato"])

        cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        pre = ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="median"), num_cols),
                ("cat", Pipeline(steps=[
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("ohe", choose_encoder()),
                ]), cat_cols),
            ],
            remainder="drop",
        )

        if y.nunique(dropna=True) > 20 or y.dtype.kind in {"f"}:
            model = Pipeline(steps=[
                ("pre", pre),
                ("est", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
            ])
            task = "reg"
        else:
            model = Pipeline(steps=[
                ("pre", pre),
                ("est", RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)),
            ])
            task = "clf"

        mask = y.notna()
        if mask.sum() > 0 and mask.sum() >= 10:
            Xtrain, Xtest, ytrain, ytest = train_test_split(X[mask], y[mask], test_size=0.2, random_state=42)
            model.fit(Xtrain, ytrain)
            preds = model.predict(Xtest)

            if task == "reg":
                mae = mean_absolute_error(ytest, preds)
                r2 = r2_score(ytest, preds)
                log.info("Validação (Regressor) - MAE: %.4f | R2: %.4f", mae, r2)
            else:
                acc = (preds == ytest).mean()
                log.info("Validação (Classifier) - ACC: %.4f", acc)


            try:
                df["Score_Previsto"] = model.predict(X)
            except Exception:
                log.exception("Falha ao prever Score_Previsto no dataset completo.")
                df["Score_Previsto"] = np.nan


            joblib.dump(model, MODEL_OUTPUT_PATH)
            log.info("Modelo salvo em %s", MODEL_OUTPUT_PATH)
        else:
            log.warning("Dados insuficientes para treinar (Score não disponível o bastante).")
            if "Score_Previsto" not in df.columns:
                df["Score_Previsto"] = np.nan
    else:
        if "Score_Previsto" not in df.columns:
            df["Score_Previsto"] = np.nan
        log.info("Coluna 'Score' não encontrada. Pulando treino de modelo.")


    df_out = sanitize_for_parquet(df)


    df_out.to_parquet(PROCESSED_DATA_PATH, index=False)
    log.info("Parquet salvo em %s", PROCESSED_DATA_PATH)


if __name__ == "__main__":
    main()
