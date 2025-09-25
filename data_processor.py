import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# --- Configuração ---
DATA_SOURCE_PATH = 'data/gerop.xlsx' 
MODEL_OUTPUT_PATH = 'models/score_model.joblib'
PROCESSED_DATA_PATH = 'data/gerop_processed.parquet'

def main():
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)

    try:
        df_original = pd.read_excel(DATA_SOURCE_PATH)
        df_original.columns = df_original.columns.str.strip()
    except Exception as e:
        print(f"ERRO ao ler o arquivo Excel: {e}")
        return

    # --- Definição do Alvo e Features ---
    target = 'Score' # USANDO A COLUNA 'Score' EXISTENTE COMO ALVO
    
    numeric_features = ['Valor Liberado', 'Valor Vencido Atual', 'ROB', 'Prazo de Concessão', 'Adimplência até Fechamento']
    categorical_features = ['Porte', 'Subprograma_1', 'Possui FAE?', 'Finalidade', 'Fonte']
    features = numeric_features + categorical_features

    if target not in df_original.columns:
        print(f"ERRO: A coluna alvo '{target}' não foi encontrada no arquivo Excel. Verifique o nome da coluna.")
        return
    if not all(f in df_original.columns for f in features):
        print(f"ERRO: Colunas de features necessárias para o modelo não encontradas.")
        return

    # Limpa a coluna alvo para garantir que seja numérica
    df_original[target] = pd.to_numeric(df_original[target], errors='coerce')
    df_original.dropna(subset=[target], inplace=True) # Remove linhas onde o score original é inválido
    df_original[target] = df_original[target].astype(int)

    # --- Divisão 50/50 para Treino e Teste ---
    X = df_original[features]
    y = df_original[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)
    print(f"Dados divididos: {len(X_train)} linhas para treino, {len(X_test)} linhas para teste.")

    # --- Pipeline de Pré-processamento e Modelo ---
    numeric_transformer = SimpleImputer(strategy='median')
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    # --- Treinamento e Avaliação ---
    print("Treinando o modelo com 50% dos dados...")
    model_pipeline.fit(X_train, y_train)
    
    print("Avaliando o modelo com os outros 50% dos dados...")
    predictions_test = model_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions_test)
    print(f"Modelo treinado. Acurácia no conjunto de teste: {accuracy:.2%}")
    
    joblib.dump(model_pipeline, MODEL_OUTPUT_PATH)
    print(f"Pipeline salvo em '{MODEL_OUTPUT_PATH}'")

    # --- Geração de Previsões e Preparação do Arquivo Final ---
    print("\nGerando previsões para o conjunto de dados completo...")
    
    # Usa o pipeline treinado para prever em todo o conjunto de dados original
    full_predictions = model_pipeline.predict(df_original[features])
    df_original['Score_Previsto'] = full_predictions
    
    # Limpezas mínimas para o dashboard
    for col in ['Atrasou na Carência?', 'Possui FAE?', 'Possui Honra no Fundo?', 'Liquidado?']:
        if col in df_original.columns:
            df_original[col] = df_original[col].map({'Sim': 1, 'Não': 0}).fillna(0).astype(int)

    if 'Dt. Contrato' in df_original.columns:
        df_original['Dt. Contrato'] = pd.to_datetime(df_original['Dt. Contrato'], errors='coerce')
    
    for col in df_original.select_dtypes(include=['object']).columns:
        df_original[col] = df_original[col].astype(str).fillna('N/A')

    df_original.to_parquet(PROCESSED_DATA_PATH, index=False)
    print(f"Processo concluído. Arquivo final salvo em '{PROCESSED_DATA_PATH}'")

if __name__ == '__main__':
    main()