from flask import Flask, render_template, jsonify, request
import pandas as pd
import os
import plot_utils

app = Flask(__name__)
DATA_PATH = 'data/gerop_processed.parquet'

def get_data():
    """
    Função central para carregar e retornar o DataFrame processado.
    """
    if not os.path.exists(DATA_PATH):
        print(f"ERRO: Arquivo de dados '{DATA_PATH}' não encontrado. Execute data_processor.py")
        return pd.DataFrame()
    try:
        df = pd.read_parquet(DATA_PATH)
        if 'Dt. Contrato' in df.columns:
            df['Dt. Contrato'] = pd.to_datetime(df['Dt. Contrato'], errors='coerce')
        return df
    except Exception as e:
        print(f"ERRO ao carregar o arquivo Parquet: {e}")
        return pd.DataFrame()

def get_dashboard_data(dataframe, initial_load=False):
    """
    Agrupa todos os cálculos de dados para o dashboard.
    Recebe o dataframe (já filtrado ou não) e um flag para a carga inicial.
    """
    if dataframe.empty:
        return {
            'filters': {}, 
            'kpis': {'total_operacoes': '0', 'total_liberado': 'R$ 0,00', 'ticket_medio': 'R$ 0,00', 'perc_atraso_carencia': '0,0%', 'media_score': '0,00'}, 
            'graphs': {'score_gauge': None, 'subprograma_chart': None, 'porte_pie_chart': None, 'treemap_finalidade': None}
        }
    
    # --- Cálculos dos KPIs ---
    kpis = {
        'total_operacoes': f"{len(dataframe):,}".replace(',', '.'),
        'total_liberado': f"R$ {dataframe['Valor Liberado'].sum():,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'),
        'ticket_medio': f"R$ {dataframe['Valor Liberado'].mean():,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'),
        'perc_atraso_carencia': f"{(dataframe['Atrasou na Carência?'].sum() / len(dataframe) * 100 if len(dataframe) > 0 else 0):.1f}%".replace('.', ','),
        'media_score': f"{dataframe['Score_Previsto'].mean():.2f}".replace('.', ',')
    }

    # --- Geração dos Gráficos ---
    graphs = {
        'score_gauge': plot_utils.create_score_gauge(dataframe),
        'subprograma_chart': plot_utils.create_subprograma_chart(dataframe),
        'porte_pie_chart': plot_utils.create_porte_pie_chart(dataframe),
        'treemap_finalidade': plot_utils.create_treemap_finalidade(dataframe)
    }
    
    response_data = {'kpis': kpis, 'graphs': graphs}

    # --- População dos Filtros (apenas na carga inicial) ---
    if initial_load:
        full_df = get_data()
        date_col = 'Dt. Contrato'
        filters = {
            'portes': sorted(full_df['Porte'].astype(str).unique().tolist()),
            'subprogramas': sorted(full_df['Subprograma_1'].astype(str).unique().tolist()),
            'min_date': full_df[date_col].min().strftime('%Y-%m-%d') if not full_df[date_col].isnull().all() else None,
            'max_date': full_df[date_col].max().strftime('%Y-%m-%d') if not full_df[date_col].isnull().all() else None,
            'min_valor': float(full_df['Valor Liberado'].min()),
            'max_valor': float(full_df['Valor Liberado'].max()),
        }
        response_data['filters'] = filters

    return response_data

# --- ROTAS DA APLICAÇÃO ---

@app.route('/')
def tabela():
    """ Rota para a página da Tabela de Dados. """
    df = get_data()
    if df.empty: return "Dados não encontrados. Execute data_processor.py.", 500
    return render_template('tabela.html', 
                           active_page='tabela',
                           cabecalho=df.columns.tolist(),
                           dados=df.to_dict(orient='records'),
                           total_operacoes=f"{len(df):,}".replace(',', '.'))

@app.route('/dashboard')
def dashboard():
    """ Rota para a página do Dashboard. """
    return render_template('dashboard.html', active_page='dashboard')

@app.route('/api/dashboard_data', methods=['POST'])
def api_dashboard_data():
    """ Endpoint da API para fornecer os dados ao front-end do dashboard. """
    df = get_data()
    if df.empty: return jsonify({"error": "Dados não disponíveis"}), 500

    filters = request.json
    is_initial_load = not filters 

    if not is_initial_load:
        # --- FILTROS DA SIDEBAR ---
        if filters.get('start_date') and filters.get('end_date'):
            df = df[df['Dt. Contrato'].between(filters['start_date'], filters['end_date'])]
        if filters.get('portes'):
            df = df[df['Porte'].isin(filters['portes'])]
        if filters.get('subprogramas'):
            df = df[df['Subprograma_1'].isin(filters['subprogramas'])]
        if filters.get('valor_range'):
            try:
                min_val = float(str(filters['valor_range'][0]).replace('R$ ', '').replace('.', ''))
                max_val = float(str(filters['valor_range'][1]).replace('R$ ', '').replace('.', ''))
                df = df[df['Valor Liberado'].between(min_val, max_val)]
            except (ValueError, TypeError, IndexError): pass
        
        # --- FILTROS DE CLIQUE (CROSS-FILTERING) ---
        if filters.get('clicked_porte'):
            df = df[df['Porte'] == filters['clicked_porte']]
        if filters.get('clicked_subprograma'):
            df = df[df['Subprograma_1'] == filters['clicked_subprograma']]
        if filters.get('clicked_finalidade'):
            df = df[df['Finalidade'] == filters['clicked_finalidade']]
            
    return jsonify(get_dashboard_data(df, initial_load=is_initial_load))

if __name__ == '__main__':
    app.run(debug=True)