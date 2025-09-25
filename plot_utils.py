import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Cor principal do tema para consistência visual
THEME_COLOR = '#004C3F'

def create_score_gauge(df):
    """
    BI Aprimoramento:
    1. Reintroduzido no dashboard com design limpo e focado.
    2. O título e o número são destacados para leitura rápida.
    3. Usa a paleta de cores do tema para se integrar ao design.
    """
    if df.empty or df['Score_Previsto'].isnull().all(): return None
    
    avg_score = df['Score_Previsto'].mean()
    max_score = 10 
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_score,
        number={'font': {'size': 48}, 'valueformat': '.2f'},
        title={'text': "Média do Score Previsto", 'font': {'size': 20}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, max_score], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': THEME_COLOR, 'thickness': 0.4},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#EAEAEA",
            'steps': [
                {'range': [0, max_score * 0.4], 'color': '#EAEAEA'},
                {'range': [max_score * 0.4, max_score * 0.7], 'color': '#D0D0D0'}
            ],
        }))
    fig.update_layout(
        height=350, 
        margin=dict(l=30, r=30, t=60, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#333")
    )
    return fig.to_json()

def create_subprograma_chart(df):
    if df.empty or 'Subprograma_1' not in df.columns: return None
    
    top_10 = df['Subprograma_1'].value_counts().nlargest(10).reset_index()
    top_10.columns = ['Subprograma', 'Nº de Operações']
    top_10 = top_10.sort_values(by='Nº de Operações', ascending=True)

    fig = px.bar(top_10, 
                 x='Nº de Operações', 
                 y='Subprograma', 
                 orientation='h',
                 title='<b>Top 10 Subprogramas por Nº de Operações</b>',
                 text='Nº de Operações')
                 
    fig.update_traces(marker_color=THEME_COLOR, textposition='outside', textfont_size=12)
    fig.update_layout(
        title_x=0.5,
        xaxis_title='Quantidade de Operações',
        yaxis_title=None,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#333"),
        yaxis=dict(tickfont=dict(size=12)),
        margin=dict(l=10, r=10, t=50, b=20)
    )
    return fig.to_json()

def create_porte_pie_chart(df):
    """
    BI Aprimoramento (LÓGICA CORRIGIDA):
    1. AGREGAÇÃO POR VALOR: Agrupa os dados por 'Porte' e soma o 'Valor Liberado'.
    2. TOOLTIP AVANÇADO: Mostra Valor Total, % do Total, Nº de Operações e Ticket Médio.
    3. VISUAL CORRIGIDO: Rótulos dentro do gráfico para evitar cortes e legenda otimizada.
    """
    if df.empty or 'Porte' not in df.columns: return None
    
    # Agrupa por Porte, somando o Valor Liberado e contando as operações
    agg_df = df.groupby('Porte').agg(
        Valor_Total_Liberado=('Valor Liberado', 'sum'),
        Num_Operacoes=('Porte', 'count')
    ).reset_index()

    # Calcula o Ticket Médio
    agg_df['Ticket_Medio'] = agg_df['Valor_Total_Liberado'] / agg_df['Num_Operacoes']

    # Formata valores para o tooltip
    agg_df['Valor_Total_Formatado'] = [f'R$ {v:,.2f}'.replace(',', 'X').replace('.', ',').replace('X', '.') for v in agg_df['Valor_Total_Liberado']]
    agg_df['Ticket_Medio_Formatado'] = [f'R$ {v:,.2f}'.replace(',', 'X').replace('.', ',').replace('X', '.') for v in agg_df['Ticket_Medio']]

    fig = px.pie(agg_df, 
                 values='Valor_Total_Liberado', 
                 names='Porte', 
                 title='<b>% do Valor Total Liberado por Porte</b>', 
                 hole=0.4,
                 color_discrete_sequence=px.colors.sequential.Teal)
                 
    # Constrói o hovertemplate customizado
    hovertemplate = (
        "<b>%{label}</b><br><br>" +
        "Valor Liberado: %{customdata[0]}<br>" +
        "Percentual do Total: %{percent}<br>" +
        "Nº de Operações: %{customdata[1]}<br>" +
        "Ticket Médio: %{customdata[2]}" +
        "<extra></extra>" # Remove o trace 'fantasma'
    )

    fig.update_traces(
        customdata=agg_df[['Valor_Total_Formatado', 'Num_Operacoes', 'Ticket_Medio_Formatado']],
        hovertemplate=hovertemplate,
        textposition='inside', # Põe o texto dentro para evitar cortes
        textinfo='percent',
        textfont_size=14,
        insidetextorientation='radial'
    )
    fig.update_layout(
        title_x=0.5,
        legend_title_text='Porte da Empresa',
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#333"),
        margin=dict(l=10, r=10, t=50, b=20)
    )
    return fig.to_json()

def create_treemap_finalidade(df):
    if df.empty or 'Finalidade' not in df.columns or 'Valor Liberado' not in df.columns: return None
    
    df_tree = df.groupby('Finalidade')['Valor Liberado'].sum().reset_index()
    formatted_values = [f'R$ {v:,.2f}'.replace(',', 'X').replace('.', ',').replace('X', '.') for v in df_tree['Valor Liberado']]
    
    fig = px.treemap(df_tree, 
                     path=[px.Constant("Todas as Finalidades"), 'Finalidade'], 
                     values='Valor Liberado',
                     title='<b>Valor Total Liberado por Finalidade</b>',
                     color='Valor Liberado',
                     color_continuous_scale='Teal')
    
    fig.update_traces(
        customdata=formatted_values,
        texttemplate="<b>%{label}</b><br>%{customdata}",
        hovertemplate="<b>%{label}</b><br>Valor Liberado: %{customdata}<extra></extra>",
        textfont_size=16
    )
    fig.update_layout(
        title_x=0.5,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#333"),
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return fig.to_json()