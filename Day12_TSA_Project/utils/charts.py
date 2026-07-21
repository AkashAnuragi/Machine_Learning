import plotly.graph_objects as go
import plotly.express as px

def line_chart(df , column):
    fig = px.line(
        df , 
        x = 'Date',
        y = column , 
        title= f"{column} Price Trend"
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=column,
        template="plotly_white",
        hovermode="x unified",
        
    )
    return fig