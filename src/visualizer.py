import plotly.graph_objects as go

def create_radar_chart(candidate_name, skill_match, semantic_match, experience_relevance):
    """
    Renders a 3-axis radar chart using the Yield-AI score breakdown.
    Each axis maps to one of the three weighted scoring dimensions.

    Args:
        candidate_name (str): Name shown in the chart title.
        skill_match (float): Direct keyword overlap score (0–100).
        semantic_match (float): LLM contextual alignment score (0–100).
        experience_relevance (float): Career seniority/tool depth score (0–100).
    """
    categories = ['Skill Match', 'Semantic Match', 'Experience Relevance']
    values = [
        float(skill_match or 0),
        float(semantic_match or 0),
        float(experience_relevance or 0)
    ]
    # Close the polygon by repeating the first value
    values_closed = values + [values[0]]
    categories_closed = categories + [categories[0]]

    fig = go.Figure()

    # --- Filled neon polygon ---
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        fillcolor='rgba(0, 255, 204, 0.15)',
        line=dict(color='#00ffcc', width=3),
        marker=dict(color='#00ffcc', size=10, symbol='circle',
                    line=dict(color='#ffffff', width=1)),
        name=candidate_name,
        hovertemplate='<b>%{theta}</b><br>Score: %{r}%<extra></extra>'
    ))

    # --- Reference ring at 50% (benchmark line) ---
    fig.add_trace(go.Scatterpolar(
        r=[50, 50, 50, 50],
        theta=categories_closed,
        mode='lines',
        line=dict(color='rgba(138, 43, 226, 0.5)', width=1, dash='dot'),
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.update_layout(
        polar=dict(
            bgcolor='rgba(10, 12, 30, 0.4)',
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor='rgba(138, 43, 226, 0.3)',
                linecolor='rgba(138, 43, 226, 0.5)',
                tickfont=dict(color='#8a8d9e', size=10),
                ticksuffix='%',
                dtick=25,
            ),
            angularaxis=dict(
                gridcolor='rgba(138, 43, 226, 0.3)',
                linecolor='rgba(138, 43, 226, 0.5)',
                tickfont=dict(color='#00ffcc', size=14, weight='bold')
            )
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        title=dict(
            text=f"📊 Score Breakdown: {candidate_name}",
            font=dict(color='#ffffff', size=18),
            x=0.5
        ),
        margin=dict(t=80, b=40, l=60, r=60),
        height=420
    )

    return fig
