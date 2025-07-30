#!/usr/bin/env python3
"""
Create professional dual visualization for Filo-Transformer results using Plotly.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Data from the results
metrics = ['Accuracy', 'AUC', 'Recall', 'F1-Score']
baseline_scores = [0.8671, 0.8882, 0.7605, 0.7790]
filo_scores = [0.8702, 0.9071, 0.7661, 0.7847]
improvements = [(filo - base) * 100 for filo, base in zip(filo_scores, baseline_scores)]

# Fusion weights
fusion_weights = {'Phylogenetic Features': 65, 'Semantic Features': 35}

# Create subplots
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('<b>(a) Performance Comparison: Baseline vs Filo-Transformer</b>', 
                    '<b>(b) Learned Fusion Weights Distribution</b>'),
    column_widths=[0.55, 0.45],
    specs=[[{"type": "bar"}, {"type": "pie"}]],
    horizontal_spacing=0.15
)

# Color scheme
baseline_color = 'rgb(52, 152, 219)'  # Blue
filo_color = 'rgb(231, 76, 60)'       # Red
phylo_color = 'rgb(46, 204, 113)'     # Green
semantic_color = 'rgb(243, 156, 18)'  # Orange
improvement_color = 'rgb(39, 174, 96)' # Dark green

# ========== Left subplot: Performance Comparison ==========
# Add baseline bars
fig.add_trace(
    go.Bar(
        name='Baseline (GPT + FT)',
        x=metrics,
        y=baseline_scores,
        marker_color=baseline_color,
        marker_line_color='black',
        marker_line_width=1.5,
        opacity=0.85,
        text=[f'{score:.4f}' for score in baseline_scores],
        textposition='outside',
        textfont=dict(size=10, family='Arial Black'),
        showlegend=True
    ),
    row=1, col=1
)

# Add Filo-Transformer bars
fig.add_trace(
    go.Bar(
        name='Filo-Transformer',
        x=metrics,
        y=filo_scores,
        marker_color=filo_color,
        marker_line_color='black',
        marker_line_width=1.5,
        opacity=0.85,
        text=[f'{score:.4f}' for score in filo_scores],
        textposition='outside',
        textfont=dict(size=10, family='Arial Black'),
        showlegend=True
    ),
    row=1, col=1
)

# Add improvement annotations
for i, (metric, improvement) in enumerate(zip(metrics, improvements)):
    # Add improvement percentage above the bars
    fig.add_annotation(
        x=metric,
        y=max(baseline_scores[i], filo_scores[i]) + 0.025,
        text=f'<b>+{improvement:.2f}%</b>',
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor=improvement_color,
        ax=0,
        ay=-20,
        font=dict(size=11, color=improvement_color, family='Arial Black'),
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor=improvement_color,
        borderwidth=1,
        borderpad=3,
        row=1, col=1
    )

# Customize left subplot
fig.update_xaxes(
    title_text='<b>Metrics</b>',
    title_font=dict(size=12),
    tickfont=dict(size=11),
    showgrid=True,
    gridwidth=0.5,
    gridcolor='rgba(200, 200, 200, 0.3)',
    row=1, col=1
)

fig.update_yaxes(
    title_text='<b>Score</b>',
    title_font=dict(size=12),
    tickfont=dict(size=11),
    range=[0.74, 0.94],
    showgrid=True,
    gridwidth=0.5,
    gridcolor='rgba(200, 200, 200, 0.3)',
    row=1, col=1
)

# ========== Right subplot: Fusion Weights ==========
# Create pie chart
fig.add_trace(
    go.Pie(
        labels=list(fusion_weights.keys()),
        values=list(fusion_weights.values()),
        hole=0.4,
        marker=dict(
            colors=[phylo_color, semantic_color],
            line=dict(color='black', width=2)
        ),
        textinfo='label+percent',
        textposition='outside',
        textfont=dict(size=12, family='Arial Black'),
        pull=[0.1, 0],  # Pull out the phylogenetic slice
        showlegend=False
    ),
    row=1, col=2
)

# Add center text for donut chart
fig.add_annotation(
    text='<b>Learned<br>Fusion<br>Weights</b>',
    x=0.77,  # Adjust based on subplot position
    y=0.5,
    xref='paper',
    yref='paper',
    showarrow=False,
    font=dict(size=13, family='Arial Black'),
    bgcolor='white',
    bordercolor='black',
    borderwidth=2,
    borderpad=10
)

# Add explanatory text below the pie chart
fig.add_annotation(
    text='Model automatically learned to prioritize<br>phylogenetic features (65%) over semantic<br>features (35%), validating cascade structure importance',
    x=0.77,
    y=-0.15,
    xref='paper',
    yref='paper',
    showarrow=False,
    font=dict(size=10, family='Arial', color='gray'),
    bgcolor='rgba(255, 255, 224, 0.8)',
    bordercolor='gray',
    borderwidth=1,
    borderpad=8,
    align='center'
)

# Update overall layout
fig.update_layout(
    title=dict(
        text='<b>Filo-Transformer: Comprehensive Performance Analysis</b>',
        font=dict(size=16, family='Arial Black'),
        x=0.5,
        xanchor='center'
    ),
    showlegend=True,
    legend=dict(
        x=0.02,
        y=0.98,
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor='black',
        borderwidth=1,
        font=dict(size=11)
    ),
    plot_bgcolor='white',
    paper_bgcolor='white',
    height=500,
    width=1000,
    margin=dict(l=50, r=50, t=100, b=100)
)

# Save the visualization
output_path = '/home/acauan/ufam/papers/01_sbseg_filo_trans/visualizations/dual_results_visualization.html'
fig.write_html(output_path)
print(f"Interactive visualization saved to: {output_path}")

# Also save as static images
fig.write_image('/home/acauan/ufam/papers/01_sbseg_filo_trans/visualizations/dual_results_visualization.pdf')
fig.write_image('/home/acauan/ufam/papers/01_sbseg_filo_trans/visualizations/dual_results_visualization.png', scale=2)
print("Static images saved!")

# ===== Create an alternative version with radar chart =====
fig2 = make_subplots(
    rows=1, cols=2,
    subplot_titles=('<b>(a) Multi-Metric Performance Comparison</b>', 
                    '<b>(b) Feature Importance Distribution</b>'),
    column_widths=[0.6, 0.4],
    specs=[[{"type": "scatterpolar"}, {"type": "bar"}]]
)

# Normalize scores for radar chart
categories = metrics + [metrics[0]]  # Complete the circle
baseline_radar = baseline_scores + [baseline_scores[0]]
filo_radar = filo_scores + [filo_scores[0]]

# Add radar traces
fig2.add_trace(
    go.Scatterpolar(
        r=baseline_radar,
        theta=categories,
        fill='toself',
        name='Baseline',
        line_color=baseline_color,
        fillcolor=baseline_color,
        opacity=0.4,
        line=dict(width=2),
        marker=dict(size=8)
    ),
    row=1, col=1
)

fig2.add_trace(
    go.Scatterpolar(
        r=filo_radar,
        theta=categories,
        fill='toself',
        name='Filo-Transformer',
        line_color=filo_color,
        fillcolor=filo_color,
        opacity=0.4,
        line=dict(width=2),
        marker=dict(size=8)
    ),
    row=1, col=1
)

# Update radar chart
fig2.update_polars(
    radialaxis=dict(
        visible=True,
        range=[0.74, 0.92],
        tickfont=dict(size=10)
    ),
    angularaxis=dict(
        tickfont=dict(size=11, family='Arial Black')
    ),
    row=1, col=1
)

# Add fusion weights as horizontal bar chart
fig2.add_trace(
    go.Bar(
        y=list(fusion_weights.keys()),
        x=list(fusion_weights.values()),
        orientation='h',
        marker_color=[phylo_color, semantic_color],
        marker_line_color='black',
        marker_line_width=2,
        text=[f'{v}%' for v in fusion_weights.values()],
        textposition='outside',
        textfont=dict(size=14, family='Arial Black'),
        showlegend=False
    ),
    row=1, col=2
)

# Update bar chart axes
fig2.update_xaxes(
    title_text='<b>Weight (%)</b>',
    range=[0, 80],
    showgrid=True,
    gridcolor='rgba(200, 200, 200, 0.3)',
    row=1, col=2
)

fig2.update_yaxes(
    tickfont=dict(size=11, family='Arial Black'),
    row=1, col=2
)

# Update overall layout for alternative version
fig2.update_layout(
    title=dict(
        text='<b>Filo-Transformer: Performance and Feature Analysis</b>',
        font=dict(size=16, family='Arial Black'),
        x=0.5,
        xanchor='center'
    ),
    showlegend=True,
    legend=dict(
        x=0.4,
        y=0.5,
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor='black',
        borderwidth=1
    ),
    plot_bgcolor='white',
    paper_bgcolor='white',
    height=500,
    width=1000,
    margin=dict(l=50, r=50, t=100, b=50)
)

# Save alternative version
fig2.write_html('/home/acauan/ufam/papers/01_sbseg_filo_trans/visualizations/dual_results_alternative.html')
fig2.write_image('/home/acauan/ufam/papers/01_sbseg_filo_trans/visualizations/dual_results_alternative.pdf')
fig2.write_image('/home/acauan/ufam/papers/01_sbseg_filo_trans/visualizations/dual_results_alternative.png', scale=2)
print("Alternative visualization saved!")