# Create a clean, hierarchical MLOps architecture diagram using Plotly instead
import plotly.graph_objects as go
import plotly.express as px

# Create a clean architecture diagram using Plotly shapes and annotations
fig = go.Figure()

# Define positions for each layer
layers = {
    'Data Sources': {'y': 5, 'color': '#B3E5EC', 'border': '#1FB8CD'},
    'Data Processing': {'y': 4, 'color': '#A5D6A7', 'border': '#2E8B57'},
    'ML Pipeline': {'y': 3, 'color': '#FFEB8A', 'border': '#D2BA4C'},
    'Orchestration': {'y': 2, 'color': '#F5F5F5', 'border': '#757575'},
    'Serving': {'y': 1, 'color': '#E1BEE7', 'border': '#8E24AA'},
    'Monitoring': {'y': 0, 'color': '#FFCDD2', 'border': '#DB4545'}
}

# Component definitions with positions
components = [
    # Data Sources Layer
    {'name': 'Raw Customer<br>Data<br>(PostgreSQL)', 'x': 1, 'y': 5, 'layer': 'Data Sources'},
    {'name': 'DVC Versioning<br>(Git + S3)', 'x': 3, 'y': 5, 'layer': 'Data Sources'},
    {'name': 'Feature Store<br>(Redis)', 'x': 5, 'y': 5, 'layer': 'Data Sources'},
    
    # Data Processing Layer
    {'name': 'Data Ingestion', 'x': 1, 'y': 4, 'layer': 'Data Processing'},
    {'name': 'Preprocessing<br>(Pandas)', 'x': 3, 'y': 4, 'layer': 'Data Processing'},
    {'name': 'Feature Eng<br>(Sklearn)', 'x': 5, 'y': 4, 'layer': 'Data Processing'},
    
    # ML Pipeline Layer
    {'name': 'Model Training<br>(LGBMClassifier)', 'x': 1, 'y': 3, 'layer': 'ML Pipeline'},
    {'name': 'Model Eval<br>(MLflow)', 'x': 3, 'y': 3, 'layer': 'ML Pipeline'},
    {'name': 'Model Registry<br>(S3/DVC)', 'x': 5, 'y': 3, 'layer': 'ML Pipeline'},
    
    # Orchestration Layer
    {'name': 'Airflow<br>(Scheduler)', 'x': 2, 'y': 2, 'layer': 'Orchestration'},
    {'name': 'Training DAGs<br>(Pipeline)', 'x': 4, 'y': 2, 'layer': 'Orchestration'},
    
    # Serving Layer
    {'name': 'Flask API<br>(REST)', 'x': 1, 'y': 1, 'layer': 'Serving'},
    {'name': 'Model Load<br>(Pickle)', 'x': 3, 'y': 1, 'layer': 'Serving'},
    {'name': 'Prediction<br>(Real-time)', 'x': 5, 'y': 1, 'layer': 'Serving'},
    
    # Monitoring Layer
    {'name': 'Drift Detect<br>(Alibi)', 'x': 1, 'y': 0, 'layer': 'Monitoring'},
    {'name': 'Logging', 'x': 3, 'y': 0, 'layer': 'Monitoring'},
    {'name': 'Health Monitoring<br>(Prometheus)', 'x': 5, 'y': 0, 'layer': 'Monitoring'}
]

# Add component boxes
for comp in components:
    layer_info = layers[comp['layer']]
    
    # Add rectangle for component
    fig.add_shape(
        type="rect",
        x0=comp['x']-0.4, y0=comp['y']-0.3,
        x1=comp['x']+0.4, y1=comp['y']+0.3,
        fillcolor=layer_info['color'],
        line=dict(color=layer_info['border'], width=2),
    )
    
    # Add text label
    fig.add_annotation(
        x=comp['x'], y=comp['y'],
        text=comp['name'],
        showarrow=False,
        font=dict(size=12, color='black'),
        align='center'
    )

# Add arrows for data flow
arrows = [
    # Data flow
    {'from': (1, 5), 'to': (1, 4), 'label': 'Raw Data'},
    {'from': (3, 5), 'to': (1, 4), 'label': 'Versions'},
    {'from': (1, 4), 'to': (3, 4), 'label': 'Clean Data'},
    {'from': (3, 4), 'to': (5, 4), 'label': 'Processed'},
    {'from': (5, 4), 'to': (5, 5), 'label': 'Features'},
    {'from': (5, 4), 'to': (1, 3), 'label': 'Train Data'},
    
    # ML flow
    {'from': (1, 3), 'to': (3, 3), 'label': 'Model'},
    {'from': (3, 3), 'to': (5, 3), 'label': 'Validated'},
    {'from': (5, 3), 'to': (3, 1), 'label': 'Artifacts'},
    
    # Orchestration
    {'from': (2, 2), 'to': (4, 2), 'label': 'Schedule'},
    {'from': (4, 2), 'to': (1, 3), 'label': 'Trigger'},
    
    # Serving
    {'from': (3, 1), 'to': (5, 1), 'label': 'Load'},
    {'from': (1, 1), 'to': (5, 1), 'label': 'Request'},
    {'from': (5, 5), 'to': (5, 1), 'label': 'Features'},
    
    # Monitoring
    {'from': (5, 1), 'to': (1, 0), 'label': 'Metrics'},
    {'from': (1, 1), 'to': (3, 0), 'label': 'Logs'},
    {'from': (5, 1), 'to': (5, 0), 'label': 'Health'}
]

# Add arrows
for arrow in arrows:
    x0, y0 = arrow['from']
    x1, y1 = arrow['to']
    
    # Add arrow line
    fig.add_annotation(
        x=x1, y=y1,
        ax=x0, ay=y0,
        xref='x', yref='y',
        axref='x', ayref='y',
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor='#333333'
    )

# Add layer labels
for layer_name, layer_info in layers.items():
    fig.add_annotation(
        x=-0.5, y=layer_info['y'],
        text=f"<b>{layer_name}</b>",
        showarrow=False,
        font=dict(size=14, color='black'),
        align='right',
        bgcolor=layer_info['color'],
        bordercolor=layer_info['border'],
        borderwidth=2
    )

# Update layout
fig.update_layout(
    title="MLOps Customer Churn Prediction Architecture",
    showlegend=False,
    xaxis=dict(range=[-1, 6], showgrid=False, showticklabels=False, zeroline=False),
    yaxis=dict(range=[-0.5, 5.5], showgrid=False, showticklabels=False, zeroline=False),
    plot_bgcolor='white',
    font=dict(family="Arial, sans-serif", size=12)
)

# Save the chart
fig.write_image("mlops_architecture_final.png")
fig.write_image("mlops_architecture_final.svg", format="svg")

print("MLOps architecture diagram created successfully!")
print("Files saved: mlops_architecture_final.png and mlops_architecture_final.svg")