#!/usr/bin/env python3
"""
Create a deep cascade visualization for rumor propagation.
Emphasizes the depth and complexity of rumor cascades with multiple debate threads.
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
import random

def create_rumor_cascade():
    """Create a deep, complex cascade structure typical of rumors."""
    G = nx.DiGraph()
    
    # Root node (original tweet)
    G.add_node(0, level=0, debate_thread=0, timestamp=0)
    
    node_id = 1
    
    # Create main debate threads with different depths (8-12 levels for VERY deep tree)
    main_threads = [
        {'depth': 12, 'branch_prob': 0.6, 'multi_response_prob': 0.5},  # Very long debate
        {'depth': 10, 'branch_prob': 0.5, 'multi_response_prob': 0.45},  # Extended discussion
        {'depth': 9, 'branch_prob': 0.45, 'multi_response_prob': 0.4},   # Deep questioning
        {'depth': 8, 'branch_prob': 0.4, 'multi_response_prob': 0.35},   # Verification attempts
        {'depth': 7, 'branch_prob': 0.35, 'multi_response_prob': 0.3},   # Counter-arguments
        {'depth': 8, 'branch_prob': 0.5, 'multi_response_prob': 0.4},   # Additional debate
        {'depth': 6, 'branch_prob': 0.6, 'multi_response_prob': 0.3},   # Side discussions
    ]
    
    for thread_id, thread in enumerate(main_threads):
        # Start each thread from root or early nodes
        parent = 0 if thread_id < 2 else random.choice(range(min(5, node_id)))
        current_branch = [parent]
        
        for level in range(1, thread['depth'] + 1):
            new_branch = []
            for parent_node in current_branch:
                # Determine number of responses (irregular branching)
                if random.random() < thread['multi_response_prob']:
                    num_responses = random.randint(2, 4)  # Multiple responses (debate)
                elif random.random() < thread['branch_prob']:
                    num_responses = 1  # Single response
                else:
                    num_responses = 0  # Dead end
                
                for _ in range(num_responses):
                    G.add_node(node_id, 
                             level=level,
                             debate_thread=thread_id,
                             timestamp=level * 100 + random.randint(0, 50))
                    G.add_edge(parent_node, node_id)
                    new_branch.append(node_id)
                    node_id += 1
            
            current_branch = new_branch
            if not current_branch:  # If branch died out, restart from another node
                if level < thread['depth'] - 1:
                    current_branch = [random.choice(list(G.nodes())[:max(1, node_id//2)])]
    
    # Add some cross-connections (debates referencing each other)
    nodes = list(G.nodes())
    for _ in range(5):
        if len(nodes) > 10:
            source = random.choice(nodes[5:])
            target = random.choice(nodes[5:])
            if source != target and not G.has_edge(source, target):
                G.add_edge(source, target, cross_reference=True)
    
    return G

def create_hierarchical_positions(G):
    """Create hierarchical positions for nodes based on their levels."""
    pos = {}
    levels = {}
    
    # Group nodes by level
    for node in G.nodes():
        level = G.nodes[node]['level']
        if level not in levels:
            levels[level] = []
        levels[level].append(node)
    
    # Position nodes
    for level, nodes in levels.items():
        n_nodes = len(nodes)
        for i, node in enumerate(nodes):
            # Spread nodes horizontally at each level
            x = (i - n_nodes / 2) * (10 / max(1, n_nodes)) + random.uniform(-0.5, 0.5)
            y = -level * 3  # Vertical spacing between levels
            pos[node] = (x, y)
    
    return pos

def visualize_rumor_cascade(G):
    """Create a visualization emphasizing depth and debate structure."""
    # Calculate max depth to adjust figure size
    max_depth = max(G.nodes[n]['level'] for n in G.nodes())
    fig_height = max(24, max_depth * 2.5)  # Dynamic height based on depth
    
    fig, ax = plt.subplots(figsize=(20, fig_height))
    
    # Calculate positions using custom hierarchical layout to emphasize depth
    pos = create_hierarchical_positions(G)
    
    # Normalize positions for better visualization
    positions = np.array(list(pos.values()))
    min_pos = positions.min(axis=0)
    max_pos = positions.max(axis=0)
    scale = max_pos - min_pos
    
    if scale[0] > 0 and scale[1] > 0:
        for node in pos:
            x, y = pos[node]
            # Adjust scaling to show all nodes properly
            pos[node] = ((x - min_pos[0]) / scale[0] * 18 + 1,
                         (y - min_pos[1]) / scale[1] * (fig_height - 2) + 1)
    
    # Color scheme for rumor (red/orange gradient based on depth)
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        level = G.nodes[node]['level']
        # Deeper nodes get darker red colors
        color_intensity = 0.3 + (level / 8) * 0.7
        if node == 0:  # Root node
            node_colors.append('#FF0000')
            node_sizes.append(1200)
        else:
            # Gradient from orange to dark red based on depth
            red = min(1.0, 1.0)
            green = min(1.0, max(0, 0.6 * (1 - color_intensity)))
            blue = min(1.0, max(0, 0.1 * (1 - color_intensity)))
            node_colors.append((red, green, blue))
            # Smaller nodes as we go deeper (but ensure minimum size)
            node_sizes.append(max(300, 800 - level * 60))
    
    # Draw edges with varying styles
    regular_edges = [(u, v) for u, v in G.edges() if not G.edges[u, v].get('cross_reference', False)]
    cross_edges = [(u, v) for u, v in G.edges() if G.edges[u, v].get('cross_reference', False)]
    
    # Regular edges (tree structure)
    nx.draw_networkx_edges(G, pos, 
                          edgelist=regular_edges,
                          edge_color='#FF6B6B',
                          width=2,
                          alpha=0.6,
                          arrows=True,
                          arrowsize=10,
                          arrowstyle='->')
    
    # Cross-reference edges (debates)
    if cross_edges:
        nx.draw_networkx_edges(G, pos,
                              edgelist=cross_edges,
                              edge_color='#FF9999',
                              width=1,
                              alpha=0.4,
                              style='dashed',
                              arrows=True,
                              arrowsize=8)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos,
                          node_color=node_colors,
                          node_size=node_sizes,
                          alpha=0.9,
                          edgecolors='darkred',
                          linewidths=2)
    
    # Add level indicators
    levels = {}
    for node in G.nodes():
        level = G.nodes[node]['level']
        if level not in levels:
            levels[level] = []
        levels[level].append(pos[node][1])
    
    # Draw level lines and labels
    for level, y_positions in levels.items():
        if y_positions:
            avg_y = np.mean(y_positions)
            ax.axhline(y=avg_y, color='gray', linestyle=':', alpha=0.3)
            ax.text(-0.5, avg_y, f'Level {level}', 
                   fontsize=10, ha='right', va='center',
                   color='gray', fontweight='bold')
    
    # Get all positions for calculating plot area
    all_x = [pos[n][0] for n in G.nodes()]
    all_y = [pos[n][1] for n in G.nodes()]
    
    # Add title at the top with more padding
    ax.set_title('RUMOR CASCADE STRUCTURE\nDeep Propagation with Multiple Debate Threads',
                fontsize=20, fontweight='bold', pad=30, y=1.02)
    
    # Add example tweet content at the very top
    tweet_text = 'RUMOR EXAMPLE TWEET:\n"BREAKING: Major bank reportedly facing collapse after \nhidden losses discovered. Multiple sources confirm. \nPlease share to warn others! #BankingCrisis"'
    ax.text(0.5, 1.08, tweet_text, 
            transform=ax.transAxes, ha='center', va='top',
            fontsize=11, style='italic', color='darkred',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFE5E5', edgecolor='red', alpha=0.8))
    
    # Add legend box - position it in the top-right corner, outside the tree area
    legend_x = max(all_x) + 3  # Move further right
    legend_y = max(all_y) + 1  # Move up
    
    legend_box = FancyBboxPatch((legend_x - 0.5, legend_y - 3), 5, 3,
                                boxstyle="round,pad=0.1",
                                facecolor='white',
                                edgecolor='gray',
                                alpha=0.95)
    ax.add_patch(legend_box)
    
    ax.text(legend_x + 2, legend_y - 0.3, 'RUMOR CHARACTERISTICS', fontsize=12, fontweight='bold', ha='center')
    ax.text(legend_x, legend_y - 0.8, f'• Nodes: {G.number_of_nodes()}', fontsize=10)
    ax.text(legend_x, legend_y - 1.2, f'• Max Depth: {max(G.nodes[n]["level"] for n in G.nodes())} levels', fontsize=10)
    ax.text(legend_x, legend_y - 1.6, f'• Debate Threads: 7 major', fontsize=10)
    ax.text(legend_x, legend_y - 2.0, '• Color: Red/Orange (controversy)', fontsize=10)
    ax.text(legend_x, legend_y - 2.4, '• Pattern: Deep debates & verification', fontsize=10)
    
    # Add source annotation to the left side (not overlapping)
    source_x = min(all_x) - 3
    ax.annotate('Original Tweet\n(Source)', xy=pos[0], 
                xytext=(source_x, pos[0][1]),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, fontweight='bold', color='red',
                ha='right')
    
    # Find and annotate deep debate threads - position to the right
    deepest_level = max(G.nodes[n]['level'] for n in G.nodes())
    deepest_nodes = [n for n in G.nodes() if G.nodes[n]['level'] >= deepest_level - 1]
    if deepest_nodes:
        # Choose a node that's more centered horizontally
        sample_deep = min(deepest_nodes, key=lambda n: abs(pos[n][0]))
        deep_x = max(all_x) + 2
        ax.annotate(f'Deep Debate\n(Level {G.nodes[sample_deep]["level"]})', xy=pos[sample_deep], 
                   xytext=(deep_x, pos[sample_deep][1]),
                   arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5),
                   fontsize=10, color='darkred',
                   ha='left')
    
    # Style the plot - adjust limits to show ALL nodes and annotations
    x_margin = 5  # Increased to accommodate side annotations
    y_margin = 3
    ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin + 3)
    ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)
    ax.axis('off')
    ax.set_facecolor('#FFF5F5')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    return fig

def main():
    """Generate and save rumor cascade visualization."""
    print("Generating deep rumor cascade structure...")
    
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Create cascade
    G = create_rumor_cascade()
    print(f"Created cascade with {G.number_of_nodes()} nodes")
    print(f"Maximum depth: {max(G.nodes[n]['level'] for n in G.nodes())} levels")
    
    # Visualize
    print("Creating visualization...")
    fig = visualize_rumor_cascade(G)
    
    # Save
    output_path = 'visualizations/rumor_cascade_deep.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Visualization saved to {output_path}")
    
    # Also save a PDF version
    output_pdf = 'visualizations/rumor_cascade_deep.pdf'
    fig.savefig(output_pdf, bbox_inches='tight', facecolor='white')
    print(f"PDF version saved to {output_pdf}")
    
    plt.show()

if __name__ == "__main__":
    main()