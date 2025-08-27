#!/usr/bin/env python3
"""
Create a shallow cascade visualization for non-rumor propagation.
Emphasizes the width and star-like structure of verified information cascades.
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
import random

def create_nonrumor_cascade():
    """Create a shallow, wide cascade structure typical of non-rumors/verified information."""
    G = nx.DiGraph()
    
    # Root node (original tweet)
    G.add_node(0, level=0, timestamp=0, engagement='source')
    
    node_id = 1
    
    # Non-rumors have different propagation patterns:
    # - Most responses are direct (level 1)
    # - Few deep threads (max 4-5 levels)
    # - Wide initial spread (viral explosion)
    
    # Level 1: Massive initial spread (viral explosion)
    level_1_nodes = random.randint(40, 60)  # Large number of direct shares/retweets
    level_1_ids = []
    for _ in range(level_1_nodes):
        G.add_node(node_id, level=1, timestamp=random.randint(0, 30), engagement='share')
        G.add_edge(0, node_id)
        level_1_ids.append(node_id)
        node_id += 1
    
    # Level 2: Some secondary sharing (but much less than level 1)
    level_2_prob = 0.3  # Only 30% of level 1 nodes generate responses
    level_2_ids = []
    for parent in level_1_ids:
        if random.random() < level_2_prob:
            num_responses = random.randint(1, 3)  # Few responses per node
            for _ in range(num_responses):
                G.add_node(node_id, level=2, timestamp=random.randint(30, 60), engagement='reshare')
                G.add_edge(parent, node_id)
                level_2_ids.append(node_id)
                node_id += 1
    
    # Level 3: Minimal further propagation
    level_3_prob = 0.15  # Only 15% continue
    level_3_ids = []
    for parent in level_2_ids:
        if random.random() < level_3_prob:
            num_responses = random.randint(1, 2)
            for _ in range(num_responses):
                G.add_node(node_id, level=3, timestamp=random.randint(60, 90), engagement='comment')
                G.add_edge(parent, node_id)
                level_3_ids.append(node_id)
                node_id += 1
    
    # Level 4: Very rare deep threads (only a few)
    level_4_prob = 0.1
    level_4_ids = []
    for parent in level_3_ids[:5]:  # Only first few nodes might continue
        if random.random() < level_4_prob:
            G.add_node(node_id, level=4, timestamp=random.randint(90, 120), engagement='discussion')
            G.add_edge(parent, node_id)
            level_4_ids.append(node_id)
            node_id += 1
    
    # Rarely, level 5 (maximum depth for non-rumors)
    if level_4_ids and random.random() < 0.3:
        for parent in level_4_ids[:2]:  # At most 2 threads reach level 5
            G.add_node(node_id, level=5, timestamp=random.randint(120, 150), engagement='late_comment')
            G.add_edge(parent, node_id)
            node_id += 1
    
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
    
    # Position nodes - wider spread for shallow structure
    for level, nodes in levels.items():
        n_nodes = len(nodes)
        for i, node in enumerate(sorted(nodes)):  # Sort for consistent layout
            if level == 0:
                # Root at center
                x = 0
            else:
                # Wide horizontal spread for each level
                spread_factor = 15 if level == 1 else 12  # Wider spread at level 1
                x = (i - n_nodes / 2) * (spread_factor / max(1, n_nodes)) + random.uniform(-0.2, 0.2)
            
            y = -level * 3  # Vertical spacing between levels
            pos[node] = (x, y)
    
    return pos

def visualize_nonrumor_cascade(G):
    """Create a visualization emphasizing width and shallow structure."""
    # Calculate max depth to adjust figure size
    max_depth = max(G.nodes[n]['level'] for n in G.nodes())
    fig_height = max(12, max_depth * 3)  # Smaller height for shallow tree
    
    fig, ax = plt.subplots(figsize=(20, fig_height))
    
    # Calculate positions using custom hierarchical layout
    pos = create_hierarchical_positions(G)
    
    # Normalize positions for better visualization
    positions = np.array(list(pos.values()))
    min_pos = positions.min(axis=0)
    max_pos = positions.max(axis=0)
    scale = max_pos - min_pos
    
    if scale[0] > 0 and scale[1] > 0:
        for node in pos:
            x, y = pos[node]
            # Adjust scaling to show wide spread
            pos[node] = ((x - min_pos[0]) / scale[0] * 18 + 1,
                         (y - min_pos[1]) / scale[1] * (fig_height - 2) + 1)
    
    # Color scheme for non-rumor (green/blue gradient based on level)
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        level = G.nodes[node]['level']
        if node == 0:  # Root node
            node_colors.append('#00AA00')  # Bright green for verified source
            node_sizes.append(1500)  # Large source node
        else:
            # Gradient from green to blue based on level
            if level == 1:
                # Level 1: Green (immediate shares)
                node_colors.append('#00CC66')
                node_sizes.append(600)
            elif level == 2:
                # Level 2: Teal
                node_colors.append('#00CCAA')
                node_sizes.append(500)
            elif level == 3:
                # Level 3: Light blue
                node_colors.append('#0099CC')
                node_sizes.append(400)
            else:
                # Level 4-5: Blue
                node_colors.append('#0066CC')
                node_sizes.append(350)
    
    # Draw edges with varying styles
    edge_colors = []
    edge_widths = []
    for u, v in G.edges():
        level_v = G.nodes[v]['level']
        if level_v == 1:
            edge_colors.append('#00AA44')  # Green for immediate spread
            edge_widths.append(2.5)
        elif level_v == 2:
            edge_colors.append('#00AAAA')  # Teal
            edge_widths.append(2.0)
        else:
            edge_colors.append('#0088CC')  # Blue
            edge_widths.append(1.5)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos,
                          edge_color=edge_colors,
                          width=edge_widths,
                          alpha=0.7,
                          arrows=True,
                          arrowsize=10,
                          arrowstyle='->')
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos,
                          node_color=node_colors,
                          node_size=node_sizes,
                          alpha=0.9,
                          edgecolors='darkgreen',
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
    ax.set_title('NON-RUMOR CASCADE STRUCTURE\nWide Spread with Shallow Propagation',
                fontsize=20, fontweight='bold', pad=30, y=1.02, color='darkgreen')
    
    # Add example tweet content at the very top
    tweet_text = 'NON-RUMOR EXAMPLE TWEET:\n"Official: WHO announces successful vaccine trial results \nwith 95% efficacy. Peer-reviewed study published in \nThe Lancet. Link to full report: [verified URL]"'
    ax.text(0.5, 1.08, tweet_text, 
            transform=ax.transAxes, ha='center', va='top',
            fontsize=11, style='italic', color='darkgreen',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#E5FFE5', edgecolor='green', alpha=0.8))
    
    # Add legend box - position in top-right, outside tree area
    legend_x = max(all_x) + 3  # Move further right
    legend_y = max(all_y) + 0.5  # Move up
    
    legend_box = FancyBboxPatch((legend_x - 0.5, legend_y - 2.5), 5, 3,
                                boxstyle="round,pad=0.1",
                                facecolor='white',
                                edgecolor='green',
                                alpha=0.95)
    ax.add_patch(legend_box)
    
    ax.text(legend_x + 2, legend_y - 0.3, 'NON-RUMOR CHARACTERISTICS', 
            fontsize=12, fontweight='bold', ha='center', color='darkgreen')
    ax.text(legend_x, legend_y - 0.8, f'• Nodes: {G.number_of_nodes()}', fontsize=10)
    ax.text(legend_x, legend_y - 1.2, f'• Max Depth: {max_depth} levels', fontsize=10)
    ax.text(legend_x, legend_y - 1.6, '• Structure: Wide star-like', fontsize=10)
    ax.text(legend_x, legend_y - 2.0, '• Color: Green/Blue (verified)', fontsize=10)
    ax.text(legend_x, legend_y - 2.4, '• Pattern: Quick viral spread', fontsize=10)
    
    # Add source annotation to the left side (not overlapping)
    source_x = min(all_x) - 3
    ax.annotate('Verified Source\n(Original Tweet)', xy=pos[0], 
                xytext=(source_x, pos[0][1]),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=11, fontweight='bold', color='darkgreen',
                ha='right')
    
    # Annotate the viral spread at level 1 - position to the right
    level_1_nodes = [n for n in G.nodes() if G.nodes[n]['level'] == 1]
    if len(level_1_nodes) > 10:
        # Choose a node on the right side
        sample_node = level_1_nodes[-5]  # Near the end for right side
        spread_x = max(all_x) + 2
        ax.annotate('Viral Spread\n(Direct Shares)', xy=pos[sample_node], 
                   xytext=(spread_x, pos[sample_node][1]),
                   arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                   fontsize=10, color='darkgreen',
                   ha='left')
    
    # Count nodes per level for statistics
    level_counts = {}
    for node in G.nodes():
        level = G.nodes[node]['level']
        level_counts[level] = level_counts.get(level, 0) + 1
    
    # Add distribution info - position at bottom left, outside tree
    info_text = "Level Distribution:\n"
    for level in sorted(level_counts.keys()):
        info_text += f"L{level}: {level_counts[level]} nodes\n"
    
    info_x = min(all_x) - 4
    info_y = min(all_y) + 1
    ax.text(info_x, info_y, info_text.strip(), 
            fontsize=9, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3),
            ha='left')
    
    # Style the plot - adjust limits to show ALL nodes and annotations
    x_margin = 5  # Increased to accommodate side annotations
    y_margin = 3
    ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin + 3)
    ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)
    ax.axis('off')
    ax.set_facecolor('#F0FFF0')  # Light green background
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    return fig

def main():
    """Generate and save non-rumor cascade visualization."""
    print("Generating shallow non-rumor cascade structure...")
    
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Create cascade
    G = create_nonrumor_cascade()
    print(f"Created cascade with {G.number_of_nodes()} nodes")
    print(f"Maximum depth: {max(G.nodes[n]['level'] for n in G.nodes())} levels")
    
    # Count nodes per level
    level_counts = {}
    for node in G.nodes():
        level = G.nodes[node]['level']
        level_counts[level] = level_counts.get(level, 0) + 1
    
    print("Level distribution:")
    for level in sorted(level_counts.keys()):
        print(f"  Level {level}: {level_counts[level]} nodes")
    
    # Visualize
    print("Creating visualization...")
    fig = visualize_nonrumor_cascade(G)
    
    # Save
    output_path = 'visualizations/nonrumor_cascade_shallow.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Visualization saved to {output_path}")
    
    # Also save a PDF version
    output_pdf = 'visualizations/nonrumor_cascade_shallow.pdf'
    fig.savefig(output_pdf, bbox_inches='tight', facecolor='white')
    print(f"PDF version saved to {output_pdf}")
    
    plt.show()

if __name__ == "__main__":
    main()