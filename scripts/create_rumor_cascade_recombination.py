#!/usr/bin/env python3
"""
Create a rumor cascade visualization with colored phylogenetic recombinations.
Identifies and colors different sub-cascades that were merged to form the main cascade.
Based on the recombination detection algorithm with similarity threshold ρ = 0.7
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
import random
import colorsys
from collections import defaultdict

def create_rumor_cascade_with_recombination():
    """Create a deep cascade structure with tracked lineages for recombination detection."""
    G = nx.DiGraph()
    
    # Track lineage information for each node
    lineages = {}  # node -> lineage_id
    lineage_roots = {}  # lineage_id -> root_node
    lineage_counter = 0
    
    # Track semantic similarity between nodes
    semantic_similarities = {}
    
    # Root node (original tweet)
    G.add_node(0, level=0, debate_thread=0, timestamp=0, lineage=lineage_counter)
    lineages[0] = lineage_counter
    lineage_roots[lineage_counter] = 0
    lineage_counter += 1
    
    node_id = 1
    
    # Create main debate threads with different depths
    main_threads = [
        {'depth': 12, 'branch_prob': 0.6, 'multi_response_prob': 0.5, 'recomb_prob': 0.3},
        {'depth': 10, 'branch_prob': 0.5, 'multi_response_prob': 0.45, 'recomb_prob': 0.25},
        {'depth': 9, 'branch_prob': 0.45, 'multi_response_prob': 0.4, 'recomb_prob': 0.2},
        {'depth': 8, 'branch_prob': 0.4, 'multi_response_prob': 0.35, 'recomb_prob': 0.15},
        {'depth': 7, 'branch_prob': 0.35, 'multi_response_prob': 0.3, 'recomb_prob': 0.2},
        {'depth': 8, 'branch_prob': 0.5, 'multi_response_prob': 0.4, 'recomb_prob': 0.25},
        {'depth': 6, 'branch_prob': 0.6, 'multi_response_prob': 0.3, 'recomb_prob': 0.1},
    ]
    
    # Track nodes by timestamp for recombination detection
    nodes_by_time = defaultdict(list)
    
    for thread_id, thread in enumerate(main_threads):
        # Potentially start a new lineage for this thread (recombination event)
        if thread_id > 0 and random.random() < thread['recomb_prob']:
            # This thread represents a recombined cascade
            parent = 0 if thread_id < 2 else random.choice(range(min(5, node_id)))
            current_lineage = lineage_counter
            lineage_counter += 1
            lineage_roots[current_lineage] = node_id  # First node of new lineage
        else:
            # Continue existing lineage
            parent = 0 if thread_id < 2 else random.choice(range(min(5, node_id)))
            if parent in lineages:
                current_lineage = lineages[parent]
            else:
                current_lineage = 0
        
        current_branch = [parent]
        
        for level in range(1, thread['depth'] + 1):
            new_branch = []
            for parent_node in current_branch:
                # Determine number of responses
                if random.random() < thread['multi_response_prob']:
                    num_responses = random.randint(2, 4)
                elif random.random() < thread['branch_prob']:
                    num_responses = 1
                else:
                    num_responses = 0
                
                for _ in range(num_responses):
                    timestamp = level * 100 + random.randint(0, 50)
                    
                    # Check for potential recombination
                    is_recombination = False
                    recomb_source = None
                    
                    if level > 2 and random.random() < thread['recomb_prob'] * 0.5:
                        # Look for candidates from different lineages with high similarity
                        candidates = []
                        for existing_node in G.nodes():
                            if existing_node != node_id and existing_node != parent_node:
                                existing_time = G.nodes[existing_node].get('timestamp', 0)
                                if existing_time < timestamp:
                                    # Check if from different lineage
                                    if existing_node in lineages:
                                        existing_lineage = lineages[existing_node]
                                        if existing_lineage != current_lineage:
                                            # Simulate semantic similarity (ρ threshold = 0.7)
                                            similarity = random.uniform(0.5, 1.0)
                                            if similarity >= 0.7:
                                                candidates.append((existing_node, similarity, existing_lineage))
                        
                        if candidates:
                            # Select best candidate for recombination
                            recomb_source, sim, source_lineage = max(candidates, key=lambda x: x[1])
                            is_recombination = True
                            # Create new lineage for recombined subtree
                            current_lineage = lineage_counter
                            lineage_counter += 1
                            lineage_roots[current_lineage] = node_id
                            semantic_similarities[(recomb_source, node_id)] = sim
                    
                    G.add_node(node_id, 
                             level=level,
                             debate_thread=thread_id,
                             timestamp=timestamp,
                             lineage=current_lineage,
                             is_recombination=is_recombination)
                    
                    lineages[node_id] = current_lineage
                    nodes_by_time[timestamp].append(node_id)
                    
                    # Add primary edge
                    G.add_edge(parent_node, node_id, edge_type='primary')
                    
                    # Add recombination edge if applicable
                    if is_recombination and recomb_source is not None:
                        G.add_edge(recomb_source, node_id, 
                                 edge_type='recombination',
                                 similarity=semantic_similarities[(recomb_source, node_id)])
                    
                    new_branch.append(node_id)
                    node_id += 1
            
            current_branch = new_branch
            if not current_branch:
                if level < thread['depth'] - 1:
                    current_branch = [random.choice(list(G.nodes())[:max(1, node_id//2)])]
    
    return G, lineages, lineage_roots

def get_lineage_colors(lineages, lineage_roots):
    """Generate distinct colors for each lineage/recombined cascade."""
    unique_lineages = set(lineages.values())
    n_lineages = len(unique_lineages)
    
    # Generate distinct colors using HSV color space
    colors = {}
    
    # Main lineage (0) gets red color
    colors[0] = '#FF0000'
    
    # Other lineages get distinct colors
    hue_step = 360 / (n_lineages + 1)
    color_idx = 1
    
    for lineage_id in sorted(unique_lineages):
        if lineage_id == 0:
            continue
        
        # Generate color with good saturation and value
        hue = (color_idx * hue_step) % 360
        # Avoid red hues (reserved for main cascade)
        if 330 < hue or hue < 30:
            hue = (hue + 60) % 360
        
        rgb = colorsys.hsv_to_rgb(hue/360, 0.8, 0.9)
        colors[lineage_id] = '#%02x%02x%02x' % tuple(int(c*255) for c in rgb)
        color_idx += 1
    
    return colors

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
    
    # Position nodes with clustering by lineage
    for level, nodes in levels.items():
        # Group nodes by lineage for better visual clustering
        lineage_groups = defaultdict(list)
        for node in nodes:
            lineage = G.nodes[node].get('lineage', 0)
            lineage_groups[lineage].append(node)
        
        # Position each lineage group
        n_groups = len(lineage_groups)
        group_width = 20 / max(1, n_groups)
        
        group_idx = 0
        for lineage, group_nodes in sorted(lineage_groups.items()):
            group_center = (group_idx - n_groups/2) * group_width
            n_nodes = len(group_nodes)
            
            for i, node in enumerate(group_nodes):
                # Position within group
                x = group_center + (i - n_nodes/2) * (group_width / max(1, n_nodes)) * 0.8
                x += random.uniform(-0.2, 0.2)  # Small random offset
                y = -level * 3
                pos[node] = (x, y)
            
            group_idx += 1
    
    return pos

def visualize_rumor_cascade_with_recombination(G, lineages, lineage_roots):
    """Create visualization with colored recombined cascades."""
    # Calculate max depth
    max_depth = max(G.nodes[n]['level'] for n in G.nodes())
    fig_height = max(24, max_depth * 2.5)
    
    fig, ax = plt.subplots(figsize=(22, fig_height))
    
    # Get positions
    pos = create_hierarchical_positions(G)
    
    # Normalize positions
    positions = np.array(list(pos.values()))
    min_pos = positions.min(axis=0)
    max_pos = positions.max(axis=0)
    scale = max_pos - min_pos
    
    if scale[0] > 0 and scale[1] > 0:
        for node in pos:
            x, y = pos[node]
            pos[node] = ((x - min_pos[0]) / scale[0] * 20 + 1,
                         (y - min_pos[1]) / scale[1] * (fig_height - 2) + 1)
    
    # Get colors for each lineage
    lineage_colors = get_lineage_colors(lineages, lineage_roots)
    
    # Prepare node colors and sizes
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        lineage = G.nodes[node].get('lineage', 0)
        level = G.nodes[node]['level']
        is_recomb = G.nodes[node].get('is_recombination', False)
        
        # Get base color from lineage
        node_colors.append(lineage_colors[lineage])
        
        # Size based on importance
        if node == 0:  # Root
            node_sizes.append(1200)
        elif is_recomb:  # Recombination point
            node_sizes.append(900)
        elif node in lineage_roots.values():  # Lineage root
            node_sizes.append(800)
        else:
            node_sizes.append(max(300, 700 - level * 40))
    
    # Separate edges by type
    primary_edges = [(u, v) for u, v, d in G.edges(data=True) 
                     if d.get('edge_type', 'primary') == 'primary']
    recomb_edges = [(u, v) for u, v, d in G.edges(data=True) 
                    if d.get('edge_type', 'primary') == 'recombination']
    
    # Draw primary edges (colored by source lineage)
    for u, v in primary_edges:
        source_lineage = lineages.get(u, 0)
        edge_color = lineage_colors[source_lineage]
        nx.draw_networkx_edges(G, pos, 
                              edgelist=[(u, v)],
                              edge_color=edge_color,
                              width=2,
                              alpha=0.6,
                              arrows=True,
                              arrowsize=10,
                              arrowstyle='->')
    
    # Draw recombination edges (dashed, shows mixing)
    if recomb_edges:
        for u, v in recomb_edges:
            sim = G.edges[u, v].get('similarity', 0.7)
            # Thicker line for higher similarity
            width = 1 + (sim - 0.7) * 4
            nx.draw_networkx_edges(G, pos,
                                  edgelist=[(u, v)],
                                  edge_color='purple',
                                  width=width,
                                  alpha=0.7,
                                  style='dashed',
                                  arrows=True,
                                  arrowsize=12,
                                  arrowstyle='->')
    
    # Draw nodes with borders indicating recombination
    for i, node in enumerate(G.nodes()):
        is_recomb = G.nodes[node].get('is_recombination', False)
        edge_color = 'purple' if is_recomb else 'darkgray'
        edge_width = 3 if is_recomb else 2
        
        nx.draw_networkx_nodes(G, pos,
                              nodelist=[node],
                              node_color=[node_colors[i]],
                              node_size=[node_sizes[i]],
                              alpha=0.9,
                              edgecolors=edge_color,
                              linewidths=edge_width)
    
    # Add level indicators
    levels = {}
    for node in G.nodes():
        level = G.nodes[node]['level']
        if level not in levels:
            levels[level] = []
        levels[level].append(pos[node][1])
    
    for level, y_positions in levels.items():
        if y_positions:
            avg_y = np.mean(y_positions)
            ax.axhline(y=avg_y, color='gray', linestyle=':', alpha=0.3)
            ax.text(-0.5, avg_y, f'Level {level}', 
                   fontsize=10, ha='right', va='center',
                   color='gray', fontweight='bold')
    
    # Get plot boundaries
    all_x = [pos[n][0] for n in G.nodes()]
    all_y = [pos[n][1] for n in G.nodes()]
    
    # Title
    ax.set_title('RUMOR CASCADE WITH PHYLOGENETIC RECOMBINATION\nColored Sub-Cascades Show Different Merged Lineages',
                fontsize=20, fontweight='bold', pad=30, y=1.02)
    
    # Example tweet
    tweet_text = 'RUMOR WITH RECOMBINATION:\nMultiple narrative threads merge as different groups\nadd their own interpretations and evidence to the rumor'
    ax.text(0.5, 1.08, tweet_text, 
            transform=ax.transAxes, ha='center', va='top',
            fontsize=11, style='italic', color='darkred',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFE5E5', edgecolor='red', alpha=0.8))
    
    # Legend for lineages
    legend_x = max(all_x) + 3
    legend_y = max(all_y) + 1
    
    # Count nodes per lineage
    lineage_counts = defaultdict(int)
    recomb_count = 0
    for node in G.nodes():
        lineage_counts[lineages[node]] += 1
        if G.nodes[node].get('is_recombination', False):
            recomb_count += 1
    
    # Legend box
    n_lineages = len(set(lineages.values()))
    box_height = 4 + n_lineages * 0.4
    legend_box = FancyBboxPatch((legend_x - 0.5, legend_y - box_height), 6, box_height,
                                boxstyle="round,pad=0.1",
                                facecolor='white',
                                edgecolor='gray',
                                alpha=0.95)
    ax.add_patch(legend_box)
    
    ax.text(legend_x + 2.5, legend_y - 0.3, 'RECOMBINATION ANALYSIS', 
           fontsize=12, fontweight='bold', ha='center')
    
    # Show each lineage
    y_offset = 0.8
    for lineage_id in sorted(set(lineages.values())):
        color = lineage_colors[lineage_id]
        count = lineage_counts[lineage_id]
        
        # Draw color indicator
        ax.plot([legend_x, legend_x + 0.5], 
               [legend_y - y_offset, legend_y - y_offset],
               color=color, linewidth=4)
        
        # Label
        if lineage_id == 0:
            label = f'Main cascade: {count} nodes'
        else:
            label = f'Lineage {lineage_id}: {count} nodes'
        ax.text(legend_x + 0.7, legend_y - y_offset, label, 
               fontsize=10, va='center')
        y_offset += 0.4
    
    # Add recombination statistics
    y_offset += 0.3
    ax.text(legend_x, legend_y - y_offset, 
           f'• Recombination points: {recomb_count}', fontsize=10)
    ax.text(legend_x, legend_y - y_offset - 0.4, 
           f'• Total nodes: {G.number_of_nodes()}', fontsize=10)
    ax.text(legend_x, legend_y - y_offset - 0.8, 
           f'• Max depth: {max_depth} levels', fontsize=10)
    ax.text(legend_x, legend_y - y_offset - 1.2, 
           '• Purple edges: Recombination (ρ ≥ 0.7)', fontsize=10)
    
    # Annotate some recombination points
    recomb_nodes = [n for n in G.nodes() if G.nodes[n].get('is_recombination', False)]
    if recomb_nodes and len(recomb_nodes) > 0:
        # Annotate first few recombination points
        for i, recomb_node in enumerate(recomb_nodes[:3]):
            if i == 0:
                annotation_x = min(all_x) - 3
                ha = 'right'
            else:
                annotation_x = max(all_x) + 2
                ha = 'left'
            
            ax.annotate(f'Recombination\nPoint', 
                       xy=pos[recomb_node], 
                       xytext=(annotation_x, pos[recomb_node][1]),
                       arrowprops=dict(arrowstyle='->', color='purple', lw=1.5),
                       fontsize=9, color='purple', ha=ha,
                       fontweight='bold')
    
    # Style
    x_margin = 6
    y_margin = 3
    ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin + 4)
    ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)
    ax.axis('off')
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    return fig

def main():
    """Generate and save rumor cascade with recombination visualization."""
    print("Generating rumor cascade with phylogenetic recombination...")
    
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Create cascade with recombination tracking
    G, lineages, lineage_roots = create_rumor_cascade_with_recombination()
    
    # Count statistics
    n_lineages = len(set(lineages.values()))
    recomb_nodes = sum(1 for n in G.nodes() if G.nodes[n].get('is_recombination', False))
    
    print(f"Created cascade with {G.number_of_nodes()} nodes")
    print(f"Maximum depth: {max(G.nodes[n]['level'] for n in G.nodes())} levels")
    print(f"Number of distinct lineages: {n_lineages}")
    print(f"Recombination points: {recomb_nodes}")
    
    # Visualize
    print("Creating colored visualization...")
    fig = visualize_rumor_cascade_with_recombination(G, lineages, lineage_roots)
    
    # Save
    output_path = 'visualizations/rumor_cascade_recombination.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Visualization saved to {output_path}")
    
    # Save PDF version
    output_pdf = 'visualizations/rumor_cascade_recombination.pdf'
    fig.savefig(output_pdf, bbox_inches='tight', facecolor='white')
    print(f"PDF version saved to {output_pdf}")
    
    plt.show()

if __name__ == "__main__":
    main()