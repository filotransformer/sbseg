"""
Script para criar visualização comparativa entre cascatas de rumor e não-rumor
Demonstra as diferenças estruturais características entre os dois tipos
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
from datetime import datetime
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from collections import defaultdict
import random

# Configuração para melhor qualidade visual
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 9
plt.rcParams['font.family'] = 'sans-serif'

class ComparativeCascadeVisualizer:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        
    def load_cascade(self, event, rumour_type, cascade_id):
        """Carrega uma cascata específica do dataset"""
        base_path = self.dataset_path / event / rumour_type / cascade_id
        
        # Carrega tweet fonte
        source_path = base_path / 'source-tweet' / f'{cascade_id}.json'
        with open(source_path, 'r', encoding='utf-8') as f:
            source = json.load(f)
            
        # Carrega reactions
        reactions = []
        reactions_path = base_path / 'reactions'
        if reactions_path.exists():
            for reaction_file in reactions_path.glob('*.json'):
                with open(reaction_file, 'r', encoding='utf-8') as f:
                    reactions.append(json.load(f))
                    
        return source, reactions
    
    def extract_tweet_info(self, tweet):
        """Extrai informações relevantes de um tweet"""
        return {
            'id': tweet['id_str'],
            'text': tweet['text'],
            'created_at': datetime.strptime(tweet['created_at'], '%a %b %d %H:%M:%S %z %Y'),
            'user': tweet['user']['screen_name'],
            'followers': tweet['user']['followers_count'],
            'retweets': tweet.get('retweet_count', 0),
            'favorites': tweet.get('favorite_count', 0),
            'in_reply_to': tweet.get('in_reply_to_status_id_str'),
            'is_verified': tweet['user'].get('verified', False),
        }
    
    def build_graph(self, source, reactions, max_nodes=80):
        """Constrói o grafo da cascata"""
        G = nx.DiGraph()
        
        # Extrai informações do tweet fonte
        source_info = self.extract_tweet_info(source)
        source_id = source_info['id']
        
        # Adiciona nó fonte
        G.add_node(source_id, 
                  **source_info,
                  level=0, 
                  node_type='source')
        
        # Para rumores, prioriza threads longas para mostrar profundidade
        # Para não-rumores, prioriza engajamento
        if max_nodes and len(reactions) > max_nodes - 1:
            # Ordena por critérios diferentes
            reactions_sorted = sorted(reactions, 
                                    key=lambda r: (r.get('retweet_count', 0) + 
                                                 r.get('favorite_count', 0)), 
                                    reverse=True)
            reactions_to_use = reactions_sorted[:max_nodes-1]
        else:
            reactions_to_use = reactions
        
        # Mapeia tweets por ID
        tweets_map = {source_id: source_info}
        
        # Processa reactions
        for reaction in reactions_to_use:
            reaction_info = self.extract_tweet_info(reaction)
            reaction_id = reaction_info['id']
            tweets_map[reaction_id] = reaction_info
            
            # Determina parent
            parent_id = reaction_info.get('in_reply_to')
            if parent_id not in tweets_map:
                parent_id = source_id
                
            # Adiciona nó e edge
            G.add_node(reaction_id, 
                      **reaction_info,
                      node_type='reaction')
            G.add_edge(parent_id, reaction_id)
        
        # Calcula níveis
        self._calculate_levels(G, source_id)
        
        return G, len(reactions)
    
    def _calculate_levels(self, G, source_id):
        """Calcula níveis hierárquicos dos nós"""
        queue = [(source_id, 0)]
        visited = {source_id}
        
        while queue:
            node, level = queue.pop(0)
            G.nodes[node]['level'] = level
            
            for successor in G.successors(node):
                if successor not in visited:
                    visited.add(successor)
                    queue.append((successor, level + 1))
    
    def create_comparison_visualization(self, 
                                       rumour_data, non_rumour_data,
                                       save_path=None):
        """Cria visualização comparativa lado a lado"""
        
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Título principal
        fig.suptitle('Comparação Estrutural: Rumor vs Não-Rumor', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # Lado esquerdo - RUMOR
        ax_rumour = fig.add_subplot(gs[:2, :2])
        self._draw_cascade(ax_rumour, rumour_data['graph'], 
                          title=f"RUMOR - {rumour_data['event']}",
                          subtitle=f"{rumour_data['total']} reactions totais",
                          color_scheme='rumour')
        
        # Lado direito - NÃO-RUMOR
        ax_non_rumour = fig.add_subplot(gs[:2, 2:])
        self._draw_cascade(ax_non_rumour, non_rumour_data['graph'],
                          title=f"NÃO-RUMOR - {non_rumour_data['event']}",
                          subtitle=f"{non_rumour_data['total']} reactions totais",
                          color_scheme='non_rumour')
        
        # Estatísticas comparativas - Rumor
        ax_stats_rumour = fig.add_subplot(gs[2, :2])
        self._add_statistics_panel(ax_stats_rumour, rumour_data, 'RUMOR')
        
        # Estatísticas comparativas - Não-Rumor
        ax_stats_non_rumour = fig.add_subplot(gs[2, 2:])
        self._add_statistics_panel(ax_stats_non_rumour, non_rumour_data, 'NÃO-RUMOR')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"✅ Visualização comparativa salva em: {save_path}")
        
        return fig
    
    def _draw_cascade(self, ax, G, title, subtitle, color_scheme):
        """Desenha uma cascata individual"""
        
        # Escolhe layout baseado no tipo
        if color_scheme == 'rumour':
            pos = self._hierarchical_tree_layout(G)  # Enfatiza profundidade
        else:
            pos = self._radial_tree_layout(G)  # Enfatiza largura
        
        # Define esquema de cores
        if color_scheme == 'rumour':
            source_color = '#DC2626'  # Vermelho intenso
            verified_color = '#F59E0B'  # Laranja
            level_colors = ['#EF4444', '#F87171', '#FCA5A5', '#FECACA', '#FEE2E2']
        else:
            source_color = '#059669'  # Verde
            verified_color = '#0891B2'  # Ciano
            level_colors = ['#10B981', '#34D399', '#6EE7B7', '#A7F3D0', '#D1FAE5']
        
        # Prepara cores e tamanhos dos nós
        node_colors = []
        node_sizes = []
        
        for node in G.nodes():
            node_data = G.nodes[node]
            
            if node_data['node_type'] == 'source':
                node_colors.append(source_color)
                node_sizes.append(800)
            elif node_data.get('is_verified', False):
                node_colors.append(verified_color)
                node_sizes.append(400)
            else:
                level = min(node_data['level'], len(level_colors) - 1)
                node_colors.append(level_colors[level])
                engagement = node_data.get('retweets', 0) + node_data.get('favorites', 0)
                node_sizes.append(150 + min(engagement * 2, 300))
        
        # Cores das edges baseadas em tempo
        edge_colors = []
        edge_widths = []
        
        for edge in G.edges():
            source_time = G.nodes[edge[0]]['created_at']
            target_time = G.nodes[edge[1]]['created_at']
            time_diff = (target_time - source_time).total_seconds() / 3600
            
            if time_diff < 1:
                edge_colors.append('#1F2937')
                edge_widths.append(2.0)
            elif time_diff < 6:
                edge_colors.append('#6B7280')
                edge_widths.append(1.5)
            else:
                edge_colors.append('#D1D5DB')
                edge_widths.append(1.0)
        
        # Desenha edges
        nx.draw_networkx_edges(G, pos, ax=ax,
                              edge_color=edge_colors,
                              width=edge_widths,
                              alpha=0.5,
                              arrows=True,
                              arrowsize=10,
                              arrowstyle='-|>')
        
        # Desenha nós
        nx.draw_networkx_nodes(G, pos, ax=ax,
                              node_color=node_colors,
                              node_size=node_sizes,
                              alpha=0.9,
                              edgecolors='white',
                              linewidths=1.5)
        
        # Labels para nós importantes
        labels = {}
        for node in G.nodes():
            node_data = G.nodes[node]
            if node_data['node_type'] == 'source':
                labels[node] = 'FONTE'
            elif node_data.get('is_verified', False):
                labels[node] = '✓'
        
        nx.draw_networkx_labels(G, pos, labels, ax=ax,
                               font_size=8, font_weight='bold')
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.text(0.5, 0.95, subtitle, transform=ax.transAxes,
               ha='center', fontsize=10, style='italic')
        ax.axis('off')
    
    def _hierarchical_tree_layout(self, G):
        """Layout hierárquico vertical (melhor para mostrar profundidade)"""
        pos = {}
        levels = defaultdict(list)
        
        # Agrupa por nível
        for node in G.nodes():
            level = G.nodes[node]['level']
            levels[level].append(node)
        
        # Posiciona verticalmente
        max_width = max(len(nodes) for nodes in levels.values()) if levels else 1
        
        for level, nodes in levels.items():
            y = -level * 0.8  # Espaçamento vertical
            x_spacing = 2.0 / (len(nodes) + 1) if nodes else 1
            
            for i, node in enumerate(nodes):
                x = -1.0 + (i + 1) * x_spacing
                pos[node] = (x, y)
        
        return pos
    
    def _radial_tree_layout(self, G):
        """Layout radial (melhor para mostrar largura)"""
        pos = {}
        
        # Encontra fonte
        source = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'source'][0]
        pos[source] = (0, 0)
        
        # Agrupa por nível
        levels = defaultdict(list)
        for node in G.nodes():
            if node != source:
                level = G.nodes[node]['level']
                levels[level].append(node)
        
        # Posiciona em círculos
        for level, nodes in levels.items():
            if not nodes:
                continue
            
            radius = level * 0.3
            angle_step = 2 * np.pi / len(nodes)
            
            for i, node in enumerate(nodes):
                angle = i * angle_step
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                pos[node] = (x, y)
        
        return pos
    
    def _add_statistics_panel(self, ax, data, label):
        """Adiciona painel de estatísticas"""
        ax.axis('off')
        
        G = data['graph']
        
        # Calcula estatísticas
        levels = [G.nodes[n]['level'] for n in G.nodes()]
        max_depth = max(levels) if levels else 0
        
        level_counts = defaultdict(int)
        for level in levels:
            level_counts[level] += 1
        
        # Conta threads longas
        long_threads = sum(1 for n in G.nodes() if G.nodes[n]['level'] >= 3)
        
        # Taxa de ramificação
        out_degrees = [G.out_degree(n) for n in G.nodes()]
        avg_branching = np.mean(out_degrees) if out_degrees else 0
        
        # Tempo de vida
        times = [G.nodes[n]['created_at'] for n in G.nodes()]
        if times:
            lifetime = (max(times) - min(times)).total_seconds() / 3600
        else:
            lifetime = 0
        
        # Cria texto de estatísticas
        stats_text = f"""📊 Estatísticas - {label}

🌳 ESTRUTURA:
• Profundidade máxima: {max_depth} níveis
• Taxa de ramificação: {avg_branching:.2f}
• Threads longas (3+ níveis): {long_threads}
• Nós visualizados: {G.number_of_nodes()}

📈 DISTRIBUIÇÃO POR NÍVEL:"""
        
        # Adiciona distribuição
        for level in sorted(level_counts.keys())[:5]:
            count = level_counts[level]
            bar = '█' * min(int(count/5), 20)
            stats_text += f"\n  Nível {level}: {count:3d} {bar}"
        
        stats_text += f"\n\n⏱️ TEMPORAL:\n• Tempo de vida: {lifetime:.1f}h"
        
        # Adiciona características distintivas
        if label == 'RUMOR':
            stats_text += "\n\n🔴 CARACTERÍSTICAS:\n• Estrutura mais profunda\n• Múltiplas threads de debate\n• Questionamentos em cascata"
        else:
            stats_text += "\n\n🟢 CARACTERÍSTICAS:\n• Estrutura mais larga\n• Compartilhamento direto\n• Menos debates aninhados"
        
        # Renderiza texto
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.5', 
                        facecolor='#F3F4F6' if label == 'RUMOR' else '#F0FDF4',
                        edgecolor='#DC2626' if label == 'RUMOR' else '#059669',
                        linewidth=2))


def create_rumour_visualization():
    """Cria visualização individual de rumor"""
    
    dataset_path = Path('datasets/pheme-rnr-dataset')
    output_dir = Path('visualizations')
    output_dir.mkdir(exist_ok=True)
    
    visualizer = ComparativeCascadeVisualizer(dataset_path)
    
    # Carrega cascata de rumor com estrutura interessante
    event = 'charliehebdo'
    cascade_id = '552993818816299008'  # 177 reactions, profundidade 6
    
    print(f"🔄 Processando cascata de RUMOR: {event}/rumours/{cascade_id}")
    
    source, reactions = visualizer.load_cascade(event, 'rumours', cascade_id)
    G, total = visualizer.build_graph(source, reactions, max_nodes=70)
    
    # Cria visualização individual do rumor
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    
    visualizer._draw_cascade(ax, G, 
                           title=f"Cascata de RUMOR - Charlie Hebdo",
                           subtitle=f"Total: {total} reactions | Profundidade: 6 níveis",
                           color_scheme='rumour')
    
    # Adiciona anotações sobre características
    ax.text(0.02, 0.02, 
           "⚠️ Características de Rumor:\n" +
           "• Estrutura profunda (6 níveis)\n" +
           "• Múltiplas threads de discussão\n" +
           "• Debates e questionamentos aninhados\n" +
           "• Propagação fragmentada",
           transform=ax.transAxes,
           fontsize=10,
           bbox=dict(boxstyle='round', facecolor='#FEE2E2', 
                    edgecolor='#DC2626', linewidth=2))
    
    output_path = output_dir / 'cascade_rumour_presentation.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Visualização de RUMOR salva em: {output_path}")
    
    return G, total


def create_comparison():
    """Cria visualização comparativa completa"""
    
    dataset_path = Path('datasets/pheme-rnr-dataset')
    output_dir = Path('visualizations')
    output_dir.mkdir(exist_ok=True)
    
    visualizer = ComparativeCascadeVisualizer(dataset_path)
    
    # Carrega cascata de RUMOR
    rumour_event = 'charliehebdo'
    rumour_id = '552993818816299008'
    
    print(f"📊 Carregando RUMOR: {rumour_event}/rumours/{rumour_id}")
    rumour_source, rumour_reactions = visualizer.load_cascade(
        rumour_event, 'rumours', rumour_id)
    rumour_G, rumour_total = visualizer.build_graph(
        rumour_source, rumour_reactions, max_nodes=60)
    
    rumour_data = {
        'graph': rumour_G,
        'total': rumour_total,
        'event': 'Charlie Hebdo (Rumor)'
    }
    
    # Carrega cascata de NÃO-RUMOR
    non_rumour_event = 'charliehebdo'
    non_rumour_id = '552797154692300800'  # 300 reactions, mais largo
    
    print(f"📊 Carregando NÃO-RUMOR: {non_rumour_event}/non-rumours/{non_rumour_id}")
    non_rumour_source, non_rumour_reactions = visualizer.load_cascade(
        non_rumour_event, 'non-rumours', non_rumour_id)
    non_rumour_G, non_rumour_total = visualizer.build_graph(
        non_rumour_source, non_rumour_reactions, max_nodes=60)
    
    non_rumour_data = {
        'graph': non_rumour_G,
        'total': non_rumour_total,
        'event': 'Charlie Hebdo (Não-Rumor)'
    }
    
    # Cria visualização comparativa
    output_path = output_dir / 'cascade_comparison_presentation.png'
    visualizer.create_comparison_visualization(
        rumour_data, non_rumour_data, save_path=output_path)
    
    print(f"\n🎯 Visualizações criadas com sucesso!")


if __name__ == "__main__":
    # Cria visualização individual do rumor
    create_rumour_visualization()
    
    # Cria visualização comparativa
    create_comparison()