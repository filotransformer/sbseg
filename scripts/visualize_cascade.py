"""
Script para visualizar cascatas de informações do dataset PHEME
Cria visualizações representativas das árvores filogenéticas de propagação
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime
import numpy as np
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from collections import defaultdict

class CascadeVisualizer:
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
            'text': tweet['text'][:100] + '...' if len(tweet['text']) > 100 else tweet['text'],
            'created_at': datetime.strptime(tweet['created_at'], '%a %b %d %H:%M:%S %z %Y'),
            'user': tweet['user']['screen_name'],
            'followers': tweet['user']['followers_count'],
            'retweets': tweet.get('retweet_count', 0),
            'favorites': tweet.get('favorite_count', 0),
            'in_reply_to': tweet.get('in_reply_to_status_id_str'),
            'is_verified': tweet['user'].get('verified', False)
        }
    
    def build_graph(self, source, reactions):
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
        
        # Mapeia tweets por ID para encontrar parents
        tweets_map = {source_id: source_info}
        
        # Processa reactions
        for reaction in reactions:
            reaction_info = self.extract_tweet_info(reaction)
            reaction_id = reaction_info['id']
            tweets_map[reaction_id] = reaction_info
            
            # Determina parent
            parent_id = reaction_info.get('in_reply_to')
            if parent_id not in tweets_map:
                parent_id = source_id  # Se parent não encontrado, conecta ao source
                
            # Adiciona nó e edge
            G.add_node(reaction_id, 
                      **reaction_info,
                      node_type='reaction')
            G.add_edge(parent_id, reaction_id)
            
        # Calcula níveis (profundidade)
        for node in nx.topological_sort(G):
            if node == source_id:
                G.nodes[node]['level'] = 0
            else:
                parent = list(G.predecessors(node))[0]
                G.nodes[node]['level'] = G.nodes[parent]['level'] + 1
                
        return G
    
    def visualize_cascade_tree(self, G, title="Information Cascade", save_path=None):
        """Cria visualização em árvore da cascata"""
        plt.figure(figsize=(16, 12))
        
        # Calcula layout hierárquico
        pos = self.hierarchical_layout(G)
        
        # Define cores e tamanhos baseados em características
        node_colors = []
        node_sizes = []
        edge_colors = []
        
        for node in G.nodes():
            node_data = G.nodes[node]
            
            # Cor baseada no tipo de nó
            if node_data['node_type'] == 'source':
                node_colors.append('#FF6B6B')  # Vermelho para fonte
            elif node_data.get('is_verified', False):
                node_colors.append('#4ECDC4')  # Verde-azul para verificado
            else:
                node_colors.append('#95E1D3')  # Verde claro para normal
            
            # Tamanho baseado em engajamento
            engagement = node_data.get('retweets', 0) + node_data.get('favorites', 0)
            node_sizes.append(300 + min(engagement * 10, 2000))
        
        # Cor das edges baseada em tempo
        for edge in G.edges():
            source_time = G.nodes[edge[0]]['created_at']
            target_time = G.nodes[edge[1]]['created_at']
            time_diff = (target_time - source_time).total_seconds() / 3600  # em horas
            
            if time_diff < 1:
                edge_colors.append('#2E86AB')  # Azul escuro para propagação rápida
            elif time_diff < 6:
                edge_colors.append('#7FB069')  # Verde para propagação média
            else:
                edge_colors.append('#F4A259')  # Laranja para propagação lenta
        
        # Desenha o grafo
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, 
                              alpha=0.6, width=2, arrows=True,
                              arrowsize=15, arrowstyle='->')
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.9,
                              edgecolors='black', linewidths=1.5)
        
        # Adiciona labels apenas para nós importantes
        labels = {}
        for node in G.nodes():
            node_data = G.nodes[node]
            if node_data['node_type'] == 'source' or \
               node_data.get('retweets', 0) > 5 or \
               node_data.get('is_verified', False):
                labels[node] = f"@{node_data['user'][:8]}"
        
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
        
        # Adiciona estatísticas
        stats_text = self.get_cascade_stats(G)
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Adiciona legenda
        legend_elements = [
            mpatches.Patch(color='#FF6B6B', label='Tweet Fonte'),
            mpatches.Patch(color='#4ECDC4', label='Usuário Verificado'),
            mpatches.Patch(color='#95E1D3', label='Usuário Normal'),
            mlines.Line2D([], [], color='#2E86AB', linewidth=2, label='Propagação < 1h'),
            mlines.Line2D([], [], color='#7FB069', linewidth=2, label='Propagação 1-6h'),
            mlines.Line2D([], [], color='#F4A259', linewidth=2, label='Propagação > 6h'),
        ]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Visualização salva em: {save_path}")
        
        plt.show()
        return pos
    
    def hierarchical_layout(self, G):
        """Cria layout hierárquico para a árvore"""
        pos = {}
        levels = defaultdict(list)
        
        # Agrupa nós por nível
        for node in G.nodes():
            level = G.nodes[node]['level']
            levels[level].append(node)
        
        # Calcula posições
        max_width = max(len(nodes) for nodes in levels.values())
        y_gap = 1.0
        
        for level, nodes in levels.items():
            x_gap = 2.0 / (len(nodes) + 1)
            for i, node in enumerate(nodes):
                x = -1.0 + (i + 1) * x_gap
                y = -level * y_gap
                pos[node] = (x, y)
        
        return pos
    
    def get_cascade_stats(self, G):
        """Calcula estatísticas da cascata"""
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        
        levels = [G.nodes[node]['level'] for node in G.nodes()]
        max_depth = max(levels)
        
        # Conta nós por nível
        level_counts = defaultdict(int)
        for level in levels:
            level_counts[level] += 1
        
        max_breadth = max(level_counts.values())
        
        # Calcula tempo de vida da cascata
        times = [G.nodes[node]['created_at'] for node in G.nodes()]
        lifetime_hours = (max(times) - min(times)).total_seconds() / 3600
        
        # Conta usuários verificados
        verified_count = sum(1 for node in G.nodes() 
                           if G.nodes[node].get('is_verified', False))
        
        stats = f"""Estatísticas da Cascata:
• Nós totais: {num_nodes}
• Profundidade máxima: {max_depth}
• Largura máxima: {max_breadth}
• Tempo de vida: {lifetime_hours:.1f}h
• Usuários verificados: {verified_count}
• Taxa de ramificação: {num_edges/num_nodes:.2f}"""
        
        return stats
    
    def visualize_temporal_evolution(self, G, save_path=None):
        """Visualiza evolução temporal da cascata"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Extrai dados temporais
        times = []
        for node in G.nodes():
            times.append(G.nodes[node]['created_at'])
        
        times_sorted = sorted(times)
        base_time = times_sorted[0]
        hours_from_start = [(t - base_time).total_seconds() / 3600 for t in times_sorted]
        
        # 1. Crescimento cumulativo
        ax = axes[0, 0]
        ax.plot(hours_from_start, range(1, len(hours_from_start) + 1), 
               'b-', linewidth=2)
        ax.fill_between(hours_from_start, 0, range(1, len(hours_from_start) + 1), 
                        alpha=0.3)
        ax.set_xlabel('Horas desde o início')
        ax.set_ylabel('Número de tweets')
        ax.set_title('Crescimento Cumulativo da Cascata')
        ax.grid(True, alpha=0.3)
        
        # 2. Taxa de propagação
        ax = axes[0, 1]
        bins = np.linspace(0, max(hours_from_start), 20)
        ax.hist(hours_from_start, bins=bins, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Horas desde o início')
        ax.set_ylabel('Número de tweets')
        ax.set_title('Taxa de Propagação por Hora')
        ax.grid(True, alpha=0.3)
        
        # 3. Profundidade ao longo do tempo
        ax = axes[1, 0]
        time_depth = []
        for node in G.nodes():
            node_time = (G.nodes[node]['created_at'] - base_time).total_seconds() / 3600
            node_depth = G.nodes[node]['level']
            time_depth.append((node_time, node_depth))
        
        time_depth.sort()
        times_plot, depths_plot = zip(*time_depth)
        ax.scatter(times_plot, depths_plot, alpha=0.6, s=30)
        ax.set_xlabel('Horas desde o início')
        ax.set_ylabel('Profundidade na árvore')
        ax.set_title('Evolução da Profundidade')
        ax.grid(True, alpha=0.3)
        
        # 4. Engajamento ao longo do tempo
        ax = axes[1, 1]
        time_engagement = []
        for node in G.nodes():
            node_time = (G.nodes[node]['created_at'] - base_time).total_seconds() / 3600
            engagement = G.nodes[node].get('retweets', 0) + G.nodes[node].get('favorites', 0)
            time_engagement.append((node_time, engagement))
        
        time_engagement.sort()
        if time_engagement:
            times_eng, eng = zip(*time_engagement)
            ax.scatter(times_eng, eng, alpha=0.6, s=30, c=times_eng, cmap='viridis')
            ax.set_xlabel('Horas desde o início')
            ax.set_ylabel('Engajamento (RT + Fav)')
            ax.set_title('Engajamento ao Longo do Tempo')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Análise Temporal da Cascata de Informação', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Análise temporal salva em: {save_path}")
        
        plt.show()


def main():
    # Configuração
    dataset_path = Path('datasets/pheme-rnr-dataset')
    output_dir = Path('visualizations')
    output_dir.mkdir(exist_ok=True)
    
    # Inicializa visualizador
    visualizer = CascadeVisualizer(dataset_path)
    
    # Cascatas interessantes para visualizar (escolhidas por tamanho e estrutura)
    cascades_to_visualize = [
        # Formato: (event, rumour_type, cascade_id, description)
        ('charliehebdo', 'rumours', '552783667052167168', 'Rumor sobre Charlie Hebdo'),
        ('sydneysiege', 'rumours', '544825761033732096', 'Rumor sobre Sydney Siege'),
        ('germanwings-crash', 'non-rumours', '580703530510123009', 'Não-rumor sobre Germanwings'),
    ]
    
    # Tenta visualizar a primeira cascata disponível
    for event, rumour_type, cascade_id, description in cascades_to_visualize:
        try:
            print(f"\nProcessando: {description}")
            print(f"Event: {event}, Type: {rumour_type}, ID: {cascade_id}")
            
            # Carrega cascata
            source, reactions = visualizer.load_cascade(event, rumour_type, cascade_id)
            
            if reactions:  # Só visualiza se houver reactions
                print(f"Cascata carregada: 1 fonte + {len(reactions)} reactions")
                
                # Constrói grafo
                G = visualizer.build_graph(source, reactions)
                
                # Cria visualização principal
                tree_path = output_dir / f'cascade_tree_{event}_{cascade_id}.png'
                visualizer.visualize_cascade_tree(G, 
                                                title=f"Cascata de Informação: {description}",
                                                save_path=tree_path)
                
                # Cria análise temporal
                temporal_path = output_dir / f'cascade_temporal_{event}_{cascade_id}.png'
                visualizer.visualize_temporal_evolution(G, save_path=temporal_path)
                
                print(f"Visualizações criadas com sucesso!")
                break  # Para após criar a primeira visualização bem-sucedida
                
        except Exception as e:
            print(f"Erro ao processar {cascade_id}: {e}")
            continue


if __name__ == "__main__":
    main()