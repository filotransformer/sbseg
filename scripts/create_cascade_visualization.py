"""
Script otimizado para criar visualização de cascata de informações para apresentação
Seleciona automaticamente a cascata mais representativa e cria visualização de alta qualidade
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime
import numpy as np
from matplotlib.patches import FancyBboxPatch, Circle
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from collections import defaultdict
import random

# Configuração para melhor qualidade visual
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'

class EnhancedCascadeVisualizer:
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
            'user_desc': tweet['user'].get('description', ''),
        }
    
    def build_graph(self, source, reactions, max_nodes=50):
        """Constrói o grafo da cascata com limite de nós para visualização"""
        G = nx.DiGraph()
        
        # Extrai informações do tweet fonte
        source_info = self.extract_tweet_info(source)
        source_id = source_info['id']
        
        # Adiciona nó fonte
        G.add_node(source_id, 
                  **source_info,
                  level=0, 
                  node_type='source')
        
        # Ordena reactions por tempo e engajamento para pegar as mais relevantes
        reactions_sorted = sorted(reactions, 
                                key=lambda r: (r.get('retweet_count', 0) + 
                                             r.get('favorite_count', 0)), 
                                reverse=True)
        
        # Limita número de reactions se necessário
        if len(reactions_sorted) > max_nodes - 1:
            # Pega os mais engajados e alguns aleatórios
            top_reactions = reactions_sorted[:int(max_nodes * 0.7)]
            random_sample = random.sample(reactions_sorted[int(max_nodes * 0.7):], 
                                        int(max_nodes * 0.3))
            reactions_to_use = top_reactions + random_sample
        else:
            reactions_to_use = reactions_sorted
        
        # Mapeia tweets por ID
        tweets_map = {source_id: source_info}
        
        # Processa reactions selecionadas
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
        
        return G, len(reactions)  # Retorna também o total de reactions
    
    def _calculate_levels(self, G, source_id):
        """Calcula níveis hierárquicos dos nós"""
        # BFS para calcular níveis
        queue = [(source_id, 0)]
        visited = {source_id}
        
        while queue:
            node, level = queue.pop(0)
            G.nodes[node]['level'] = level
            
            for successor in G.successors(node):
                if successor not in visited:
                    visited.add(successor)
                    queue.append((successor, level + 1))
    
    def create_presentation_visualization(self, G, total_reactions, title="", save_path=None):
        """Cria visualização otimizada para apresentação"""
        fig = plt.figure(figsize=(18, 12))
        
        # Layout principal para a cascata
        ax_main = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=3)
        
        # Calcula layout radial hierárquico
        pos = self._radial_tree_layout(G)
        
        # Prepara cores e tamanhos
        node_colors = []
        node_sizes = []
        node_alphas = []
        
        for node in G.nodes():
            node_data = G.nodes[node]
            
            # Cores por tipo
            if node_data['node_type'] == 'source':
                node_colors.append('#E63946')  # Vermelho vibrante
                node_sizes.append(1500)
                node_alphas.append(1.0)
            elif node_data.get('is_verified', False):
                node_colors.append('#2A9D8F')  # Verde-azulado
                engagement = node_data.get('retweets', 0) + node_data.get('favorites', 0)
                node_sizes.append(500 + min(engagement * 5, 1000))
                node_alphas.append(0.9)
            else:
                level = node_data['level']
                if level == 1:
                    node_colors.append('#457B9D')  # Azul
                elif level == 2:
                    node_colors.append('#A8DADC')  # Azul claro
                else:
                    node_colors.append('#F1FAEE')  # Quase branco
                
                engagement = node_data.get('retweets', 0) + node_data.get('favorites', 0)
                node_sizes.append(200 + min(engagement * 3, 600))
                node_alphas.append(0.7)
        
        # Desenha edges com gradiente de cor baseado em tempo
        edge_colors = []
        edge_widths = []
        
        for edge in G.edges():
            source_time = G.nodes[edge[0]]['created_at']
            target_time = G.nodes[edge[1]]['created_at']
            time_diff = (target_time - source_time).total_seconds() / 3600
            
            if time_diff < 1:
                edge_colors.append('#264653')  # Azul escuro - propagação rápida
                edge_widths.append(2.5)
            elif time_diff < 6:
                edge_colors.append('#2A9D8F')  # Verde - propagação média
                edge_widths.append(1.5)
            else:
                edge_colors.append('#E76F51')  # Laranja - propagação lenta
                edge_widths.append(1.0)
        
        # Desenha a rede
        nx.draw_networkx_edges(G, pos, ax=ax_main,
                              edge_color=edge_colors, 
                              width=edge_widths,
                              alpha=0.4,
                              arrows=True,
                              arrowsize=12,
                              arrowstyle='-|>',
                              connectionstyle='arc3,rad=0.1')
        
        # Desenha nós com efeito de brilho
        for i, node in enumerate(G.nodes()):
            x, y = pos[node]
            
            # Adiciona halo para nó fonte
            if G.nodes[node]['node_type'] == 'source':
                circle = Circle((x, y), 0.08, color=node_colors[i], alpha=0.2)
                ax_main.add_patch(circle)
                circle2 = Circle((x, y), 0.06, color=node_colors[i], alpha=0.3)
                ax_main.add_patch(circle2)
        
        nx.draw_networkx_nodes(G, pos, ax=ax_main,
                              node_color=node_colors,
                              node_size=node_sizes,
                              alpha=node_alphas,
                              edgecolors='white',
                              linewidths=2)
        
        # Labels seletivos
        labels = {}
        for node in G.nodes():
            node_data = G.nodes[node]
            if node_data['node_type'] == 'source':
                labels[node] = f"FONTE\n@{node_data['user'][:10]}"
            elif node_data.get('is_verified', False) or node_data.get('retweets', 0) > 10:
                labels[node] = f"@{node_data['user'][:8]}"
        
        nx.draw_networkx_labels(G, pos, labels, ax=ax_main,
                               font_size=8, font_weight='bold',
                               font_color='#1D3557')
        
        # Título principal
        ax_main.set_title(f'Cascata de Propagação de Informação\n{title}',
                         fontsize=16, fontweight='bold', pad=20)
        ax_main.axis('off')
        
        # Painel de estatísticas
        ax_stats = plt.subplot2grid((3, 3), (0, 2))
        ax_stats.axis('off')
        
        # Calcula estatísticas
        stats = self._calculate_stats(G, total_reactions)
        
        # Adiciona box de estatísticas
        stats_text = f"""📊 MÉTRICAS DA CASCATA

🔢 Total de nós: {stats['total_nodes']}
   (Visualizados: {G.number_of_nodes()})

📐 Profundidade: {stats['max_depth']} níveis

🌿 Taxa de ramificação: {stats['branching_factor']:.2f}

⏱️ Tempo de vida: {stats['lifetime_hours']:.1f}h

✅ Usuários verificados: {stats['verified_users']}

🔄 Engajamento total:
   • Retweets: {stats['total_retweets']}
   • Favoritos: {stats['total_favorites']}

📈 Velocidade de propagação:
   • Primeiras 6h: {stats['fast_propagation']}%
   • 6-24h: {stats['medium_propagation']}%
   • >24h: {stats['slow_propagation']}%"""
        
        ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes,
                     fontsize=11, verticalalignment='top',
                     bbox=dict(boxstyle='round,pad=0.5', 
                              facecolor='#F1FAEE', 
                              edgecolor='#457B9D',
                              linewidth=2))
        
        # Legenda
        ax_legend = plt.subplot2grid((3, 3), (1, 2))
        ax_legend.axis('off')
        
        legend_elements = [
            mpatches.Circle((0.5, 0.5), 0.15, facecolor='#E63946', 
                          edgecolor='white', linewidth=2, label='Tweet Fonte'),
            mpatches.Circle((0.5, 0.5), 0.12, facecolor='#2A9D8F', 
                          edgecolor='white', linewidth=2, label='Usuário Verificado'),
            mpatches.Circle((0.5, 0.5), 0.10, facecolor='#457B9D', 
                          edgecolor='white', linewidth=2, label='Resposta Direta'),
            mpatches.Circle((0.5, 0.5), 0.08, facecolor='#A8DADC', 
                          edgecolor='white', linewidth=2, label='Resposta Indireta'),
            mlines.Line2D([], [], color='#264653', linewidth=3, 
                         label='Propagação < 1h'),
            mlines.Line2D([], [], color='#2A9D8F', linewidth=2, 
                         label='Propagação 1-6h'),
            mlines.Line2D([], [], color='#E76F51', linewidth=1.5, 
                         label='Propagação > 6h'),
        ]
        
        ax_legend.legend(handles=legend_elements, loc='center', 
                        fontsize=10, frameon=True,
                        fancybox=True, shadow=True,
                        title='LEGENDA', title_fontsize=12)
        
        # Timeline
        ax_timeline = plt.subplot2grid((3, 3), (2, 2))
        self._add_timeline(ax_timeline, G)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"✅ Visualização salva em: {save_path}")
        
        return fig
    
    def _radial_tree_layout(self, G):
        """Cria layout radial hierárquico"""
        pos = {}
        
        # Encontra o nó fonte
        source = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'source'][0]
        
        # Posiciona fonte no centro
        pos[source] = (0, 0)
        
        # Agrupa nós por nível
        levels = defaultdict(list)
        for node in G.nodes():
            if node != source:
                level = G.nodes[node]['level']
                levels[level].append(node)
        
        # Posiciona cada nível em círculos concêntricos
        for level, nodes in levels.items():
            if not nodes:
                continue
                
            radius = level * 0.4  # Distância do centro
            angle_step = 2 * np.pi / len(nodes)
            
            for i, node in enumerate(nodes):
                angle = i * angle_step
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                pos[node] = (x, y)
        
        return pos
    
    def _calculate_stats(self, G, total_reactions):
        """Calcula estatísticas detalhadas"""
        stats = {
            'total_nodes': total_reactions + 1,  # +1 para o fonte
            'max_depth': 0,
            'branching_factor': 0,
            'lifetime_hours': 0,
            'verified_users': 0,
            'total_retweets': 0,
            'total_favorites': 0,
            'fast_propagation': 0,
            'medium_propagation': 0,
            'slow_propagation': 0
        }
        
        # Profundidade máxima
        levels = [G.nodes[n]['level'] for n in G.nodes()]
        stats['max_depth'] = max(levels) if levels else 0
        
        # Taxa de ramificação
        out_degrees = [G.out_degree(n) for n in G.nodes()]
        stats['branching_factor'] = np.mean(out_degrees) if out_degrees else 0
        
        # Tempo de vida
        times = [G.nodes[n]['created_at'] for n in G.nodes()]
        if times:
            stats['lifetime_hours'] = (max(times) - min(times)).total_seconds() / 3600
        
        # Contagens
        base_time = min(times) if times else None
        fast, medium, slow = 0, 0, 0
        
        for node in G.nodes():
            node_data = G.nodes[node]
            
            if node_data.get('is_verified', False):
                stats['verified_users'] += 1
            
            stats['total_retweets'] += node_data.get('retweets', 0)
            stats['total_favorites'] += node_data.get('favorites', 0)
            
            if base_time:
                time_diff = (node_data['created_at'] - base_time).total_seconds() / 3600
                if time_diff < 6:
                    fast += 1
                elif time_diff < 24:
                    medium += 1
                else:
                    slow += 1
        
        total = fast + medium + slow
        if total > 0:
            stats['fast_propagation'] = int(100 * fast / total)
            stats['medium_propagation'] = int(100 * medium / total)
            stats['slow_propagation'] = int(100 * slow / total)
        
        return stats
    
    def _add_timeline(self, ax, G):
        """Adiciona mini timeline de propagação"""
        times = []
        for node in G.nodes():
            times.append(G.nodes[node]['created_at'])
        
        if not times:
            return
            
        times_sorted = sorted(times)
        base_time = times_sorted[0]
        hours = [(t - base_time).total_seconds() / 3600 for t in times_sorted]
        
        ax.plot(hours, range(1, len(hours) + 1), 
               'b-', linewidth=2, marker='o', markersize=3)
        ax.fill_between(hours, 0, range(1, len(hours) + 1), 
                        alpha=0.3, color='#457B9D')
        
        ax.set_xlabel('Horas desde início', fontsize=9)
        ax.set_ylabel('Nº acumulado', fontsize=9)
        ax.set_title('Timeline de Propagação', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)


def main():
    """Função principal para gerar visualização para apresentação"""
    
    # Configuração
    dataset_path = Path('datasets/pheme-rnr-dataset')
    output_dir = Path('visualizations')
    output_dir.mkdir(exist_ok=True)
    
    # Inicializa visualizador
    visualizer = EnhancedCascadeVisualizer(dataset_path)
    
    # Usa uma das maiores cascatas para melhor visualização
    # charliehebdo/non-rumours/552806610490646528 tem 345 reactions
    event = 'charliehebdo'
    rumour_type = 'non-rumours'
    cascade_id = '552806610490646528'
    
    print(f"🔄 Processando cascata: {event}/{rumour_type}/{cascade_id}")
    
    try:
        # Carrega dados
        source, reactions = visualizer.load_cascade(event, rumour_type, cascade_id)
        print(f"✅ Cascata carregada: 1 fonte + {len(reactions)} reactions")
        
        # Constrói grafo (limitado para visualização)
        G, total_reactions = visualizer.build_graph(source, reactions, max_nodes=60)
        print(f"📊 Grafo construído: {G.number_of_nodes()} nós visualizados")
        
        # Extrai contexto do tweet fonte
        source_text = source['text'][:100] + "..." if len(source['text']) > 100 else source['text']
        title = f"Charlie Hebdo - Cascata de Informação Não-Rumor\n\"{source_text}\""
        
        # Cria visualização
        output_path = output_dir / 'cascade_presentation.png'
        fig = visualizer.create_presentation_visualization(
            G, total_reactions, 
            title=title,
            save_path=output_path
        )
        
        print(f"\n🎯 Visualização criada com sucesso!")
        print(f"📁 Arquivo salvo em: {output_path}")
        
        # Salva também em formato de alta resolução
        hq_path = output_dir / 'cascade_presentation_hq.png'
        fig.savefig(hq_path, dpi=600, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"📁 Versão alta resolução salva em: {hq_path}")
        
    except Exception as e:
        print(f"❌ Erro ao processar cascata: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()