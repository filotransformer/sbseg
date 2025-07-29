#!/usr/bin/env python3
import pandas as pd
import numpy as np
import re
from collections import Counter
from scipy.stats import ttest_ind, mannwhitneyu
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings('ignore')

# Carregar dataset
df = pd.read_csv('/home/acauan/ufam/papers/01_sbseg_filo_trans/datasets/pheme/pheme_all.csv')
df = df.dropna(subset=['text', 'label'])
df = df[df['label'].isin([0, 1])]

print("="*70)
print("ANÁLISE DETALHADA DE FEATURES DISCRIMINATIVAS - DATASET PHEME")
print("="*70)

print(f"Total de amostras: {len(df)}")
print(f"Rumores (1): {len(df[df['label'] == 1])}")
print(f"Não-rumores (0): {len(df[df['label'] == 0])}")

# Separar por classe
rumors = df[df['label'] == 1]
non_rumors = df[df['label'] == 0]

# ===== ANÁLISE 1: PALAVRAS DE URGÊNCIA E ALERTA =====
print("\n" + "="*50)
print("1. PALAVRAS DE URGÊNCIA E ALERTA")
print("="*50)

urgency_words = [
    'breaking', 'urgent', 'alert', 'emergency', 'crisis', 'immediate',
    'now', 'just', 'confirmed', 'reports', 'latest', 'update',
    'developing', 'live', 'happening', 'ongoing'
]

def count_urgency_words(text):
    if not isinstance(text, str):
        return 0
    text_lower = text.lower()
    return sum(1 for word in urgency_words if word in text_lower)

rumors['urgency_count'] = rumors['text'].apply(count_urgency_words)
non_rumors['urgency_count'] = non_rumors['text'].apply(count_urgency_words)

rumor_urgency_mean = rumors['urgency_count'].mean()
non_rumor_urgency_mean = non_rumors['urgency_count'].mean()

print(f"Média de palavras de urgência:")
print(f"  Rumores: {rumor_urgency_mean:.4f}")
print(f"  Não-rumores: {non_rumor_urgency_mean:.4f}")
print(f"  Diferença: {((rumor_urgency_mean - non_rumor_urgency_mean) / non_rumor_urgency_mean * 100):+.2f}%")

# Teste estatístico
stat, p_value = mannwhitneyu(rumors['urgency_count'], non_rumors['urgency_count'])
print(f"  Significância estatística (Mann-Whitney U): p = {p_value:.6f}")

# ===== ANÁLISE 2: PADRÕES DE NÚMEROS E ESTATÍSTICAS =====
print("\n" + "="*50)
print("2. PADRÕES DE NÚMEROS E ESTATÍSTICAS")
print("="*50)

def extract_number_patterns(text):
    if not isinstance(text, str):
        return {
            'number_count': 0,
            'percentage_count': 0,
            'time_pattern_count': 0,
            'casualty_numbers': 0
        }
    
    # Números gerais
    numbers = re.findall(r'\b\d+\b', text)
    
    # Percentagens
    percentages = re.findall(r'\d+%', text)
    
    # Padrões de tempo (HH:MM)
    time_patterns = re.findall(r'\b\d{1,2}:\d{2}\b', text)
    
    # Números que podem indicar vítimas/mortos
    casualty_words = ['dead', 'killed', 'injured', 'wounded', 'victims', 'casualties']
    casualty_numbers = 0
    for word in casualty_words:
        pattern = rf'\b\d+\s+{word}|\b{word}\s+\d+|\b\d+\s+.*{word}'
        casualty_numbers += len(re.findall(pattern, text.lower()))
    
    return {
        'number_count': len(numbers),
        'percentage_count': len(percentages),
        'time_pattern_count': len(time_patterns),
        'casualty_numbers': casualty_numbers
    }

# Aplicar análise de números
rumor_number_patterns = rumors['text'].apply(extract_number_patterns)
non_rumor_number_patterns = non_rumors['text'].apply(extract_number_patterns)

for pattern_type in ['number_count', 'percentage_count', 'time_pattern_count', 'casualty_numbers']:
    rumor_values = [p[pattern_type] for p in rumor_number_patterns]
    non_rumor_values = [p[pattern_type] for p in non_rumor_number_patterns]
    
    rumor_mean = np.mean(rumor_values)
    non_rumor_mean = np.mean(non_rumor_values)
    
    print(f"\n{pattern_type.replace('_', ' ').title()}:")
    print(f"  Rumores: {rumor_mean:.4f}")
    print(f"  Não-rumores: {non_rumor_mean:.4f}")
    
    if non_rumor_mean > 0:
        diff_pct = ((rumor_mean - non_rumor_mean) / non_rumor_mean * 100)
        print(f"  Diferença: {diff_pct:+.2f}%")
    
    # Teste estatístico
    if len(set(rumor_values + non_rumor_values)) > 1:
        stat, p_value = mannwhitneyu(rumor_values, non_rumor_values, alternative='two-sided')
        print(f"  Significância: p = {p_value:.6f}")

# ===== ANÁLISE 3: PADRÕES DE PONTUAÇÃO EMOCIONAL =====
print("\n" + "="*50)
print("3. PADRÕES DE PONTUAÇÃO EMOCIONAL")
print("="*50)

def extract_punctuation_patterns(text):
    if not isinstance(text, str):
        return {
            'multiple_exclamation': 0,
            'multiple_question': 0,
            'caps_words': 0,
            'ellipsis': 0
        }
    
    return {
        'multiple_exclamation': len(re.findall(r'!{2,}', text)),
        'multiple_question': len(re.findall(r'\?{2,}', text)),
        'caps_words': len(re.findall(r'\b[A-Z]{3,}\b', text)),
        'ellipsis': len(re.findall(r'\.{3,}', text))
    }

rumor_punct_patterns = rumors['text'].apply(extract_punctuation_patterns)
non_rumor_punct_patterns = non_rumors['text'].apply(extract_punctuation_patterns)

for pattern_type in ['multiple_exclamation', 'multiple_question', 'caps_words', 'ellipsis']:
    rumor_values = [p[pattern_type] for p in rumor_punct_patterns]
    non_rumor_values = [p[pattern_type] for p in non_rumor_punct_patterns]
    
    rumor_mean = np.mean(rumor_values)
    non_rumor_mean = np.mean(non_rumor_values)
    
    print(f"\n{pattern_type.replace('_', ' ').title()}:")
    print(f"  Rumores: {rumor_mean:.4f}")
    print(f"  Não-rumores: {non_rumor_mean:.4f}")
    
    if non_rumor_mean > 0:
        diff_pct = ((rumor_mean - non_rumor_mean) / non_rumor_mean * 100)
        print(f"  Diferença: {diff_pct:+.2f}%")

# ===== ANÁLISE 4: RATIOS DE ENGAGEMENT =====
print("\n" + "="*50)
print("4. RATIOS DE ENGAGEMENT SOCIAL")
print("="*50)

# Calcular ratios
def safe_divide(a, b):
    return a / b if b > 0 else 0

df['like_retweet_ratio'] = df.apply(lambda x: safe_divide(x['favorite_count'], x['retweet_count']), axis=1)
df['engagement_per_follower'] = df.apply(lambda x: safe_divide(x['favorite_count'] + x['retweet_count'], x['user.followers_count']), axis=1)
df['friends_followers_ratio'] = df.apply(lambda x: safe_divide(x['user.friends_count'], x['user.followers_count']), axis=1)

engagement_metrics = ['like_retweet_ratio', 'engagement_per_follower', 'friends_followers_ratio']

for metric in engagement_metrics:
    rumor_values = df[df['label'] == 1][metric].replace([np.inf, -np.inf], np.nan).dropna()
    non_rumor_values = df[df['label'] == 0][metric].replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(rumor_values) > 0 and len(non_rumor_values) > 0:
        rumor_mean = rumor_values.mean()
        non_rumor_mean = non_rumor_values.mean()
        rumor_median = rumor_values.median()
        non_rumor_median = non_rumor_values.median()
        
        print(f"\n{metric.replace('_', ' ').title()}:")
        print(f"  Rumores - Média: {rumor_mean:.6f}, Mediana: {rumor_median:.6f}")
        print(f"  Não-rumores - Média: {non_rumor_mean:.6f}, Mediana: {non_rumor_median:.6f}")

# ===== ANÁLISE 5: FEATURES DE CONTEXTO TEMPORAL =====
print("\n" + "="*50)
print("5. INDICADORES DE CONTEXTO TEMPORAL")
print("="*50)

temporal_indicators = [
    'yesterday', 'today', 'now', 'just', 'minutes', 'hours', 'ago',
    'breaking', 'developing', 'live', 'ongoing', 'current', 'latest'
]

def count_temporal_indicators(text):
    if not isinstance(text, str):
        return 0
    text_lower = text.lower()
    return sum(1 for indicator in temporal_indicators if indicator in text_lower)

rumors['temporal_indicators'] = rumors['text'].apply(count_temporal_indicators)
non_rumors['temporal_indicators'] = non_rumors['text'].apply(count_temporal_indicators)

rumor_temporal_mean = rumors['temporal_indicators'].mean()
non_rumor_temporal_mean = non_rumors['temporal_indicators'].mean()

print(f"Média de indicadores temporais:")
print(f"  Rumores: {rumor_temporal_mean:.4f}")
print(f"  Não-rumores: {non_rumor_temporal_mean:.4f}")
print(f"  Diferença: {((rumor_temporal_mean - non_rumor_temporal_mean) / non_rumor_temporal_mean * 100):+.2f}%")

# ===== FEATURES RECOMENDADAS =====
print("\n" + "="*60)
print("FEATURES FILOGENÉTICAS RECOMENDADAS PARA IMPLEMENTAÇÃO")
print("="*60)

print("\n🎯 FEATURES COM MAIOR PODER DISCRIMINATIVO:")
print("1. Contagem de palavras de urgência (breaking, urgent, alert, etc.)")
print("2. Ratio de caracteres maiúsculos no texto")
print("3. Presença de múltiplas exclamações (!!!, !!!!) ")
print("4. Contagem de palavras com capitalização completa (CAPS)")
print("5. Densidade de números no texto")
print("6. Indicadores temporais (now, just, minutes ago)")
print("7. Padrões de vítimas/casualidades com números")

print("\n📊 FEATURES DE ENGAGEMENT SOCIAL:")
print("1. Ratio likes/retweets")
print("2. Engagement normalizado por seguidores")
print("3. Ratio amigos/seguidores do usuário")
print("4. Desvio do engagement médio do usuário")

print("\n🔤 FEATURES LINGUÍSTICAS AVANÇADAS:")
print("1. Densidade de hashtags por palavra")
print("2. Presença de padrões de tempo (HH:MM)")
print("3. Contagem de mentions específicas (@breaking, @news)")
print("4. Análise de sentimento polarizado (muito positivo/negativo)")

print("\n💡 IMPLEMENTAÇÃO SUGERIDA:")
print("- Criar features compostas combinando múltiplos indicadores")
print("- Usar normalização por comprimento do texto")
print("- Implementar detecção de padrões em sequência")
print("- Considerar features de interação entre características")

print("\nAnálise detalhada concluída!")