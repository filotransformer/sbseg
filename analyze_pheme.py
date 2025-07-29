#!/usr/bin/env python3
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
# from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Carregar o dataset
print("Carregando dataset PHEME...")
df = pd.read_csv('/home/acauan/ufam/papers/01_sbseg_filo_trans/datasets/pheme/pheme_all.csv')

print(f"Dataset shape: {df.shape}")
print(f"Colunas: {list(df.columns)}")
print(f"Distribuição das labels:")
print(df['label'].value_counts())

# Limpeza de dados
print("\nLimpando dados...")
df = df.dropna(subset=['text', 'label'])
df = df[df['label'].isin([0, 1])]

print(f"Após limpeza: {df.shape}")
print(f"Nova distribuição das labels:")
print(df['label'].value_counts())

# Separar rumores e não-rumores
rumors = df[df['label'] == 1]['text'].tolist()
non_rumors = df[df['label'] == 0]['text'].tolist()

print(f"\nRumores: {len(rumors)}")
print(f"Não-rumores: {len(non_rumors)}")

# ===== ANÁLISE 1: CARACTERÍSTICAS BÁSICAS =====
print("\n" + "="*50)
print("ANÁLISE 1: CARACTERÍSTICAS BÁSICAS")
print("="*50)

def extract_features(texts):
    features = {
        'length': [],
        'word_count': [],
        'uppercase_ratio': [],
        'hashtag_count': [],
        'url_count': [],
        'mention_count': [],
        'exclamation_count': [],
        'question_count': [],
        'digit_ratio': [],
    }
    
    for text in texts:
        if isinstance(text, str):
            # Comprimento
            features['length'].append(len(text))
            
            # Contagem de palavras
            words = text.split()
            features['word_count'].append(len(words))
            
            # Ratio de maiúsculas
            uppercase_chars = sum(1 for c in text if c.isupper())
            features['uppercase_ratio'].append(uppercase_chars / len(text) if len(text) > 0 else 0)
            
            # Hashtags
            features['hashtag_count'].append(len(re.findall(r'#\w+', text)))
            
            # URLs
            features['url_count'].append(len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)))
            
            # Mentions
            features['mention_count'].append(len(re.findall(r'@\w+', text)))
            
            # Exclamações
            features['exclamation_count'].append(text.count('!'))
            
            # Interrogações
            features['question_count'].append(text.count('?'))
            
            # Ratio de dígitos
            digits = sum(1 for c in text if c.isdigit())
            features['digit_ratio'].append(digits / len(text) if len(text) > 0 else 0)
    
    return features

# Extrair características para rumores e não-rumores
print("Extraindo características para rumores...")
rumor_features = extract_features(rumors)
print("Extraindo características para não-rumores...")
non_rumor_features = extract_features(non_rumors)

# Comparar características
print("\nComparando características básicas:")
for feature in rumor_features.keys():
    rumor_mean = np.mean(rumor_features[feature])
    non_rumor_mean = np.mean(non_rumor_features[feature])
    diff_pct = ((rumor_mean - non_rumor_mean) / non_rumor_mean * 100) if non_rumor_mean != 0 else 0
    
    print(f"{feature}:")
    print(f"  Rumores: {rumor_mean:.4f}")
    print(f"  Não-rumores: {non_rumor_mean:.4f}")
    print(f"  Diferença: {diff_pct:+.2f}%")
    print()

# ===== ANÁLISE 2: PALAVRAS-CHAVE MAIS DISCRIMINATIVAS =====
print("\n" + "="*50)
print("ANÁLISE 2: PALAVRAS-CHAVE DISCRIMINATIVAS")
print("="*50)

# Preparar textos limpos
def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remover URLs, mentions, e caracteres especiais
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s#]', '', text)
    return text.lower().strip()

rumor_texts_clean = [clean_text(text) for text in rumors]
non_rumor_texts_clean = [clean_text(text) for text in non_rumors]

# Usar TF-IDF para encontrar palavras discriminativas
all_texts = rumor_texts_clean + non_rumor_texts_clean
labels = [1] * len(rumor_texts_clean) + [0] * len(non_rumor_texts_clean)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=5, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(all_texts)
feature_names = vectorizer.get_feature_names_out()

# Calcular pontuação média para cada classe
rumor_scores = np.mean(tfidf_matrix[:len(rumor_texts_clean)].toarray(), axis=0)
non_rumor_scores = np.mean(tfidf_matrix[len(rumor_texts_clean):].toarray(), axis=0)

# Encontrar palavras mais discriminativas
discrimination_scores = rumor_scores - non_rumor_scores
top_rumor_indices = np.argsort(discrimination_scores)[-20:][::-1]
top_non_rumor_indices = np.argsort(discrimination_scores)[:20]

print("Top 20 palavras/frases mais associadas a RUMORES:")
for i, idx in enumerate(top_rumor_indices, 1):
    word = feature_names[idx]
    score = discrimination_scores[idx]
    print(f"{i:2d}. {word:<25} (score: {score:.4f})")

print("\nTop 20 palavras/frases mais associadas a NÃO-RUMORES:")
for i, idx in enumerate(top_non_rumor_indices, 1):
    word = feature_names[idx]
    score = discrimination_scores[idx]
    print(f"{i:2d}. {word:<25} (score: {score:.4f})")

# ===== ANÁLISE 3: PADRÕES DE MÍDIA SOCIAL =====
print("\n" + "="*50)
print("ANÁLISE 3: PADRÕES DE MÍDIA SOCIAL")
print("="*50)

# Analisar métricas de engagement
social_metrics = ['favorite_count', 'retweet_count', 'user.followers_count', 'user.friends_count']

print("Comparando métricas de engagement:")
for metric in social_metrics:
    if metric in df.columns:
        rumor_values = df[df['label'] == 1][metric].dropna()
        non_rumor_values = df[df['label'] == 0][metric].dropna()
        
        if len(rumor_values) > 0 and len(non_rumor_values) > 0:
            rumor_mean = rumor_values.mean()
            non_rumor_mean = non_rumor_values.mean()
            rumor_median = rumor_values.median()
            non_rumor_median = non_rumor_values.median()
            
            print(f"\n{metric}:")
            print(f"  Rumores - Média: {rumor_mean:.2f}, Mediana: {rumor_median:.2f}")
            print(f"  Não-rumores - Média: {non_rumor_mean:.2f}, Mediana: {non_rumor_median:.2f}")
            
            if non_rumor_mean != 0:
                diff_pct = ((rumor_mean - non_rumor_mean) / non_rumor_mean * 100)
                print(f"  Diferença na média: {diff_pct:+.2f}%")

# ===== ANÁLISE 4: HASHTAGS ESPECÍFICAS =====
print("\n" + "="*50)
print("ANÁLISE 4: HASHTAGS MAIS COMUNS")
print("="*50)

def extract_hashtags(texts):
    hashtags = []
    for text in texts:
        if isinstance(text, str):
            hashtags.extend(re.findall(r'#(\w+)', text.lower()))
    return hashtags

rumor_hashtags = extract_hashtags(rumors)
non_rumor_hashtags = extract_hashtags(non_rumors)

print("Top 15 hashtags em RUMORES:")
rumor_hashtag_counts = Counter(rumor_hashtags)
for i, (hashtag, count) in enumerate(rumor_hashtag_counts.most_common(15), 1):
    print(f"{i:2d}. #{hashtag:<20} ({count} vezes)")

print("\nTop 15 hashtags em NÃO-RUMORES:")
non_rumor_hashtag_counts = Counter(non_rumor_hashtags)
for i, (hashtag, count) in enumerate(non_rumor_hashtag_counts.most_common(15), 1):
    print(f"{i:2d}. #{hashtag:<20} ({count} vezes)")

# ===== INSIGHTS PARA FEATURES FILOGENÉTICAS =====
print("\n" + "="*60)
print("INSIGHTS PARA MELHORAR FEATURES FILOGENÉTICAS")
print("="*60)

print("\n1. CARACTERÍSTICAS TEXTUAIS DISCRIMINATIVAS:")
print("   - Rumores tendem a ter mais exclamações e interrogações")
print("   - Ratio de maiúsculas pode ser indicativo")
print("   - Comprimento e densidade de palavras diferem")

print("\n2. PADRÕES LINGUÍSTICOS ESPECÍFICOS:")
print("   - Implementar detecção de palavras de urgência/alerta")
print("   - Medir densidade de hashtags e mentions")
print("   - Analisar padrões de pontuação emocional")

print("\n3. FEATURES PROPOSTAS PARA IMPLEMENTAÇÃO:")
print("   - Ratio de maiúsculas no texto")
print("   - Densidade de exclamações e interrogações")
print("   - Presença de palavras de urgência (breaking, alert, urgent)")
print("   - Ratio de hashtags por palavra")
print("   - Detecção de padrões de números/estatísticas")
print("   - Análise de sentimento específica")

print("\n4. FEATURES DE ENGAGEMENT SOCIAL:")
print("   - Normalizar métricas por número de seguidores")
print("   - Criar ratios de engagement (likes/retweets)")
print("   - Analisar velocidade de propagação")

print("\nAnálise concluída!")