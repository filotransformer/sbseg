# Filo‑Transformer Repository

<p align="center"><img src="images/logo.png" alt="Filo-Transformer logo" width="280"/></p>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KAVv9DYrWz-FnOf6X6toRrwN5CW8bIiv?usp=sharing)

*Grafo de Alinhamento de Árvores Filogenéticas e Transformers para Identificação de Rumores e Fake News*

Este repositório contém o código, os dados e o material suplementar que acompanham o artigo submetido ao **Simpósio Brasileiro de Segurança da Informação e de Sistemas Computacionais (SBSeg 2025)**.

> **Resumo curto** ‑ Propomos um pipeline que combina *embeddings* semânticos, reconstrução filogenética com **Tree Alignment Graphs (TAGs)** e um **FT‑Transformer** supervisionado, alcançando **AUC 0.9489** e **F1 0.8393** no *benchmark* PHEME.

---

## 📂 Estrutura do projeto

```
filo-trans-repo/
├── datasets/                 # Conjuntos de dados prontos para uso
│   └── pheme/                # Versão curada do dataset PHEME (5 eventos + full)
├── images/
│   └── arquitetura-filo-transformer.pdf  # Diagrama de alto nível do pipeline
├── notebooks/
│   └── filo_transformer_v4_2.ipynb       # Notebook principal de experimentos
└── README.md
```

*Somente arquivos sensíveis, como variáveis de ambiente (`.env`), são omitidos do versionamento. Todas as dependências estão em `requirements.txt`. Veja a seção **Instalação**.*

---

## ⚙️ Instalação rápida

```bash
# 1. Clone o repositório
git clone https://github.com/<seu‑usuario>/filo-trans-repo.git
cd filo-trans-repo

# 2. Crie um ambiente Python 3.10+
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Instale as dependências mínimas
pip install --upgrade pip
pip install -r requirements.txt          # gere o arquivo com as libs abaixo
```

Dependências principais:

* `tensorflow>=2.16`
* `numpy`, `pandas`, `matplotlib`, `seaborn`
* `networkx`, `node2vec`
* `scikit-learn`
* `transformers`, `openai`

Veja `requirements.txt` para a lista completa.

> **Chave OpenAI:** para reproduzir os experimentos com GPT embeddings defina `OPENAI_API_KEY` em variável de ambiente ou arquivo `.env`.

---

### 📦 Dependências completas (`requirements.txt`)

```text
numpy
pandas
matplotlib
seaborn
networkx
node2vec
tensorflow
sklearn
transformers
openai
google
os
gc
math
random
time
warnings
getpass
```

---

**Chave OpenAI:** para reproduzir os experimentos com GPT embeddings defina `OPENAI_API_KEY` em variável de ambiente ou arquivo `.env`.

---

## 🚀 Como reproduzir os experimentos

0. **Executar no Google Colab:** clique no badge “Open In Colab” acima ou acesse diretamente [este link](https://colab.research.google.com/drive/1KAVv9DYrWz-FnOf6X6toRrwN5CW8bIiv?usp=sharing).
1. (Opcional) Baixe o repositório localmente e abra \`\` em Jupyter.
2. Execute as células sequencialmente.
3. Ao final são gravados:

   * métricas por *fold* (`results/*.json`)
   * gráficos de curva ROC e distribuição de atributos
   * modelo treinado `models/ft_transformer_best.h5`

Parâmetros de *cross‑validation*, *seed* e *threshold* podem ser alterados no início do notebook.

---

## 📊 Resultados principais (dataset PHEME)

| Modelo                            | Acc        | AUC        | Recall     | F1         |
| --------------------------------- | ---------- | ---------- | ---------- | ---------- |
| **Filo‑Transformer (Ours)**       | **0.8888** | **0.9489** | **0.8530** | **0.8393** |
| SAMGAT (Li *et al.* 2024)         | 0.864      | 0.930      | 0.816      | 0.863      |
| SEMTEC (Sharma & Srivastava 2024) | 0.920      | —          | —          | 0.890      |
| RvNN (Ma *et al.* 2018)           | 0.825      | —          | 0.764      | 0.820      |

> Tabela completa e protocolos de avaliação encontram‑se na Seção 4 do artigo.

---

## 📑 Dados

O corpus base é o **PHEME v.7** (Figshare). Esta cópia já está:

* convertida para **CSV** por evento
* com campo de veracidade em `label` (`1=rumeur/rumor`, `0=non‑rumor`)
* anonimizada para fins de revisão duplo‑cego.

Caso deseje reconstruir a partir do repositório oficial, siga `datasets/README_pheme.md` (não incluído).

---

## 📜 Licença

Distribuído sob a licença **MIT**. Veja [`LICENSE`](LICENSE) para detalhes. Os dados PHEME mantêm sua licença original (Creative Commons Attribution).

---



---

