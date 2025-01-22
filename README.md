# Classificação de Ocorrências de Desastres Naturais 🌍⚡🔥🌊

Este projeto utiliza **BERTopic** para identificar e classificar ocorrências de desastres naturais a partir de frases em português. Foi desenvolvido para categorizar textos relacionados a eventos como deslizamentos, incêndios, terremotos, tempestades, entre outros.

## 📋 Objetivo

O objetivo é criar um modelo capaz de classificar automaticamente relatos de desastres naturais, permitindo que equipes de resposta rápida priorizem ações com base nos tipos de ocorrências relatadas.


---

## 🚀 Funcionalidades

1. **Classificação de Desastres Naturais**: Classifica frases em categorias como deslizamentos, incêndios, terremotos, tempestades, etc.
2. **Pipeline Automatizado**: Integração com GitHub Actions para CI/CD.
3. **Deploy no Hugging Face Spaces**: Interface de visualização do modelo no Hugging Face.

---

## 🛠️ Tecnologias Utilizadas

- **Python 3.8+**
- **Sentence Transformers** para embeddings.
- **BERTopic** para modelagem de tópicos.
- **UMAP e HDBSCAN** para redução de dimensionalidade e agrupamento.
- **GitHub Actions** para CI/CD.
- **Hugging Face Spaces** para deploy.

---

## 📊 Dataset

O dataset utilizado contém frases simulando alertas de desastres naturais, divididas em categorias como deslizamentos, incêndios, terremotos e tempestades.

Exemplo de frases:
- *"A forte chuva provocou um deslizamento de terra nas montanhas, isolando várias aldeias."*
- *"O terremoto de magnitude 7.2 abalou a costa, causando danos significativos."*

---

## ⚙️ Como Usar

### Pré-requisitos

1. Python 3.8+ instalado.
2. Hugging Face CLI configurado com seu token.

### 1️⃣ Clonar o Repositório

```bash
git clone https://github.com/username/disaster-classification.git
cd disaster-classification

##Instalar Dependências
pip install -r requirements.txt

#Treinar o modelo
python src/model_training.py

#testar o modelo
pytest tests/test_model.py

#Deploy no HUgging Face Spaces
huggingface-cli login
huggingface-cli repo create disaster-classification --type=space
git push hf main
