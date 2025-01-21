import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from umap import UMAP
import hdbscan
from bertopic import BERTopic

# Baixar as stop words em português
nltk.download('stopwords')
stop_words = stopwords.words('portuguese')

# Dados de exemplo simples
docs = [
"Estou vendo um deslizamento de terra bloquear a estrada principal, precisamos de ajuda urgente!",
"O morro está desmoronando por causa das chuvas fortes, alertem as autoridades imediatamente!",
"Um grande deslizamento acabou de atingir algumas casas aqui, muitas pessoas estão em perigo!",
"Há um deslizamento de terra acontecendo agora mesmo na encosta, evacuem a área rapidamente!",
"O fogo está se espalhando rápido pela floresta, precisamos de bombeiros aqui o quanto antes!",
"Estou vendo um incêndio enorme próximo ao bairro, evacuem as casas agora!",
"O vento está espalhando o fogo muito rápido, chame os bombeiros imediatamente!",
"As chamas estão muito altas, o incêndio está fora de controle e se aproximando das residências!",
"Acabou de acontecer um terremoto muito forte, muitos prédios estão danificados, precisamos de socorro!",
"Sentimos um grande tremor aqui, algumas construções estão caindo, enviem ajuda urgente!",
"As ruas estão rachadas por causa do terremoto, temos pessoas presas, avisem os resgatistas!",
"Um terremoto acabou de acontecer, há muitos feridos e desabrigados, precisamos de ajuda médica!",
"Uma tempestade está causando inundações aqui, as ruas estão completamente alagadas, precisamos de socorro!",
"Os ventos fortes da tempestade estão derrubando árvores e postes, é perigoso, mandem ajuda!",
"A tempestade está muito forte, estamos sem energia e há risco de desabamento, avisem as autoridades!",
"Estamos enfrentando uma tempestade severa, a água está invadindo as casas, precisamos de resgate!",
"Não temos água há semanas devido à seca, as plantações estão morrendo, precisamos de ajuda urgente!",
"A seca está tão grave que os rios secaram, não temos como irrigar as plantações, avisem as autoridades!",
"Os reservatórios estão vazios por causa da seca prolongada, precisamos de uma solução rápida!",
"A seca está causando uma crise de água aqui, não conseguimos abastecer as comunidades, enviem suporte!",
"Acabamos de ver uma grande onda se aproximando, parece um tsunami, todos devem evacuar imediatamente!",
"O mar recuou rapidamente, isso pode ser um tsunami, precisamos avisar a todos para saírem da costa!",
"Há um tsunami chegando, a água está subindo rápido, evacuem agora!",
"Acabamos de ser atingidos por um tsunami, a destruição é imensa, precisamos de ajuda urgente!",
"Estou vendo um tornado se formando, está vindo na direção da cidade, precisamos evacuar agora!",
"O tornado está destruindo tudo à sua volta, avisem para todos se abrigarem imediatamente!",
"Acabamos de avistar um tornado, está vindo em nossa direção, corram para um abrigo seguro!",
"O tornado está aqui, está arrancando telhados e árvores, precisamos de ajuda urgente!"
]

# Usar SentenceTransformers para gerar embeddings
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
embeddings = model.encode(docs, show_progress_bar=True)

# Configurar o modelo UMAP
umap_model = UMAP(n_neighbors=5, n_components=2, metric='cosine')

# Configurar o modelo HDBSCAN
hdbscan_model = hdbscan.HDBSCAN(min_samples=2, min_cluster_size=2, prediction_data=True)

# Vocabulário expandido para desastres naturais
vocabulary = [
    'deslizamento', 'terremoto', 'incêndio', 'tempestade', 'seca', 'tornado', 'tsunami'
]

# Criar o vectorizer com o vocabulário expandido
vectorizer_model = CountVectorizer(stop_words=stop_words, vocabulary=vocabulary)

# Criar e treinar o modelo BERTopic com os modelos UMAP e HDBSCAN ajustados
topic_model = BERTopic(
    language="multilingual", 
    umap_model=umap_model, 
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model
)

# Treinar o modelo
topics, probs = topic_model.fit_transform(docs, embeddings)

# Exibir os tópicos
all_topics = topic_model.get_topics()
for topic_id, topic_words in all_topics.items():
    print(f"Tópico {topic_id}: {topic_words}")

# Salvar o modelo
topic_model.save("desastres_model")
