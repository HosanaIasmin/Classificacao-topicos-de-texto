import gradio as gr
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer


topic_model = BERTopic.load("desastres_model")


embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def classify_text(input_text):
  
    embeddings = embedding_model.encode([input_text])
    print("Embeddings gerados:", embeddings) 
    
    
    topics, probs = topic_model.transform([input_text], embeddings)
    
    
    print("Tópicos identificados:", topics)
    print("Probabilidades:", probs)
    
   
    #if probs[0] < 0.2:  # Limite ajustado para tópicos com baixa confiança
        #return "Confiança insuficiente para determinar o tópico."

    topic_info = topic_model.get_topic(topics[0])

    if topic_info:
        topic_name = topic_info[0][0]
        confidence = probs[0]
        return f"Tópico detectado: {topic_name} (Confiança: {confidence:.2f})"
    else:
        return "Tópico não encontrado."

# Interface Gradio
iface = gr.Interface(
    fn=classify_text,
    inputs=gr.Textbox(lines=3, placeholder="Digite sua mensagem sobre o desastre natural..."),
    outputs="text",
    title="Classificação de Tópicos de Desastres Naturais",
    description="Digite uma mensagem sobre um desastre natural e veja o tópico correspondente."
)

iface.launch()
