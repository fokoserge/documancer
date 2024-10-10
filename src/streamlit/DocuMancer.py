import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import gdown
import tensorflow as tf
import easyocr
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification, BertConfig

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Lambda
import os
from keras.saving import register_keras_serializable

import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import json
from keras.models import load_model

import lime
import lime.lime_image
import lime.lime_text
from skimage.segmentation import mark_boundaries
import re
from tensorflow.keras.mixed_precision import set_global_policy

#-------------------------------------------------------------------------------
# Paramétrage de lancement
#-------------------------------------------------------------------------------
# Activer la précision mixte
set_global_policy('mixed_float16')

drive_model_BERT_dir = '/content/drive/MyDrive/formation Datascientest/jul24_bds_extraction/ETAPE 3/saved_BERTmodel_tensorflow_01'

model_path_multimodal = "/content/drive/MyDrive/formation Datascientest/jul24_bds_extraction/ETAPE 3/models/multimodal_model_complet_group_demo.keras"
model_path_cnn = "/content/drive/MyDrive/formation Datascientest/jul24_bds_extraction/CNN/saved_modelcnn.keras"
test_dataset_tf_path = "/content/drive/MyDrive/formation Datascientest/jul24_bds_extraction/ETAPE 3/models/merged_test_dataset_final_group"
image_path = "/content/drive/MyDrive/formation Datascientest/jul24_bds_extraction/_presentation/images"
name_app ='DocuMancer'
couleur_fond = '#2ED4DA'
couleur_police = '#382DD5'
#-------------------------------------------------------------------------------
# Paramétrage de lancement
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Outillages et initialisation
#-------------------------------------------------------------------------------
@register_keras_serializable()
class TFBertLayer(tf.keras.layers.Layer):
    def __init__(self, bert_model_dir, **kwargs):
        # Enlever 'bert_model_dir' des kwargs pour éviter de le passer à la classe de base
        kwargs.pop('bert_model_dir', None)
        super(TFBertLayer, self).__init__(**kwargs)
        self.bert_model_dir = bert_model_dir
        # Charger la configuration avec output_hidden_states=True
        config = BertConfig.from_pretrained(self.bert_model_dir, output_hidden_states=True)
        self.bert_model = TFBertForSequenceClassification.from_pretrained(self.bert_model_dir, config=config)

    def call(self, inputs, training=False):
        input_ids = inputs[0]
        attention_mask = inputs[1]
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask, training=training)
        hidden_states = outputs.hidden_states
        last_hidden_state = hidden_states[-1]
        cls_output = last_hidden_state[:, 0, :]
        return cls_output

    def get_config(self):
        config = super(TFBertLayer, self).get_config()
        config['bert_model_dir'] = self.bert_model_dir
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def load_model_multi(model_path):
    # Charger le modèle pré-entraîné avec Keras
    config = BertConfig.from_pretrained(drive_model_BERT_dir, output_hidden_states=True)
    # Charger le modèle
    model = load_model(model_path, custom_objects={'TFBertLayer': TFBertLayer})
    return model

def load_model_CNN(model_path):
    # Charger le modèle
    model = tf.keras.models.load_model(model_path)
    return model

def load_model_BERT(model_path):
    # Charger le modèle
    model = TFBertForSequenceClassification.from_pretrained(model_path)
    return model

def classify_image_text_keras(model, image_tensor, input_ids, attention_mask):
    # Obtenir les prédictions du modèle
    output = model.predict([input_ids.numpy(), image_tensor.numpy(), attention_mask.numpy()])
    predicted_class = np.argmax(output, axis=1)[0]
    confidence_score = np.max(output) * 100
    return predicted_class, confidence_score

def classify_image_keras(model, image_tensor):
    # Obtenir les prédictions du modèle
    output = model.predict([ image_tensor.numpy()])
    predicted_class = np.argmax(output, axis=1)[0]
    confidence_score = np.max(output) * 100
    return predicted_class, confidence_score

def classify_text_keras(model, input_ids, attention_mask):
    # Convertir input_ids et attention_mask en tenseurs
    input_ids_tensor = tf.convert_to_tensor(input_ids, dtype=tf.int32)
    attention_mask_tensor = tf.convert_to_tensor(attention_mask, dtype=tf.int32)

    # Obtenir la sortie du modèle
    output = model([input_ids_tensor, attention_mask_tensor])

    # Vérifier si la sortie est un objet TensorFlow contenant des logits
    if isinstance(output, tf.Tensor):
        logits = output
    else:
        logits = output.logits

    # Utiliser np.argmax pour obtenir la classe prédite
    predicted_class = np.argmax(logits, axis=-1)[0]  # axis=-1 pour la dernière dimension
    confidence_score = tf.nn.softmax(logits)[0].numpy().max() * 100

    return predicted_class, confidence_score

def download_model_from_drive(drive_url, output_path):
    gdown.download(drive_url, output_path, quiet=False)

#-------------------------------------------------------------------------------
# Outillages et initialisation
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# CSS
#-------------------------------------------------------------------------------
st.markdown("""
    <style>
    /* Styles pour le bouton de téléchargement */
    .stDownloadButton > button, .stDownloadButton > button * {
        background-color: #2ED4DA;
        color: #382DD5;
        border: none;
        padding: 2px 6px !importants;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        cursor: pointer;
        border-radius: 15px;
        font-family: Arial, sans-serif; /* Ajout de la police */
        font-weight: bold !important;
    }

    /* Styles pour les boutons LinkedIn */
    a.linkedin-button {
        display: inline-block; /* Changement de 'block' à 'inline-block' */
        background-color: #2ED4DA !important;
        color: #382DD5 !important;
        font-weight: bold;
        padding: 2px 6px;
        text-decoration: none !important;
        border: none;
        border-radius: 15px;
        text-align: center;
        margin: 10px auto;
        width: 200px;
    }

    /* Appliquer les styles aux différents états du lien */
    a.linkedin-button:link,
    a.linkedin-button:visited,
    a.linkedin-button:hover,
    a.linkedin-button:active {
        background-color: #2ED4DA !important;
        color: #382DD5 !important;
        text-decoration: none !important;
    }

    /* Styles pour la liste et les éléments de liste */
    ul.linkedin-list {
        list-style-type: none;
        padding: 0; /* Supprimer le padding par défaut */
        margin: 0;  /* Supprimer la marge par défaut */
        text-align: center; /* Centrer le contenu de la liste */
    }

    ul.linkedin-list li {
        display: block;
        margin-bottom: 2px; /* Ajouter de l'espace entre les boutons */
    }

    /* Justifier le texte des paragraphes */
    .content p, .content li {
        text-align: justify;
    }

    /* Styles pour le bloc de prédiction */
    .prediction-box {
        background-color: #2ED4DA;
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
    }

    .prediction-box h3 {
        margin: 0;
        color: #382DD5;
        text-align: center;
    }

    .prediction-box h3 span {
        font-weight: bold;
    }

    .prediction-box p {
        font-size: 18px;
        margin: 5px 0;
        color: #382DD5;
        text-align: center;
    }

    .prediction-box p span {
        font-weight: bold;
    }

    /* Centrer l'image dans la fenêtre modale en plein écran */
    div[role="dialog"] .stImage img {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }

    </style>
    """, unsafe_allow_html=True)

# Icône de baguette magique (vous pouvez choisir n'importe quelle icône de Font Awesome ou une autre bibliothèque d'icônes)
icon = "🪄"  # Ici c'est l'emoji "baguette magique", mais on pourrait aussi utiliser une icône Font Awesome

# Créer le titre avec style
title_html = f"""
    <h1 style='text-align: center; color: #2ED4DA;'>{name_app} {icon}</h1>
"""
#-------------------------------------------------------------------------------
# CSS
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# méthode principale
#-------------------------------------------------------------------------------
def main():
    #---------------------------------------------------------------------------
    # Sidebar
    #---------------------------------------------------------------------------
    image_datascientest = Image.open(image_path + '/datascientest.png')
    st.sidebar.image(image_datascientest, use_column_width=True)
    st.sidebar.markdown(title_html, unsafe_allow_html=True)

    page = st.sidebar.radio('Aller à', [
        'Introduction au Projet',
        'Le dataset RVL-CDIP',
        'Stratégies déployées',
        'Prédiction par nos modèles',
        'Conclusion & perspectives'
    ])
    st.sidebar.markdown('<br>', unsafe_allow_html=True)
    st.sidebar.markdown(
        '''
        <div style="background-color: #0E1117; padding: 10px; border-radius: 5px; text-align: center;">
            <h3 style="margin-bottom: 20px;">Projet DS - Promotion Bootcamp été 2024</h3>
            <h3>Participants</h3>
            <ul class="linkedin-list">
                <li>
                    <a href="https://www.linkedin.com/in/kevin-ory" target="_blank" class="linkedin-button">Kevin Ory</a>
                </li>
                <li>
                    <a href="https://www.linkedin.com/in/kuate-foko-serge-ulrich" target="_blank" class="linkedin-button">Kuate Foko Serge Ulrich</a>
                </li>
                <li>
                    <a href="https://www.linkedin.com/in/nadir-ali-ahmed" target="_blank" class="linkedin-button">Nadir Ali Ahmed</a>
                </li>
                <li>
                    <a href="https://www.linkedin.com/in/xavier-truong-056837295" target="_blank" class="linkedin-button">Xavier Truong</a>
                </li>
            </ul>
        </div>
        ''',
        unsafe_allow_html=True
    )
    #---------------------------------------------------------------------------
    # Sidebar
    #---------------------------------------------------------------------------


    #---------------------------------------------------------------------------
    # Introduction au Projet
    #---------------------------------------------------------------------------
    if page == 'Introduction au Projet':
        # st.image(image_path + '/Documancer.gif', use_column_width=True)
        st.image(image_path + '/Documancer3.gif', use_column_width=True)
        st.title('présentation de '+ name_app)
        st.subheader('Introduction et Contexte du Projet')
        st.markdown('''
        Dans un monde où de nombreux secteurs comme la finance, la santé, et le juridique sont submergés de documents physiques et numériques, l'automatisation de leur gestion est un enjeu crucial.
        Notre projet s'est concentré sur la classification automatique de documents en utilisant des techniques avancées de deep learning.
        Le dataset que nous avons utilisé, RVL-CDIP, contient 400 000 images de documents répartis en 16 catégories (lettres, factures, mémos, etc.).
        ''')
        st.subheader('Objectifs du Projet')
        st.markdown('''
        L'objectif principal du projet est de développer un modèle performant capable de classer automatiquement les documents en fonction de leur type.
        Nous avons exploré plusieurs approches, notamment les réseaux convolutifs (CNN) pour les images, BERT pour l'analyse textuelle après OCR, et une approche multimodale combinant les deux pour optimiser les performances.
        ''')

    #---------------------------------------------------------------------------
    # Le dataset RVL-CDIP
    #---------------------------------------------------------------------------
    elif page == 'Le dataset RVL-CDIP':

        st.title('Le dataset RVL-CDIP')
        st.markdown('''
        Le dataset **RVL-CDIP** (Ryerson Vision Lab Complex Document Information Processing) est une ressource de référence dans le domaine de la classification de documents.
        Il contient **400 000 images de documents numérisés**, réparties en **16 catégories**, offrant une diversité qui permet de tester la capacité des modèles à reconnaître et différencier des types de documents variés.
        ''')

        st.subheader('Pourquoi RVL-CDIP ?')
        st.markdown('''
        Ce dataset a été choisi pour sa richesse et sa diversité, mais aussi pour les défis qu'il représente :
        - **Données variées et hétérogènes** : Les documents proviennent de sources multiples, ce qui signifie que les modèles doivent être capables de gérer une grande diversité de structures de documents.
        - **Conditions de numérisation réalistes** : Les images ont été numérisées à partir de documents papier avec des niveaux de bruit et de qualité très variés, rendant le dataset proche des conditions réelles rencontrées en entreprise.
        - **Large volume de données** : Avec ses 400 000 images, RVL-CDIP permet de tester des modèles de deep learning à grande échelle, ce qui est crucial pour obtenir des modèles robustes.
        ''')

        # Division en deux colonnes pour afficher le texte et le pie chart côte à côte
        col1, col2 = st.columns([2, 1])  # La première colonne occupe 2/3, la seconde 1/3

        # Texte dans la colonne de gauche
        with col1:
            st.subheader('Structure des Données')
            st.markdown('''
            Le dataset est divisé en trois ensembles pour faciliter l'entraînement et l'évaluation des modèles :
            - **Ensemble d'entraînement** : 320 000 images utilisées pour ajuster les poids des modèles.
            - **Ensemble de validation** : 40 000 images permettant de vérifier la capacité du modèle à généraliser sur des données non vues.
            - **Ensemble de test** : 40 000 images pour évaluer la performance finale du modèle.
            ''')

        # Pie chart dans la colonne de droite
        with col2:
            # Création des données pour le pie chart
            labels = ['Entraînement', 'Validation', 'Test']
            sizes = [320000, 40000, 40000]
            colors = ['#66b3ff', '#99ff99', '#ffcc99']
            explode = (0.1, 0, 0)  # Mettre en avant le segment de l'entraînement

        # Création du pie chart
        fig, ax = plt.subplots()
        ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
              shadow=True, startangle=90)
        ax.axis('equal')  # Assure que le pie chart est circulaire.

        # Afficher le graphique dans Streamlit
        st.pyplot(fig)

        st.markdown('''
        Chaque image est associée à une étiquette correspondant à sa classe parmi les 16 catégories suivantes :
        1. Lettre
        2. Formulaire
        3. Messagerie électronique
        4. Manuscrit
        5. Publicité
        6. Rapport scientifique
        7. Publication scientifique
        8. Spécification
        9. Dossier de fichiers
        10. Article de presse
        11. Budget
        12. Facture
        13. Présentation
        14. Questionnaire
        15. CV
        16. Mémo
        ''')
        image_labels = Image.open(image_path + '/labels_example.png')
        st.image(image_labels, caption="Composition et exemples du dataset RVL-CDIP", use_column_width=True)

        st.subheader('Caractéristiques Techniques des Images')
        st.markdown('''
        - **Format** : Images en niveaux de gris, ce qui permet de réduire la complexité du traitement tout en conservant les informations essentielles.
        - **Résolution** : Faible résolution à **72 dpi**, ce qui ajoute un défi pour la reconnaissance des détails dans les documents.
        - **Qualité** : La qualité des images varie, avec des problèmes courants comme le bruit, les taches, les pliures, et les artefacts de numérisation.
        ''')

        st.subheader('Défis et Enjeux')
        st.markdown('''
        L'utilisation du dataset RVL-CDIP ne vient pas sans défis :
        - **Variabilité de qualité** : La qualité inégale des images rend difficile l'identification des caractéristiques visuelles des documents. Certains documents présentent des défauts tels que des tâches, des déformations, ou un faible contraste.
        - **Similarité entre classes** : Certaines catégories de documents présentent des caractéristiques visuelles très proches (par exemple, les lettres et les mémos), ce qui complique la tâche de classification même pour les modèles avancés.
        - **Volume de données** : La gestion de 400 000 images demande une infrastructure de calcul adaptée, notamment pour le stockage, la prévisualisation et le traitement en batch lors de l'entraînement des modèles.
        ''')

        st.subheader('Pourquoi ce Dataset est Essentiel ?')
        st.markdown('''
        Le dataset RVL-CDIP est un standard pour la recherche sur la classification de documents, ce qui permet de comparer les performances de nos modèles avec celles d'autres approches.
        En travaillant avec ce dataset, notre objectif est de développer des modèles capables de généraliser à partir de données réelles et de traiter efficacement les images de documents, même dans des conditions difficiles.
        ''')

        # Titre principal de la présentation
        st.title("Présentation des Données du Projet")

        # Section 1 : Qualité des Images
        with st.expander("1. Qualité des Images"):
            st.markdown("""
            **Impact de la qualité des images :**
            La qualité des images (résolution, bruit, etc.) influence directement la capacité du modèle CNN à extraire des caractéristiques pertinentes.

            **Problèmes identifiés :**
            Certaines images présentent des taches, des pliures ou des déformations, tandis que d'autres sont affectées par des problèmes de résolution ou de flou,
            rendant difficile l'extraction d'informations.
            """)
            # Ajout de l'image illustrant la qualité des images
            st.image(image_path + '/mauvaise_image_1.png', caption="Exemple de mauvaises Images", use_column_width=True)
            # Option de téléchargement de l'image
            with open(image_path + "/mauvaise_image_1.png", "rb") as img_file:
                st.download_button(label="Télécharger l'image", data=img_file, file_name="mauvaise_image_1.png", mime="image/png")

            # Ajout de l'image illustrant la qualité des images
            st.image(image_path + '/mauvaise_image_2.png', caption="Exemple de mauvaises Images", use_column_width=True)
            # Option de téléchargement de l'image
            with open(image_path + "/mauvaise_image_2.png", "rb") as img_file:
                st.download_button(label="Télécharger l'image", data=img_file, file_name="mauvaise_image_2.png", mime="image/png")

        # Section 2 : Erreurs de Labellisation
        with st.expander("2. Erreurs de Labelisation"):
            st.markdown("""
            **Ambiguïté dans la classification :**
            Certains documents peuvent appartenir à plusieurs catégories, ce qui complique l'étiquetage. Cette ambiguïté crée des incohérences dans
            les annotations et affecte la qualité des données.
            """)

            # Ajout de l'image illustrant les erreurs de labellisation
            st.image(image_path + '/erreur_de_label_1.png', caption="Erreurs de Labellisation", use_column_width=True)
            # Option de téléchargement de l'image
            with open(image_path + "/erreur_de_label_1.png", "rb") as img_file:
                st.download_button(label="Télécharger l'image", data=img_file, file_name="erreur_de_label_1.png", mime="image/png")

            # Ajout de l'image illustrant les erreurs de labellisation
            st.image(image_path + '/erreur_de_label_2.png', caption="Erreurs de Labellisation", use_column_width=True)
            # Option de téléchargement de l'image
            with open(image_path + "/erreur_de_label_2.png", "rb") as img_file:
                st.download_button(label="Télécharger l'image", data=img_file, file_name="erreur_de_label_2.png", mime="image/png")

            st.markdown("""
            **Erreurs de classification :**
            Les erreurs d'annotation dues à des similitudes visuelles, à l'erreur humaine ou à la définition personnelle du type de document
            affectent directement les performances des modèles, en particulier pour les classes qui partagent des caractéristiques communes.
            """)
            # Ajout de l'image illustrant les erreurs de labellisation
            st.image(image_path + '/error_accross_dataset.png', caption="Taux d'erreurs de Label par dataset", use_column_width=True)
            # Option de téléchargement de l'image
            with open(image_path + "/error_accross_dataset.png", "rb") as img_file:
                st.download_button(label="Télécharger l'image", data=img_file, file_name="error_accross_dataset.png", mime="image/png")

        # Section 3 : Types d'Images et Structure des Documents
        with st.expander("3. Types d'Images et Structure des Documents"):
            st.markdown("""
            **Variété des Documents :**
            Le dataset contient différents types de documents, tels que des rapports, factures et correspondances, provenant de l'industrie
            du tabac. Ces documents ont des structures variées, complexes et parfois simples. Cependant, le fait qu'ils appartiennent à l'industrie
            du tabac pose un problème de représentativité dans un contexte plus global où la structure de fond et de forme peut être totalement différente.
            """)



    #---------------------------------------------------------------------------
    # Stratégies déployées
    #---------------------------------------------------------------------------
    elif page == 'Stratégies déployées':
        st.title('Stratégies déployées')
        # Ajout de la barre de sélection pour choisir le modèle
        image_strat = Image.open(image_path + '/strategies.png')
        st.image(image_strat, use_column_width=True)

        model_choice = st.selectbox(
            "Choisissez un modèle de déploiement",
            ("CNN", "BERT", "BERT-CNN")
        )

        # Si le modèle sélectionné est CNN, affichage des informations détaillées
        if model_choice == "CNN":
            st.subheader("Modèle CNN : Classification de documents par les images")
            st.image(image_path + "/cnn_workflow.png", caption="Workflow CNN", use_column_width=True)

            st.markdown('''
            **Pourquoi avons-nous choisi CNN ?**
            - **Extraction des caractéristiques visuelles** : Les CNN sont idéaux pour capturer des informations visuelles clés telles que la structure de la mise en page et les formes des caractères. Cela les rend particulièrement efficaces pour distinguer différents types de documents (lettres, factures, mémorandums).
            - **Robustesse face aux variations** : Grâce à leur architecture, les CNN peuvent reconnaître des motifs, peu importe leur position ou échelle dans l'image. Cela assure une classification fiable, même lorsque la structure des documents varie.
            - **Traitement des images bruitées** : Les CNN sont performants même avec des images de qualité variable (artefacts de numérisation, flou). En utilisant plusieurs couches convolutives, ils filtrent les éléments non pertinents tout en conservant les informations essentielles.
            - **Transfer learning** : En exploitant des modèles pré-entraînés comme VGG16 et ResNet, nous avons pu tirer parti de connaissances acquises pour réduire le temps d'entraînement et améliorer la précision.
            - **Adaptabilité aux grands volumes** : Les CNN sont adaptés aux grands datasets comme RVL-CDIP (400 000 images), capables de gérer des millions de paramètres tout en maintenant une performance stable et précise à grande échelle.
            ''')
            st.markdown("**Étapes suivies pour entraîner CNN :**")
            with st.expander("1. Prétraitement des images avec OpenCV"):
              st.code('''
                  image_resized = cv2.resize(image_np, (224, 224), interpolation=cv2.INTER_CUBIC)
                  gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)
                  blurred = cv2.medianBlur(gray, 5)
                  _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
              ''', language='python')
              st.write("Redimensionnement des images à 224x224 pixels et conversion en niveaux de gris pour simplifier l'image, suivi d'un flou médian pour réduire le bruit.")
              # Ajout de l'image liée à cette étape
              image_preprocessing = Image.open(image_path + '/cnn_pretraitement.png')
              st.image(image_preprocessing, caption="Prétraitement des images avec OpenCV", use_column_width=True)
              # Option de téléchargement de l'image
              with open(image_path + "/cnn_pretraitement.png", "rb") as img_file:
                  st.download_button(label="Télécharger l'image", data=img_file, file_name="pretraitement_opencv.png", mime="image/png")

            with st.expander("2. Préparation des données pour TensorFlow"):
                st.code('''
                    # Conversion des images en objets TensorFlow Dataset
                    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
                    dataset = dataset.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
                    dataset = dataset.batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)
                ''', language='python')
                st.write("Création de datasets TensorFlow pour un chargement plus rapide et un traitement en parallèle, optimisant l'entraînement du modèle.")

            with st.expander("3. Modèles et entraînement avec et sans Transfer Learning"):
                st.markdown('''
                **Paramètres d'entraînement avec transfer learning:**
                - **Modèles pré-entraînés** : VGG16 et ResNet50, adaptés pour la classification de documents à partir de leurs images en niveaux de gris.
                - **Nombre d'époques** : 40 avec un taux d'apprentissage initial de 0.0001.
                ''')
                image_results = Image.open(image_path + '/cnn_w_vgg16.png')
                st.image(image_results, caption="Architecture modèle CNN avec vgg16", use_column_width=True)
                # Option de téléchargement de l'image
                with open(image_path + "/cnn_w_vgg16.png", "rb") as img_file:
                    st.download_button(label="Télécharger l'image", data=img_file, file_name="cnn_w_vgg16.png", mime="image/png")

                st.markdown('''
                **Paramètres d'entraînement sans Transfer Learning:**
                - **Entraînement** : Entraînement de CNN personnalisés avec 4 couches convolutives.
                - **Taux de régularisation (Dropout)** : 0.1 pour éviter le surapprentissage.
                - **Nombre d'époques** : 40 avec un taux d'apprentissage initial de 0.0001.
                ''')
                image_results_2 = Image.open(image_path + '/cnn_wo_vgg16.png')
                st.image(image_results_2, caption="Architecture modèle CNN avec vgg16", use_column_width=True)
                # Option de téléchargement de l'image
                with open(image_path + "/cnn_wo_vgg16.png", "rb") as img_file:
                    st.download_button(label="Télécharger l'image", data=img_file, file_name="cnn_wo_vgg16.png", mime="image/png")

                st.write("Les modèles pré-entraînés ont convergé plus rapidement et ont montré une meilleure capacité de généralisation sur le jeu de test.")

            with st.expander("4. Résultats de la Classification"):
                st.markdown('''
                **Précision sur les images non prétraitées :**
                - **VGG16 (Transfer Learning)** : 86% sur le jeu de test.
                - **ResNet50 (Transfer Learning)** : 85% sur le jeu de test.
                - **CNN entraîné depuis zéro** : 58% sur le jeu de test.
                ''')

                image_results_3 = Image.open(image_path + '/cnn_results_vgg16.png')
                st.image(image_results_3, caption="Comparaison des performances des différents modèles CNN", use_column_width=True)
                # Option de téléchargement de l'image
                with open(image_path + "/cnn_results_vgg16.png", "rb") as img_file:
                    st.download_button(label="Télécharger les résultats", data=img_file, file_name="cnn_results_vgg16.png", mime="image/png")

                image_results_4 = Image.open(image_path + '/cnn_results_ResNet50.png')
                st.image(image_results_4, caption="Comparaison des performances des différents modèles CNN", use_column_width=True)
                # Option de téléchargement de l'image
                with open(image_path + "/cnn_results_ResNet50.png", "rb") as img_file:
                    st.download_button(label="Télécharger les résultats", data=img_file, file_name="cnn_results_ResNet50.png", mime="image/png")

                image_results_5 = Image.open(image_path + '/cnn_results_zero.png')
                st.image(image_results_5, caption="Comparaison des performances des différents modèles CNN", use_column_width=True)
                # Option de téléchargement de l'image
                with open(image_path + "/cnn_results_zero.png", "rb") as img_file:
                    st.download_button(label="Télécharger les résultats", data=img_file, file_name="cnn_results_zero.png", mime="image/png")


                st.markdown('''
                **Précision sur les images prétraitées :**
                - **VGG16** : 59%, montrant une baisse de précision due à la perte d'informations lors du prétraitement.
                - **CNN depuis zéro** : 34%, soulignant l'importance de la richesse des informations visuelles pour l'entraînement.
                ''')
                image_results_6 = Image.open(image_path + '/cnn_results_processed_vgg16.png')
                st.image(image_results_6, caption="Comparaison des performances des différents modèles CNN", use_column_width=True)
                # Option de téléchargement de l'image
                with open(image_path + "/cnn_results_processed_vgg16.png", "rb") as img_file:
                    st.download_button(label="Télécharger les résultats", data=img_file, file_name="cnn_results_processed_vgg16.png", mime="image/png")

                image_results_7 = Image.open(image_path + '/cnn_results_processed_zero.png')
                st.image(image_results_7, caption="Comparaison des performances des différents modèles CNN", use_column_width=True)
                # Option de téléchargement de l'image
                with open(image_path + "/cnn_results_processed_zero.png", "rb") as img_file:
                    st.download_button(label="Télécharger les résultats", data=img_file, file_name="cnn_results_processed_zero.png", mime="image/png")

            with st.expander("5. Résumé des Résultats CNN"):
                st.markdown('''
                            **Les modèles CNN basés sur le transfer learning se sont révélés les plus performants pour la classification des images de documents**, en particulier **VGG16**, qui a atteint une précision de **86%** sur les données test.
                            Bien que cette précision soit légèrement inférieure aux meilleurs travaux sur le dataset RVL-CDIP, elle fournit une base solide pour la suite du projet.
                            Les approches de prétraitement n'ont pas apporté d'amélioration significative et ont même entraîné une dégradation des performances, renforçant l'intérêt de travailler avec les images originales.
                            ''')
                # Données pour le tableau des résultats
                data = {
                    "Modèle": [
                        "CNN sans Transfer Learning",
                        "VGG16 (Transfer Learning)",
                        "CNN sans Transfer Learning",
                        "VGG16 (Transfer Learning)",
                        "ResNet50 (Transfer Learning)"
                        ],
                    "Prétraitement": [
                        "Prétraité",
                        "Prétraité",
                        "Non prétraité",
                        "Non prétraité",
                        "Non prétraité"
                        ],
                    "Accuracy sur Test": [
                        0.34,
                        0.59,
                        0.58,
                        0.86,
                        0.85
                        ]
                    }
                # Création du DataFrame
                df = pd.DataFrame(data)
                # Arrondir les valeurs de 'Accuracy sur Test' à deux décimales
                df['Accuracy sur Test'] = df['Accuracy sur Test'].round(2)
                # Mise en évidence du modèle sélectionné
                highlight = lambda x: ['background-color: yellow' if (x['Modèle'] == "VGG16 (Transfer Learning)" and x['Prétraitement'] == "Non prétraité") else '' for i in x]
                # Affichage du tableau avec surlignage dans Streamlit
                st.subheader("Tableau des Résultats des Modèles CNN")
                st.dataframe(df.style.apply(highlight, axis=1))
                # Option de téléchargement du tableau en CSV
                @st.cache_data
                def convert_df_to_csv(df):
                    return df.to_csv(index=False).encode('utf-8')
                csv_data = convert_df_to_csv(df)
                st.download_button(
                    label="Télécharger le tableau des résultats en CSV",
                    data=csv_data,
                    file_name="resultats_cnn.csv",
                    mime="text/csv"
                    )
                st.markdown(
                    f"""
                    <div class="prediction-box">
                        <h3>
                            🎉 Modèle CNN : avec <span>27 millions</span> de paramètres
                        </h3>
                        <h3>
                            Précision sur le dataset de test: <span>86%</span>
                        </h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # Si le modèle sélectionné est BERT, affichage des informations détaillées
        elif model_choice == "BERT":
            st.subheader("Modèle BERT : Classification de documents par les textes")
            st.image(image_path + "/bert.PNG", caption="Étape finale de classification", use_column_width=True)
            st.markdown('''
            **Pourquoi avons-nous choisi BERT ?**
            - **Compréhension contextuelle bidirectionnelle** : BERT comprend le contexte des mots dans les deux sens, ce qui permet une interprétation plus précise des phrases, essentiel pour la classification de documents.
            - **Pré-entraînement sur de vastes corpus** : Grâce à un pré-entraînement sur des corpus comme Wikipédia, BERT s'adapte facilement à des tâches de classification de documents en capturant efficacement les relations entre les phrases.
            - **Efficacité sur les textes bruités** : BERT fonctionne bien avec des textes issus de l'OCR, sans avoir besoin de traitements supplémentaires comme la suppression des stop words.
            - **Performances supérieures** : BERT surpasse de nombreux modèles pour la classification de documents et saisit des contextes complexes même dans des textes longs.
            - **Robustesse face aux erreurs** : BERT est résistant aux erreurs textuelles, essentiel pour les textes extraits de l'OCR.
            ''')
            st.markdown("**Étapes suivies pour entraîner BERT :**")
            with st.expander("1. Prétraitement des images avec OpenCV"):
                st.code('''
                    image_resized = cv2.resize(image_np, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
                    gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)
                    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                ''', language='python')
                st.write("Redimensionnement des images et conversion en niveaux de gris pour simplifier l'image et améliorer la lisibilité des petits caractères.")
                # Ajout de l'image liée à cette étape
                image_preprocessing = Image.open(image_path + '/bert_pretraitement.PNG')
                st.image(image_preprocessing, caption="Prétraitement des images avec OpenCV", use_column_width=True)
                # Option de téléchargement de l'image
                with open(image_path + "/bert_pretraitement.PNG", "rb") as img_file:
                    st.download_button(label="Télécharger l'image", data=img_file, file_name="pretraitement_opencv.png", mime="image/png")
            with st.expander("2. Application de l'OCR avec EasyOCR"):
                st.code('''
                    result = reader.readtext(image_np, detail=1)
                ''', language='python')
                st.write("Utilisation d'EasyOCR pour extraire le texte à partir des images traitées.")
                # Ajout de l'image liée à cette étape
                image_ocr = Image.open(image_path + '/bert_resultat_ocr.PNG')
                st.image(image_ocr, caption="Application de l'OCR avec EasyOCR", use_column_width=True)
                # Option de téléchargement de l'image
                with open(image_path + "/bert_resultat_ocr.PNG", "rb") as img_file:
                    st.download_button(label="Télécharger l'image", data=img_file, file_name="ocr_easyocr.png", mime="image/png")
            with st.expander("3. Tokenisation avec le tokenizer de BERT"):
                st.code('''
                    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                    tokens = tokenizer(text, max_length=512, truncation=True)
                ''', language='python')
                st.write("Le texte est segmenté en tokens avec un maximum de 512 tokens par séquence.")
            with st.expander("4. Entraînement et évaluation du modèle"):
              st.markdown('''
              **Paramètres d'entraînement de BERT** :
              - Modèle utilisé : `bert-base-uncased` pour la classification avec 16 labels.
              - Taille de batch : 16
              - Nombre d'époques : 3
              - Taux de régularisation (weight_decay) : 0.01
              - Longueur maximale des séquences : 512 tokens
              **Préparation des données** :
              - Utilisation du tokenizer BERT avec padding et troncation pour uniformiser les séquences.
              - Séparation en jeu d'entraînement et de test (80% entraînement, 20% test).
              ''')
              # Ajout des résultats d'un premier échantillonnage
              st.markdown('''
              **Échantillonnage initial** :
              - 24 000 images ont été utilisées pour un premier entraînement de BERT.
              - **Performance sur 5 epochs :**
                - Accuracy : 77.85%
                - Précision : 78.37%
                - Rappel : 77.85%
                - F1-score : 77.99%
              ''')
              # Ajout de l'image liée aux résultats issus de l'échantillon
              image_sample_result = Image.open(image_path + '/bert_echant_5epochs.PNG')
              st.image(image_sample_result, caption="Résultats issus de l'échantillon de 24 000 images", use_column_width=True)
              with open(image_path + "/bert_echant_5epochs.PNG", "rb") as img_file:
                  st.download_button(label="Télécharger l'image", data=img_file, file_name="resultats_echantillon.png", mime="image/png")
              st.markdown('''
              **Entraînement complet sur le dataset** :
              - **Performance après 3 époques** :
                - Accuracy : 83.95%
                - Précision : 84.41%
                - Rappel : 83.95%
                - F1-score : 83.94%
              ''')
              # Ajout des images liées aux résultats de l'entraînement complet
              image_full_result_1 = Image.open(image_path + '/bert_datasetcomplet.PNG')
              st.image(image_full_result_1, caption="Matrice de confusion après traitement complet du dataset (Image 1)", use_column_width=True)
              with open(image_path + "/bert_datasetcomplet.PNG", "rb") as img_file:
                  st.download_button(label="Télécharger l'image", data=img_file, file_name="resultats_dataset_complet_1.png", mime="image/png")
              image_full_result_2 = Image.open(image_path + '/bert_datasetcomplet_scors.PNG')
              st.image(image_full_result_2, caption="Résultats après traitement complet du dataset (Image 2)", use_column_width=True)
              with open(image_path + "/bert_datasetcomplet_scors.PNG", "rb") as img_file:
                  st.download_button(label="Télécharger l'image", data=img_file, file_name="resultats_dataset_complet_2.png", mime="image/png")
              st.markdown(
                  f"""
                  <div class="prediction-box">
                      <h3>
                          🎉 Modèle BERT : avec <span>109 millions</span> de paramètres
                      </h3>
                      <h3>
                          Précision sur le dataset de test: <span>84.41%</span>
                      </h3>
                  </div>
                  """,
                  unsafe_allow_html=True
              )
              
        elif model_choice == "BERT-CNN":
            st.subheader("Modèle BERT-CNN : Classification de documents par les images et les textes")
            st.markdown('''
            **Pourquoi avons-nous choisi un multimodèle BERT-CNN ?**
            Le choix d'une architecture CNN-BERT se justifie par la complémentarité de leurs forces pour traiter des données multimodales. 
            - **CNN (Convolutional Neural Network) :** est particulièrement efficace pour extraire des caractéristiques visuelles à partir des images. 
            - **Tandis que BERT (Bidirectional Encoder Representations from Transformers) :** est excellent pour la compréhension des caractéristiques sémantiques du texte. 
            - **Leur combinaison** permet une meilleure représentation des documents multimodaux, améliorant ainsi la capacité du modèle à comprendre des documents complexes de manière holistique. Nous bénéficions d'une représentation enrichie qui tire parti des informations visuelles et textuelles, améliorant ainsi la précision et la capacité de classification, là où un modèle unique serait moins performant pour intégrer ces deux types de données.
            ''')
            # Section 1 : Architecture du Modèle
            with st.expander("1. Architecture du Modèle"):
                st.markdown("""
                **Composantes du Modèle :**
                - **Modèle CNN** : Extraction des caractéristiques visuelles à partir des images des documents.
                - **Modèle BERT** : Extraction des caractéristiques textuelles à partir du texte extrait des documents.

                **Fusion des Caractéristiques :**
                - Les caractéristiques visuelles et textuelles sont concaténées pour former une représentation combinée du document.
                - Cette représentation est ensuite passée à travers des couches entièrement connectées pour la classification finale.
                """)
                # Ajout de l'image d'architecture du modèle
                st.image(image_path + '/architecture_sans_tuning.png', caption="architecture du Multimodèle CNN-BERT sans tuning", use_column_width=True)
                # Option de téléchargement de l'image
                with open(image_path + "/architecture_sans_tuning.png", "rb") as img_file:
                    st.download_button(label="Télécharger l'image", data=img_file, file_name="architecture_sans_tuning.png", mime="image/png")

            # Section 2 : Préparation des Données
            with st.expander("2. Préparation des Données"):
                st.markdown("""
                **Chargement des Données :**
                """)
                # Ajout de l'image de chargement des données
                st.image(image_path + '/chargement_de_donnees.png', caption="chargement de données du BERT", use_column_width=True)
                # Option de téléchargement de l'image
                with open(image_path + "/chargement_de_donnees.png", "rb") as img_file:
                    st.download_button(label="Télécharger l'image", data=img_file, file_name="chargement_de_donnees.png", mime="image/png")

                st.markdown("""
                **Fusion des Données :**
                - Les datasets du CNN et de BERT sont fusionnés en utilisant l'ID unique des images pour associer correctement les caractéristiques visuelles et textuelles.
                """)
                # Ajout de l'image de fusion des données
                st.image(image_path + '/fusion_de_donnees.png', caption="fusion des données du CNN et OCR", use_column_width=True)
                # Option de téléchargement de l'image
                with open(image_path + "/fusion_de_donnees.png", "rb") as img_file:
                    st.download_button(label="Télécharger l'image", data=img_file, file_name="fusion_de_donnees.png", mime="image/png")

                st.markdown("""
                **Prétraitement :**
                - **Pour le CNN** : Redimensionnement des images, normalisation, conversion en niveaux de gris.
                - **Pour BERT** : Tokenisation du texte, création des `input_ids` et `attention_mask`.
                """)
                # Ajout de l'image de prétraitement des données
                st.image(image_path + '/pretraitement.png', caption="prétraitement des données d'entré du Multimodèle", use_column_width=True)
                # Option de téléchargement de l'image
                with open(image_path + "/pretraitement.png", "rb") as img_file:
                    st.download_button(label="Télécharger l'image", data=img_file, file_name="pretraitement.png", mime="image/png")

            # Section 3 : Construction et Entraînement du Modèle
            with st.expander("3. Construction et Entraînement du Modèle"):
                st.markdown("""
                **Modèle CNN :**
                - Utilisation d'un modèle pré-entraîné avec une couche dense de sortie pour les caractéristiques visuelles.
                - Les poids du CNN sont initialement gelés pour utiliser les caractéristiques pré-apprises.
                """)
                # Ajout de l'image du modèle CNN
                st.image(image_path + '/modele_CNN.png', caption="chargement du CNN pré-entraîné", use_column_width=True)
                # Option de téléchargement de l'image
                with open(image_path + "/modele_CNN.png", "rb") as img_file:
                    st.download_button(label="Télécharger l'image", data=img_file, file_name="modele_CNN.png", mime="image/png")

                st.markdown("""
                **Modèle BERT :**
                - Utilisation de `TFBertForSequenceClassification` pour extraire les logits en tant que caractéristiques textuelles.
                - Les poids de BERT sont également initialement gelés.
                """)
                # Ajout de l'image du modèle BERT
                st.image(image_path + '/modele_bert.png', caption="chargement du BERT pré-entraîné", use_column_width=True)
                # Option de téléchargement de l'image
                with open(image_path + "/modele_bert.png", "rb") as img_file:
                    st.download_button(label="Télécharger l'image", data=img_file, file_name="modele_bert.png", mime="image/png")

                st.markdown("""
                **Couche Personnalisée pour BERT :**
                - Pour faire correspondre notre modèle BERT avec Tensorflow, une couche personnalisée est définie pour utilisé TFBertForSequenceClassification.
                """)
                # Ajout de l'image de Personnalisation de TFBERT
                st.image(image_path + '/Couche_Personnalisee_pour_BERT.png', caption="Personnalisation de TFBERT", use_column_width=True)
                # Option de téléchargement de l'image
                with open(image_path + "/Couche_Personnalisee_pour_BERT.png", "rb") as img_file:
                    st.download_button(label="Télécharger l'image", data=img_file, file_name="Couche_Personnalisee_pour_BERT.png", mime="image/png")

                st.markdown("""
                **Combinaison des Caractéristiques et Construction du Modèle :**
                - Les sorties du CNN et de BERT sont concaténées.
                - Passage par des couches denses avec Dropout pour réduire le surapprentissage.
                """)
                # Ajout de l'image de combinaison des caractéristiques
                st.image(image_path + '/Combinaison_des_Caractéristiques_Construction_du_Modele.png', caption="Construction du Multimodèle.png", use_column_width=True)
                # Option de téléchargement de l'image
                with open(image_path + "/Combinaison_des_Caractéristiques_Construction_du_Modele.png", "rb") as img_file:
                    st.download_button(label="Télécharger l'image", data=img_file, file_name="Combinaison_des_Caractéristiques_Construction_du_Modele.png", mime="image/png")

                st.markdown("""
                **Entraînement du Modèle :**
                - Compilation du modèle avec une fonction de perte adaptée (`sparse_categorical_crossentropy`) et un optimiseur (`Adam`).
                - Utilisation de callbacks comme `EarlyStopping` et `ReduceLROnPlateau` pour optimiser l'entraînement.
                """)
                # Ajout de l'image de l'entraînement du modèle
                st.image(image_path + '/Entrainement_du_Modele.png', caption="Entraînement du Multimodèle", use_column_width=True)
                # Option de téléchargement de l'image
                with open(image_path + "/Entrainement_du_Modele.png", "rb") as img_file:
                    st.download_button(label="Télécharger l'image", data=img_file, file_name="Entrainement_du_Modele.png", mime="image/png")

            # Section 4 : Résultats du Modèle Multimodal
            with st.expander("4. Résultats du Modèle Multimodal"):
                st.markdown("**Métriques Obtenues :**")
                # Ajout de l'image des résultats
                st.image(image_path + '/confusion_matrix_1.png', caption="confusion matrix", use_column_width=True)
                # Option de téléchargement de l'image
                with open(image_path + "/confusion_matrix_1.png", "rb") as img_file:
                    st.download_button(label="Télécharger l'image", data=img_file, file_name="confusion_matrix_1.png", mime="image/png")

                # Ajout de l'image des résultats
                st.image(image_path + '/courbes_precision_multimodele_1.png', caption="courbes de précision du Multimodèle", use_column_width=True)
                # Option de téléchargement de l'image
                with open(image_path + "/courbes_precision_multimodele_1.png", "rb") as img_file:
                    st.download_button(label="Télécharger l'image", data=img_file, file_name="courbes_precision_multimodele_1.png", mime="image/png")

                # Ajout de l'image des résultats
                st.image(image_path + '/Classification_report_1.png', caption="Classification report du Multimodèle", use_column_width=True)
                # Option de téléchargement de l'image
                with open(image_path + "/Classification_report_1.png", "rb") as img_file:
                    st.download_button(label="Télécharger l'image", data=img_file, file_name="Classification_report_1.png", mime="image/png")

                st.markdown("**Observations :**")
                st.markdown("""
                - Le modèle montre une bonne capacité à classer les documents.
                - Cependant, des confusions persistent entre certaines classes similaires, notamment les classes 1, 12, et 13 et les classes 6 et 7.
                """)

            # Section 5 : Améliorations Apportées dans la Seconde Version
            with st.expander("5. Améliorations Apportées dans la Seconde Version"):
                st.markdown("""
                **Limitations de la Première Version :**
                """)
                st.markdown("""
                - **Confusion des Classes Similaires** : Difficulté à distinguer les classes avec des caractéristiques proches.
                """)
                st.markdown("""
                - **Absence de Fine-tuning** : Les modèles CNN et BERT étant gelés, le modèle ne peut pas s'adapter aux spécificités du dataset.
                """)

                st.markdown("""
                **Changements Apportés :**
                """)

                # Sous-partie : Regroupement des Classes
                st.markdown("""
                - **Regroupement des Classes** : Regroupement des classes présentant des taux élevés de confusion pour simplifier la tâche de classification.
                Par exemple, les classes 1, 12 et 13 sont regroupées en une seule classe.
                """)
                # Ajout de l'image associée
                st.image(image_path + '/Regroupement_des_Classes.png', caption="Regroupement des classes", use_column_width=True)
                # Option de téléchargement de l'image
                with open(image_path + "/Regroupement_des_Classes.png", "rb") as img_file:
                    st.download_button(label="Télécharger l'image", data=img_file, file_name="Regroupement_des_Classes.png", mime="image/png")

                # Sous-partie : Fine-tuning des Modèles
                st.markdown("""
                - **Fine-tuning des Modèles** : Dégel des dernières couches du CNN et de BERT pour permettre un apprentissage plus adapté aux données spécifiques.
                Cela permet au modèle d'apprendre des caractéristiques plus pertinentes pour notre dataset.
                """)
                # Ajout de l'image associée
                st.image(image_path + '/Fine_tuning_des_Modeles.png', caption="Tuning du Multimodèle", use_column_width=True)
                # Option de téléchargement de l'image
                with open(image_path + "/Fine_tuning_des_Modeles.png", "rb") as img_file:
                    st.download_button(label="Télécharger l'image", data=img_file, file_name="Fine_tuning_des_Modeles.png", mime="image/png")

                # Sous-partie : Augmentation du Taux de Dropout
                st.markdown("""
                - **Augmentation du Taux de Dropout** : Le taux de Dropout est augmenté de 0.5 à 0.6 pour améliorer la généralisation du modèle.
                """)
                # Ajout de l'image associée
                st.image(image_path + '/Augmentation_Taux_Dropout.png', caption="Variation du Dropout", use_column_width=True)
                # Option de téléchargement de l'image
                with open(image_path + "/Augmentation_Taux_Dropout.png", "rb") as img_file:
                    st.download_button(label="Télécharger l'image", data=img_file, file_name="Augmentation_Taux_Dropout.png", mime="image/png")

                # Sous-partie : Architecture du Multimodèle avec tuning
                st.markdown("""
                - **Architecture du Multimodèle avec tuning** : On peut observer la nouvèle architecture du multimodèle après le tuning.
                """)
                # Ajout de l'image de l'architecture
                st.image(image_path + '/architecture_avec_tuning.png', caption="Nouvelle architecture du Multimodèle", use_column_width=True)
                # Option de téléchargement de l'image
                with open(image_path + "/architecture_avec_tuning.png", "rb") as img_file:
                    st.download_button(label="Télécharger l'image", data=img_file, file_name="architecture_avec_tuning.png", mime="image/png")

                st.markdown("**Métriques Obtenues :**")
                # Ajout de l'image des résultats
                st.image(image_path + '/confusion_matrix_2.png', caption="confusion matrix", use_column_width=True)
                # Option de téléchargement de l'image
                with open(image_path + "/confusion_matrix_2.png", "rb") as img_file:
                    st.download_button(label="Télécharger l'image", data=img_file, file_name="confusion_matrix_2.png", mime="image/png")

                # Ajout de l'image des résultats
                st.image(image_path + '/courbes_précision_multimodèles_2.png', caption="courbes de précision du Multimodèle avec tuning", use_column_width=True)
                # Option de téléchargement de l'image
                with open(image_path + "/courbes_précision_multimodèles_2.png", "rb") as img_file:
                    st.download_button(label="Télécharger l'image", data=img_file, file_name="courbes_précision_multimodèles_2.png", mime="image/png")

                # Ajout de l'image des résultats
                st.image(image_path + '/Classification_report_2.png', caption="Classification report du Multimodèle avec tuning", use_column_width=True)
                # Option de téléchargement de l'image
                with open(image_path + "/Classification_report_2.png", "rb") as img_file:
                    st.download_button(label="Télécharger l'image", data=img_file, file_name="Classification_report_2.png.png", mime="image/png")

            # Section 6 : Comparaison des Performances
            with st.expander("6. Comparaison des Performances"):
                st.markdown("**Analyse Comparée**")
                st.table({
                    "Aspect": ["Validation Accuracy", "Test Accuracy", "Validation Loss", "Test Loss", "Confusion des Classes", "Fine-tuning", "Taux de Dropout"],
                    "Première Version": [0.8765, 0.8753, 0.4238, 0.4338, "Élevée entre les classes 1, 12, 13", "Non", 0.5],
                    "Seconde Version": [0.8968, 0.8932, 0.4337, 0.4496, "Réduite grâce au regroupement", "Oui", 0.6]
                })

                st.markdown(
                    f"""
                    <div class="prediction-box">
                        <h3>
                            🎉 Multimodèle BERT-CNN : avec <span>28 millions</span> de paramètres
                        </h3>
                        <h3>
                            Précision sur le dataset de test: <span>89.68%</span>
                        </h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    #---------------------------------------------------------------------------
    # Prédiction par nos modèles
    #---------------------------------------------------------------------------
    elif page == 'Prédiction par nos modèles':
        st.image(image_path + '/Documancer.gif', use_column_width=True)
        st.title('Prédiction par nos modèles')
        st.markdown('''
        Les prédictions ont été réalisées à l'aide de plusieurs modèles décrits dans la stratégie déployée : CNN, BERT, et l'approche multimodale CNN-BERT.
        Vous pouvez visualiser ici les prédictions effectuées par chaque modèle sur un échantillon de documents.
        ''')

        # Uploader un fichier image
        uploaded_file = st.file_uploader("Choisissez une image", type=['jpg', 'jpeg', 'png', 'tif', 'tiff', 'webp'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Image Importée', use_column_width=True)

            # Chemins des modèles
            model_paths = {
                'CNN': model_path_cnn,
                'BERT': drive_model_BERT_dir,
                'CNN-BERT': model_path_multimodal
            }


            # Ajout d'une option pour activer/désactiver l'analyse LIME
            st.subheader("Options d'analyse")
            lime_enabled = st.checkbox("Effectuer l'analyse d'interprétabilité avec LIME", value=False)

            # Choisir un modèle
            model_options = ['Choisissez le modèle !', 'CNN', 'BERT', 'CNN-BERT']
            model_choice = st.selectbox('Choisissez un modèle à utiliser pour la classification', model_options, index=0)
            # Vérifier que l'utilisateur a sélectionné un modèle
            if model_choice != 'Choisissez le modèle !':
                model_path = model_paths[model_choice]
                if not os.path.exists(model_path):
                    st.error("Le modèle n'a pas été trouvé. Veuillez vérifier le chemin du modèle.")
                else:
                    # Charger le modèle choisi
                    if model_choice == 'CNN-BERT':
                        model_CNNBERT = load_model_multi(model_path)
                    elif model_choice == 'CNN':
                        model_CNN = load_model_CNN(model_path)
                    elif model_choice == 'BERT':
                        model_BERT = load_model_BERT(model_path)

                    # Extraction de texte si BERT ou CNN-BERT est choisi
                    if model_choice in ['CNN-BERT', 'BERT']:
                        st.subheader('Texte extrait via EasyOCR')
                        reader = easyocr.Reader(['en'])
                        extracted_text = reader.readtext(np.array(image), detail=0)
                        extracted_text = ' '.join(extracted_text)
                        st.text_area('Texte extrait via EasyOCR', extracted_text, height=200)

                        # Tokenisation avec BERT
                        st.subheader('Tokenisation du texte avec BERT')
                        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                        tokens = tokenizer(
                            extracted_text,
                            padding='max_length',
                            truncation=True,
                            max_length=512,
                            return_tensors='tf'
                        )
                        input_ids = tokens['input_ids']
                        attention_mask = tokens['attention_mask']
                        # st.text(f"Tokens: {tokens['input_ids']}")

                        # Récupérer les tokens sous forme de liste
                        token_list = tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])

                        # Filtrer les tokens pour ne garder que ceux qui nes token != '[PAD]']
                        filtered_tokens = [token for token in token_list if token != '[PAD]']

                        # Construire une chaîne de caractères HTML pour chaque token avec un style badge
                        token_html = ""
                        for token in filtered_tokens:
                            # Ajouter des couleurs alternées pour plus de lisibilité
                            color = "#2ED4DA" if token.startswith("##") else "#382DD5"
                            token_html += f'<span style="background-color: {color}; padding: 4px 8px; margin: 2px; border-radius: 5px; display: inline-block;">{token}</span> '

                        # Afficher le HTML avec Streamlit
                        st.markdown("#### Tokens générés :")
                        st.markdown(token_html, unsafe_allow_html=True)

                    # Préparer l'image pour le modèle si CNN ou CNN-BERT est choisi
                    if model_choice in ['CNN-BERT', 'CNN']:
                        image = image.convert('L')  # Convertir en niveaux de gris
                        image = image.resize((224, 224))  # Redimensionner l'image à la taille requise
                        image_tensor = np.expand_dims(np.array(image), axis=-1)  # Ajouter une dimension pour (224, 224, 1)
                        image_tensor = np.expand_dims(image_tensor, axis=0)  # Ajouter une autre dimension pour batch (1, 224, 224, 1)
                        image_tensor = tf.convert_to_tensor(image_tensor, dtype=tf.float32)

                        # Visualisation de l'image transformée
                        st.subheader('Image transformée')
                        fig, ax = plt.subplots()
                        ax.imshow(image_tensor.numpy().squeeze(), cmap='gray')
                        ax.axis('off')
                        st.pyplot(fig)

                    # Classification de l'image ou du texte
                    st.subheader("Classification de l'image")

                    if model_choice == 'CNN-BERT':
                        predicted_class, confidence_score = classify_image_text_keras(model_CNNBERT, image_tensor, input_ids, attention_mask)
                    elif model_choice == 'CNN':
                        predicted_class, confidence_score = classify_image_keras(model_CNN, image_tensor)
                    elif model_choice == 'BERT':
                        predicted_class, confidence_score = classify_text_keras(model_BERT, input_ids, attention_mask)

                    # Affichage du résultat de la classification
                    class_labels = [
                        'Class 0: Letter', 'Class 1: Form, Presentation, Questionnaire', 'Class 2: Email', 'Class 3: Handwritten',
                        'Class 4: Advertisement', 'Class 5: Scientific Report & Publication',
                        'Class 6: Specification', 'Class 7: File Folder', 'Class 8: News Article',
                        'Class 9: Budget', 'Class 10: Invoice', 'Class 11: Resume', 'Class 12: Memo'
                    ]
                    # st.write(f'Catégorie Prédite : {class_labels[predicted_class]} avec une précision de {confidence_score:.2f}%')
                    # Affichage de la catégorie prédite avec un style visuel attrayant
                    st.markdown(
                        f"""
                        <div class="prediction-box">
                            <h3>
                                🎉 Catégorie Prédite : <span>{class_labels[predicted_class]}</span>
                            </h3>
                            <p>
                                Précision : <span>{confidence_score:.2f}%</span>
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Analyse avec LIME
                    if lime_enabled:
                      st.subheader("Analyse de l'interprétabilité avec LIME")
                      if model_choice in ['CNN-BERT', 'CNN']:
                          explainer = lime.lime_image.LimeImageExplainer()

                          # Fonction de prédiction à fournir à LIME
                          def predict_fn(images):
                              images = np.expand_dims(images[:, :, :, 0], axis=-1)  # S'assurer que les images ont la bonne dimension
                              images = tf.convert_to_tensor(images, dtype=tf.float32)
                              if model_choice == 'CNN-BERT':
                                  batch_size = images.shape[0]
                                  input_ids_placeholder = tf.repeat(input_ids, batch_size, axis=0)  # Répliquer les tokens pour chaque image
                                  attention_mask_placeholder = tf.repeat(attention_mask, batch_size, axis=0)  # Répliquer les masques
                                  # Prédire avec le modèle CNN-BERT en fournissant les images et les entrées textuelles
                                  return model_CNNBERT([ input_ids_placeholder, images, attention_mask_placeholder]).numpy()
                              elif model_choice == 'CNN':
                                  return model_CNN(images).numpy()

                          explanation = explainer.explain_instance(
                              image_tensor.numpy().squeeze(),
                              predict_fn,
                              top_labels=1,
                              hide_color=0,
                              num_samples=100
                          )

                          # Visualiser l'explication LIME
                          temp, mask = explanation.get_image_and_mask(
                              label=predicted_class,
                              positive_only=False,
                              num_features=10,
                              hide_rest=False
                          )
                          lime_image = mark_boundaries(temp, mask)
                          lime_image = lime_image.astype(np.float32) / 255.0
                          st.image(lime_image, caption='Explication LIME', use_column_width=True)

                          # Ajouter une description ade l'explication
                          st.markdown("""
                          L'explication ci-dessus montre les parties de l'image qui ont le plus influencé la classification.
                          Les zones mises en évidence indiquent les régions qui ont contribué positivement ou négativement à la prédiction.
                          """)
                      if model_choice in ['BERT']:
                        explainer = lime.lime_text.LimeTextExplainer(class_names=class_labels)
                        explanation = explainer.explain_instance(
                            extracted_text,
                            lambda x: model_BERT.predict(tokenizer(x, padding='max_length', truncation=True, max_length=512, return_tensors='tf'))['logits'] if isinstance(model_BERT.predict(tokenizer(x, padding='max_length', truncation=True, max_length=512, return_tensors='tf')), dict) else model_BERT.predict(tokenizer(x, padding='max_length', truncation=True, max_length=512, return_tensors='tf')),
                            num_samples=100,
                            num_features=10
                        )
                        # Extraire l'HTML de l'explication LIME
                        raw_html = explanation.as_html()

                        # Personnalisation du HTML pour éviter les chevauchements et améliorer la lisibilité
                        # Modifier les styles pour augmenter la lisibilité
                        modified_html = raw_html
                        modified_html = re.sub(r"font-size:\s*\d+px;", "font-size: 12px;", modified_html)  # Réduire la taille de la police si nécessaire
                        modified_html = re.sub(r"width:\s*\d+px;", "width: auto;", modified_html)  # Ajuster la largeur automatiquement pour éviter le débordement
                        modified_html = re.sub(r"color:\s*[^;]+;", "color: black;", modified_html)  # Changer la couleur de la police en blanc
                        modified_html = re.sub(r"background-color:\s*[^;]+;", "background-color: #382DD5;", modified_html)  # Changer la couleur de fond pour correspondre à la police
                        modified_html = re.sub(r"max-width:\s*\d+px;", "max-width: 100%;", modified_html)  # Augmenter la largeur maximale des conteneurs
                        # Insérer une balise <style> globale au début de l'HTML pour changer la couleur par défaut de la police
                        style_tag = """
                        <style>
                            body {
                                color: #382DD5 !important;
                                background-color: #2ED4DA !important;
                            }
                        </style>
                        """
                        modified_html = style_tag + modified_html

                        # Utiliser st.components.v1.html pour afficher le HTML modifié
                        st.markdown("### Explication LIME du texte")
                        st.components.v1.html(modified_html, height=600)

                        # Message d'explication
                        st.markdown("""
                        L'explication ci-dessus montre les mots du texte qui ont le plus influencé la classification.
                        Les mots avec des contributions positives ou négatives sont affichés avec leur poids.
                        """)

                      if model_choice in ['CNN-BERT']:

                        def generate_dataset(texts, image):
                          # Préparer l'image
                          if isinstance(image, Image.Image):
                              image = np.array(image)
                          if len(image.shape) == 2:  # Si l'image est en niveaux de gris
                              image = np.expand_dims(image, axis=-1)
                          image = tf.convert_to_tensor(image, dtype=tf.float16)
                          image = tf.expand_dims(image, axis=0)  # Ajouter la dimension du lot

                          # Créer un Dataset TensorFlow pour charger les textes un par un
                          dataset = tf.data.Dataset.from_tensor_slices(texts)
                          dataset = dataset.map(lambda text: (
                              tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='tf')
                          ))
                          dataset = dataset.batch(1)
                          results = []
                          for inputs in dataset:
                              input_ids = inputs['input_ids']
                              attention_mask = inputs['attention_mask']
                              input_ids = tf.cast(input_ids, dtype=tf.float16)
                              attention_mask = tf.cast(attention_mask, dtype=tf.float16)
                              result = model_CNNBERT([input_ids, image, attention_mask]).numpy()
                              results.append(result)
                          return np.array(results)

                        def predict_text_batch(image, texts, batch_size=16):
                            # Convertir l'image PIL en tableau NumPy si nécessaire
                            if isinstance(image, Image.Image):
                                image = np.array(image)

                            # Ajouter la dimension des canaux si l'image est en niveaux de gris
                            if len(image.shape) == 2:  # Image en niveaux de gris sans dimension des canaux
                                image = np.expand_dims(image, axis=-1)

                            # Ajouter la dimension de lot pour l'image
                            image = np.expand_dims(image, axis=0)

                            # Convertir l'image en tenseur float16
                            image = tf.convert_to_tensor(image, dtype=tf.float16)

                            # Diviser les textes en batches
                            results = []
                            for i in range(0, len(texts), batch_size):
                                batch_texts = texts[i:i + batch_size]
                                # Tokenizer les textes avec une longueur maximale de 256 (ou ajuster selon la mémoire disponible)
                                inputs = tokenizer(
                                    batch_texts, padding='max_length', truncation=True, max_length=512, return_tensors='tf'
                                )
                                input_ids = inputs['input_ids']
                                attention_mask = inputs['attention_mask']
                                # Convertir les entrées textuelles en float16 pour réduire la consommation de mémoire
                                input_ids = tf.cast(input_ids, dtype=tf.float16)
                                attention_mask = tf.cast(attention_mask, dtype=tf.float16)
                                # Répliquer l'image pour correspondre à la taille du batch
                                tiled_image = tf.tile(image, [len(batch_texts), 1, 1, 1])
                                # Passer l'image et les autres entrées au modèle
                                batch_results = model_CNNBERT([input_ids, tiled_image, attention_mask]).numpy()
                                results.extend(batch_results)
                            return np.array(results)

                        explainer = lime.lime_text.LimeTextExplainer(class_names=class_labels)
                        explanation = explainer.explain_instance(
                            extracted_text,
                            classifier_fn=lambda texts: predict_text_batch(image, texts, batch_size=16),
                            num_samples=100,  # Réduire ou augmenter le nombre d'échantillons en fonction de la performance mémoire
                            num_features=10
                        )
                        # Extraire l'HTML de l'explication LIME
                        raw_html = explanation.as_html()

                        # Personnalisation du HTML pour éviter les chevauchements et améliorer la lisibilité
                        # Modifier les styles pour augmenter la lisibilité
                        modified_html = raw_html
                        modified_html = re.sub(r"font-size:\s*\d+px;", "font-size: 12px;", modified_html)  # Réduire la taille de la police si nécessaire
                        modified_html = re.sub(r"width:\s*\d+px;", "width: auto;", modified_html)  # Ajuster la largeur automatiquement pour éviter le débordement
                        modified_html = re.sub(r"color:\s*[^;]+;", "color: black;", modified_html)  # Changer la couleur de la police en blanc
                        modified_html = re.sub(r"background-color:\s*[^;]+;", "background-color: white;", modified_html)  # Changer la couleur de fond pour correspondre à la police
                        modified_html = re.sub(r"max-width:\s*\d+px;", "max-width: 100%;", modified_html)  # Augmenter la largeur maximale des conteneurs
                        # Insérer une balise <style> globale au début de l'HTML pour changer la couleur par défaut de la police
                        style_tag = """
                        <style>
                            body {
                                color: #382DD5 !important;
                                background-color: #2ED4DA !important;
                            }
                        </style>
                        """
                        modified_html = style_tag + modified_html

                        # Utiliser st.components.v1.html pour afficher le HTML modifié
                        st.markdown("### Explication LIME du texte")
                        st.components.v1.html(modified_html, height=600)

                        # Message d'explication
                        st.markdown("""
                        L'explication ci-dessus montre les mots du texte qui ont le plus influencé la classification.
                        Les mots avec des contributions positives ou négatives sont affichés avec leur poids.
                        """)
    #---------------------------------------------------------------------------
    # Conclusion & perspectives
    #---------------------------------------------------------------------------
    elif page == 'Conclusion & perspectives':
        st.title('Conclusion & perspectives')
        st.markdown('''
        Le projet a démontré l'efficacité des modèles CNN et BERT pour la classification de documents.
        L'approche multimodale a surpassé les performances des modèles individuels, atteignant une précision de **89%**, avec seulement  28,219,755 de paramètres !.

        Perspectives d'amélioration :
        - **Augmentation des données** : Utiliser des techniques d'augmentation pour enrichir le dataset.
        - **Optimisation du prétraitement** : Améliorer le prétraitement des images et l'OCR.
        - **Optimisation des hyperparamètres** : Ajuster les hyperparamètres pour optimiser les performances.
        ''')
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)
        image_sample_result = Image.open(image_path + '/Real_DocuMancer_parameters.png')
        st.image(image_sample_result, caption="Résultats mondiaux sur RVL-CDIP", use_column_width=True)
        with open(image_path + "/Real_DocuMancer_parameters.png", "rb") as img_file:
            st.download_button(label="Télécharger l'image", data=img_file, file_name="Real_DocuMancer_parameters.png", mime="image/png")

#-------------------------------------------------------------------------------
# Lancement
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
