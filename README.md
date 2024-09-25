jul24_bds_classif_docs
modélisation de catégorisation sur DST_RVL_CDIP
==============================

installation du projet
------------

python3 -m venv env_RVL_CDIP

source env_RVL_CDIP/bin/activate

pip install -r requirements.txt

Description des imports
------------
pandas numpy

scikit-learn

matplotlib

seaborn

opencv-python

tesseract-ocr


Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Should be in your computer but not on Github (only in .gitignore)
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's name, and a short `-` delimited description, e.g.
    │                         `1.0-alban-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, links, and all other explanatory materials.
    │
    ├── reports            <- The reports that you'll make during this project as PDF
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize.py

Description des datas
------------
lien des datas : https://adamharley.com/rvl-cdip/

projets existants sur ces datas : https://paperswithcode.com/sota/document-image-classification-on-rvl-cdip

![RVL_CDIP](https://github.com/user-attachments/assets/c6b260cf-418d-4f9d-9ba8-ffac4b8f37b4)

________________

RVL-CDIP Dataset
________________

The RVL-CDIP (Ryerson Vision Lab Complex Document Information Processing) dataset consists of 400,000 grayscale images in 16 classes, with 25,000 images per class. There are 320,000 training images, 40,000 validation images, and 40,000 test images. The images are sized so their largest dimension does not exceed 1000 pixels.

For questions and comments please contact Adam Harley (aharley@scs.ryerson.ca).

_________

CHANGELOG
_________

05/JUN/2015	First version of the dataset

_______

DETAILS
_______

The label files list the images and their categories in the following format:

path/to/the/image.tif category

where the categories are numbered 0 to 15, in the following order:

0 letter

1 form

2 email

3 handwritten

4 advertisement

5 scientific report

6 scientific publication

7 specification

8 file folder

9 news article

10 budget

11 invoice

12 presentation

13 questionnaire

14 resume

15 memo

________

CITATION
________

If you use this dataset, please cite:

A. W. Harley, A. Ufkes, K. G. Derpanis, "Evaluation of Deep Convolutional Nets for Document Image Classification and Retrieval," in ICDAR, 2015

Bibtex format:

@inproceedings{harley2015icdar,
    title = {Evaluation of Deep Convolutional Nets for Document Image Classification and Retrieval},
    author = {Adam W Harley and Alex Ufkes and Konstantinos G Derpanis},
    booktitle = {International Conference on Document Analysis and Recognition ({ICDAR})}},
    year = {2015}
}

___________________

FURTHER INFORMATION
___________________

This dataset is a subset of the IIT-CDIP Test Collection 1.0 [1]. The file structure of this dataset is the same as in the IIT collection, so it is possible to refer to that dataset for OCR and additional metadata. The IIT-CDIP dataset is itself a subset of the Legacy Tobacco Document Library [2].

[1] D. Lewis, G. Agam, S. Argamon, O. Frieder, D. Grossman, and J. Heard, "Building a test collection for complex document information processing," in Proc. 29th Annual Int. ACM SIGIR Conference (SIGIR 2006), pp. 665-666, 2006
[2] The Legacy Tobacco Document Library (LTDL), University of California, San Francisco, 2007. http://legacy.library.ucsf.edu/.

More information about this dataset can be obtained at the following URL: http://scs.ryerson.ca/~aharley/rvl-cdip/


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

___________________

STRATEGY INFORMATION
___________________

Introduction
Dans le cadre de notre projet de classification de documents sur le dataset RVL-CDIP, nous avons conçu un modèle multimodal combinant deux architectures puissantes : un réseau de neurones convolutionnel (CNN) pour traiter les images, et un modèle BERT (Bidirectional Encoder Representations from Transformers) pour la classification des séquences textuelles. L'objectif de ce modèle est d'exploiter simultanément les informations visuelles et textuelles présentes dans les documents afin d'améliorer les performances de classification. Cette approche est particulièrement pertinente pour des données riches en contenu, comme les documents numérisés, qui combinent souvent des images (mises en page, diagrammes, etc.) et du texte.
1. Motivation pour un modèle multimodal
Dans les tâches de classification de documents, les images et le texte fournissent des indices complémentaires. Les approches classiques de classification basées uniquement sur les images ou uniquement sur le texte peuvent ne pas capturer toute la complexité du document. En combinant un modèle CNN pour analyser la structure visuelle des documents et un modèle BERT pour extraire les caractéristiques textuelles, nous exploitons les deux sources d'information de manière synergique.
Le CNN est capable de capter des motifs visuels (comme la disposition des textes, les formes des caractères, les graphiques, etc.), tandis que BERT excelle dans la compréhension sémantique du texte. Cette complémentarité rend notre modèle plus robuste pour des tâches de classification complexes telles que celles rencontrées avec le dataset RVL-CDIP.
2. Description du modèle
Notre approche repose sur l'assemblage de deux sous-modèles indépendants : un CNN et un modèle BERT, tous deux pré-entraînés.
a. Modèle CNN
Nous avons d'abord chargé un modèle CNN déjà pré-entraîné sur des images de documents. Le CNN est utilisé pour extraire des caractéristiques visuelles des images des documents. Une couche dense en sortie du CNN génère des vecteurs de caractéristiques, que nous avons retenus pour capturer des informations visuelles essentielles à la classification.
b. Modèle BERT
Nous avons ensuite utilisé un modèle BERT pré-entraîné pour classifier les séquences textuelles présentes dans les documents. La structure de BERT, basée sur des Transformers, nous permet de capturer le contexte et les relations complexes entre les mots, ce qui est crucial pour des documents dont le texte peut être ambigu ou varier en longueur. En gelant les poids du modèle BERT, nous nous sommes assurés que seules les couches finales de classification seraient entraînées, tout en conservant la richesse des représentations textuelles générées par BERT.
c. Extraction et combinaison des caractéristiques
Une fois les deux sous-modèles chargés, nous avons extrait les caractéristiques des images via le CNN et les logits (scores bruts avant la couche de classification) du modèle BERT pour le texte. Ces deux ensembles de caractéristiques, visuelles et textuelles, ont été combinés grâce à une couche de concaténation. Cela nous a permis de fusionner efficacement les informations issues des deux modalités (texte et image).
La combinaison des caractéristiques permet de capturer à la fois l’information visuelle de la mise en page et le contenu textuel des documents, renforçant ainsi la capacité de notre modèle à différencier les classes.


___________________

FINAL RESULTS
___________________

![image](https://github.com/user-attachments/assets/c216a6cb-329e-423f-9b5b-2f14654c5470)
![image](https://github.com/user-attachments/assets/bb5cfeb3-2420-4efe-8ea7-43fef9d070e8)
![image](https://github.com/user-attachments/assets/1222bae6-802e-4f04-8743-6479e2f59982)

Notre projet visait à créer un modèle multimodal combinant un CNN pour les images et un modèle BERT pour les textes afin de classifier les documents du dataset RVL-CDIP. Après l’entraînement et l’évaluation du modèle, nous avons obtenu des résultats globalement très satisfaisants, mais quelques points de réflexion et d’amélioration ont émergé.

1. Performance globale du modèle
Les résultats obtenus, avec une précision de 87,53% et un F1-score global similaire, montrent que notre modèle multimodal est performant pour la classification de documents complexes combinant texte et images. La complémentarité entre le traitement des images par CNN et l’analyse textuelle par BERT a permis de capturer des caractéristiques essentielles dans les documents.

2. Courbes ROC et AUC
Les courbes ROC et les scores AUC montrent une performance élevée pour la majorité des classes, avec des AUC proches de 1.0 pour de nombreuses classes, telles que Email, Manuscrit, Dossier, et Résumé. Ces résultats confirment la capacité du modèle à distinguer avec précision les différentes catégories de documents, en particulier celles présentant des caractéristiques visuelles ou textuelles bien distinctes.
Cependant, certaines classes, telles que Formulaire, Présentation, et Rapport scientifique, montrent des AUC légèrement inférieurs, autour de 0.98. Cela suggère une confusion potentielle dans des catégories aux structures similaires, comme nous l’avons vu dans la matrice de confusion.

3. Analyse de la matrice de confusion
La matrice de confusion met en évidence les forces du modèle, avec une majorité de prédictions correctes pour chaque classe, notamment pour les catégories Email, Dossier, et Résumé. Cependant, certaines confusions apparaissent entre des catégories visuellement ou textuellement proches, telles que :
Formulaires et Questionnaires
Rapports scientifiques et Publications scientifiques
Formulaires et Présentations
Ces confusions peuvent être attribuées à des similarités dans la mise en page ou le format textuel, des éléments que le modèle a eu du mal à différencier.

4. Évolution de la précision et de la perte
Les graphiques de précision et de perte montrent que le modèle a bien appris lors des premières époques, avec une précision en hausse rapide et une perte en diminution continue. Cependant, dès la 5ᵉ époque, un plateau s’est installé sur la précision et la perte de validation, tandis que la précision d’entraînement continuait de s’améliorer. Cela indique un surapprentissage modéré, où le modèle commence à s’adapter trop aux données d’entraînement, au détriment de sa généralisation sur les données de validation.

5. Pistes d'amélioration
Bien que les performances globales soient excellentes, nous identifions plusieurs pistes pour améliorer le modèle :
Régularisation et Dropout : Augmenter la régularisation ou le taux de Dropout pourrait aider à limiter le surapprentissage observé à partir de la 5ᵉ époque.
Fine-tuning des sous-modèles : Débloquer certaines couches de BERT ou du CNN pour un fine-tuning plus fin permettrait d’affiner l’apprentissage des représentations pour les classes présentant des confusions.
Data augmentation : L’introduction de techniques d’augmentation des données (variations des images et des textes) pourrait rendre le modèle plus robuste face aux variations des documents réels.



