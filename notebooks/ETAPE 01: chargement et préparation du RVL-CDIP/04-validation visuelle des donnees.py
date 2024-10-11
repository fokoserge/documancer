import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor

# Définition des chemins des répertoires
labels_dir = "/home/serge/project_DST/env_dstproj/data-rvl_cdip/labels"
images_dir = "/home/serge/project_DST/env_dstproj/data-rvl_cdip/images"
output_dir = "/home/serge/project_DST/env_dstproj/data-rvl_cdip/output_images"
data_dir = "/home/serge/project_DST/env_dstproj/data-rvl_cdip"  # Dossier pour enregistrer les DataFrames

# Dictionnaire des catégories de labels
labels_dict = {
    "0": "letter",
    "1": "form",
    "2": "email",
    "3": "handwritten",
    "4": "advertisement",
    "5": "scientific report",
    "6": "scientific publication",
    "7": "specification",
    "8": "file folder",
    "9": "news article",
    "10": "budget",
    "11": "invoice",
    "12": "presentation",
    "13": "questionnaire",
    "14": "resume",
    "15": "memo"
}

# Transformer les clés de labels_dict en entiers
labels_dict = {int(k): v for k, v in labels_dict.items()}

# Fonction pour lire les DataFrames des fichiers de labels
def read_labels_files():
    train_file = os.path.join(labels_dir, "train.txt")
    val_file = os.path.join(labels_dir, "val.txt")
    test_file = os.path.join(labels_dir, "test.txt")

    train_df = pd.read_csv(train_file, sep='\s+', header=None, names=['image_path', 'label'])
    val_df = pd.read_csv(val_file, sep='\s+', header=None, names=['image_path', 'label'])
    test_df = pd.read_csv(test_file, sep='\s+', header=None, names=['image_path', 'label'])
    
    return train_df, val_df, test_df


# Fonction pour échantillonner les images
def sample_images(df, sample_size, processed_df=None):
    # Si processed_df est fourni, retirer les images déjà traitées
    if processed_df is not None:
        df = df[~df['image_path'].isin(processed_df['image_path']) | (processed_df['verified'] == False)]
    sampled_dfs = []
    grouped = df.groupby('label')
    
    for label, group in grouped:
        # Échantillonner uniquement des images non traitées (sample_size par classe)
        sampled_group = group.sample(n=min(len(group), sample_size))
        sampled_dfs.append(sampled_group)
    
    sampled_df = pd.concat(sampled_dfs).reset_index(drop=True)
    
    return sampled_df

# Fonction pour vérifier les images (sans affichage)
def verify_image(row, source_images_dir):
    image_path = row['image_path']
    label = str(row['label'])
    full_image_path = os.path.join(source_images_dir, image_path)
    
    if not os.path.exists(full_image_path):
        print(f"Warning: Image {full_image_path} not found.")
        return None
    
    return {
        'image_path': image_path,
        'label': label
    }

incorrect_labels = []


# Fonction pour afficher les images en utilisant le main thread
def display_images(df, source_images_dir, dataset_type):
    # Liste pour stocker les labels incorrects durant cette exécution
    global incorrect_labels  # Déclaration de la variable globale ici
    
    
    # Charger les labels incorrects déjà sauvegardés
    incorrect_labels_path = os.path.join(data_dir, 'incorrect_labels.csv')
    if os.path.exists(incorrect_labels_path):
        previous_incorrect_labels = pd.read_csv(incorrect_labels_path)
        if 'dataset_type' in previous_incorrect_labels.columns:
            incorrect_labels = previous_incorrect_labels.to_dict('records')
           
    print(f"Displaying images from DataFrame with {len(df)} entries...")
    print("Colonnes disponibles dans le DataFrame:", df.columns)
    
    # Assurer que la colonne 'verified' existe, sinon l'initialiser
    if 'verified' not in df.columns:
        df['verified'] = False  # Si la colonne n'existe pas, on l'ajoute et on initialise à False
    
    # Filtrer uniquement les images qui n'ont pas encore été vérifiées
    df_to_verify = df[df['verified'] == False]
    print(f"Number of images to verify: {len(df_to_verify)}")
    
    if df_to_verify.empty:
        print(f"No images to verify in {dataset_type} dataset.")
        return
    
    for _, row in df_to_verify.iterrows():
        image_path = row['image_path']
        label = int(row['label'])
        full_image_path = os.path.join(source_images_dir, image_path)
        
        if not os.path.exists(full_image_path):
            print(f"Warning: Image {full_image_path} not found.")
            continue
        
        # Lire l'image en niveaux de gris
        img = Image.open(full_image_path).convert('L')
        img_array = np.array(img)
        
        plt.figure(figsize=(32, 16))  # Taille de la figure
        plt.imshow(img_array, cmap='gray')
        plt.title(f"Label: {labels_dict.get(label, 'unknown')}")   # Utilisation directe du label entier
        plt.axis('off')
        plt.show(block=False)
        
         #validation manuelle et visuelle de l'image et du labels      
        response = input(f"Is the label '{labels_dict.get(label, 'unknown')}' correct for this image? (y/n): ").strip().lower()
       
        plt.close()
        
        #récupération de la réponse sur la validation visuelle du label
        if response == 'n':
            incorrect_labels.append({
                'image_path': image_path,
                'label': label,
                'dataset_type': dataset_type  # Ajouter le type de dataset
            })

        # Mettre à jour le statut de vérification dans le DataFrame
        df.loc[df['image_path'] == image_path, 'verified'] = True  
        
        # Sauvegarder immédiatement incorrect_labels dans le fichier CSV    
        # Sauvegarder le DataFrame mis à jour
        df[df['verified'] == True].to_csv(os.path.join(data_dir, f'processed_{dataset_type}.csv'), index=False)
        pd.DataFrame(incorrect_labels).to_csv(os.path.join(data_dir, 'incorrect_labels.csv'), index= False)
        print(f"Updated and saved {dataset_type} dataset.")
        
    print(f"Total incorrect labels recorded for {dataset_type}: {len(incorrect_labels)}")


# Fonction pour vérifier les images qui ont déjà été validé manuellement (sans affichage)
def load_with_verified_check(filepath):
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0 :
        df = pd.read_csv(filepath)
        print(f"Loaded DataFrame from {filepath} with {len(df)} rows.")
        
        # Assurer que la colonne 'label' est bien en entier et 'verified' est correctement gérée
        df['label'] = df['label'].astype(int)
        
        if 'verified' not in df.columns:
            df['verified'] = False  # Initialiser la colonne 'verified' si elle n'existe pas
        else:
            df['verified'] = df['verified'].fillna(False)  # Remplir les valeurs manquantes avec False
    else :
        # Si le fichier n'existe pas ou est vide, créer un DataFrame vide
        df = pd.DataFrame(columns=['image_path', 'label', 'verified'])
        df['verified'] = False
        print(f"Created new DataFrame with columns 'image_path', 'label', 'verified'.")
    print(f"Total verified images: {df['verified'].sum()} out of {len(df)}")
    return df


# Fonction pour vérifier les labels présents dans les données
def verifier_labels(df, labels_dict):
    print("Vérification des labels présents dans le DataFrame...")
    unique_labels = df['label'].unique()
    print("Labels présents dans le DataFrame:", unique_labels)
    print("Clés de labels_dict:", list(labels_dict.keys()))

    # Vérifier les labels inconnus
    unknown_labels = [label for label in unique_labels if label not in labels_dict]
    if unknown_labels:
        print("Labels inconnus dans le DataFrame:", unknown_labels)
        # Ajouter les labels inconnus au dictionnaire si nécessaire
        for label in unknown_labels :
            labels_dict[label] = "unknown" # Ajouter un nom par défaut
    else:
        print("Tous les labels dans le DataFrame sont présents dans labels_dict.")


# Fonction pour traiter un chunk d'images en multithreading
def process_chunk_multithreaded(chunk, source_images_dir):
    incorrect_labels = []
    
    num_workers = os.cpu_count()
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(verify_image, row, source_images_dir) for _, row in chunk.iterrows()]
        for future in futures:
            result = future.result()
            if result:
                incorrect_labels.append(result)
    # Convertir en dataFrame et ajouter la colonne 'verified
    incorrect_labels_df = pd.DataFrame(incorrect_labels)
    if 'verified' not in incorrect_labels_df.columns :
        incorrect_labels_df['verified'] = False
        
    return incorrect_labels_df


# Fonction pour traiter les images en parallèle avec multiprocessing
def process_images_in_parallel(dataframe, source_images_dir):
    num_workers = cpu_count()
    chunk_size = len(dataframe) // num_workers
    chunks = [dataframe.iloc[i:i + chunk_size] for i in range(0, len(dataframe), chunk_size)]
    
    incorrect_labels_list = []
    
    with Pool(processes=num_workers) as pool:
        results = pool.starmap(process_chunk_multithreaded, [(chunk, source_images_dir) for chunk in chunks])
        for result in results:
            if not result.empty:
                incorrect_labels_list.append(result)
    
    # Combiner tous les DataFrames incorrects
    incorrect_labels_df = pd.concat(incorrect_labels_list).reset_index(drop=True)
    return incorrect_labels_df


def plot_incorrect_labels(dataset_type):
    
    incorrect_df = pd.DataFrame(incorrect_labels)
    dataset_specific_errors = incorrect_df[incorrect_df['dataset_type'] == dataset_type]
    
    if dataset_specific_errors.empty :
        print(f"No incorrect labels to plot for {dataset_type} dataset.")
        return

    # Exclure la catégorie "unknown" si elle n'est pas dans le DataFrame
    label_counts = dataset_specific_errors['label'].value_counts()
    label_names = [labels_dict.get(label, 'unknown') for label in label_counts.index]

    plt.figure(figsize=(12, 6))
    plt.bar(label_names, label_counts, color="skyblue")
    plt.xlabel('Class')
    plt.ylabel('Number of Incorrect Labels')
    plt.title(f'Number of Incorrect Labels by Class ({dataset_type.capitalize()})')
    plt.xticks(rotation=45)  # Incliner les labels de 45 degrés
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, f'incorrect_labels_{dataset_type}.png'))
    plt.show()


# Fonction pour générer un graphique de la proportion d'erreurs par dataset
def plot_error_proportions_across_datasets():
    if not incorrect_labels:
        print("No incorrect labels to plot error proportions across datasets.")
        return

    # Convert incorrect_labels to DataFrame
    incorrect_df = pd.DataFrame(incorrect_labels)
    
    # S'assurer que la colonne 'label' contient des entiers
    incorrect_df['label'] = incorrect_df['label'].astype(int)

    # S'assurer que 'dataset_type' existe
    if 'dataset_type' not in incorrect_df.columns:
        print("Error: 'dataset_type' column missing in incorrect labels data.")
        return

    # Print unique labels in incorrect_df for debugging
    print("\nUnique labels in incorrect_df after cleaning:")
    print(incorrect_df['label'].unique())

    # Count errors for each dataset
    train_errors = incorrect_df[incorrect_df['dataset_type'] == 'train']
    val_errors = incorrect_df[incorrect_df['dataset_type'] == 'val']
    test_errors = incorrect_df[incorrect_df['dataset_type'] == 'test']

    # Check for empty datasets
    if train_errors.empty:
        print("No errors found for train dataset.")
    if val_errors.empty:
        print("No errors found for val dataset.")
    if test_errors.empty:
        print("No errors found for test dataset.")

    # Load processed DataFrames
    processed_train_df = pd.read_csv(os.path.join(data_dir, 'processed_train.csv'))
    processed_val_df = pd.read_csv(os.path.join(data_dir, 'processed_val.csv'))
    processed_test_df = pd.read_csv(os.path.join(data_dir, 'processed_test.csv'))

    # Ensure DataFrames are not empty
    if processed_train_df.empty or processed_val_df.empty or processed_test_df.empty:
        print("One or more processed DataFrames are empty. Cannot compute error proportions.")
        return

    # Convertir la colonne 'label' en entier dans les DataFrames
    processed_train_df['label'] = processed_train_df['label'].astype(int)
    processed_val_df['label'] = processed_val_df['label'].astype(int)
    processed_test_df['label'] = processed_test_df['label'].astype(int)
    
    # Total number of samples in each dataset
    total_train = len(processed_train_df)
    total_val = len(processed_val_df)
    total_test = len(processed_test_df)

    # Ensure all labels are cleaned and consistent before counting errors
    train_error_count = train_errors['label'].value_counts().reindex(labels_dict.keys(), fill_value=0)
    val_error_count = val_errors['label'].value_counts().reindex(labels_dict.keys(), fill_value=0)
    test_error_count = test_errors['label'].value_counts().reindex(labels_dict.keys(), fill_value=0)

    # Print error counts for debugging
    print("\nTrain Error Count:\n", train_error_count)
    print("\nVal Error Count:\n", val_error_count)
    print("\nTest Error Count:\n", test_error_count)

    # Calculate error proportions
    train_proportions = train_error_count / total_train
    val_proportions = val_error_count / total_val
    test_proportions = test_error_count / total_test

    # Ensure all label names are consistent and clean
    label_names = [labels_dict[int(label)] for label in labels_dict.keys()]

    # Debug: Print label names
    print("\nLabel Names Used in Plot:\n", label_names)

    # Plotting
    bar_width = 0.25
    index = np.arange(len(labels_dict))

    plt.figure(figsize=(14, 7))
    plt.bar(index, train_proportions, bar_width, label='Train', color="salmon")
    plt.bar(index + bar_width, val_proportions, bar_width, label='Val', color="lightblue")
    plt.bar(index + 2 * bar_width, test_proportions, bar_width, label='Test', color="lightgreen")

    plt.xlabel('Class')
    plt.ylabel('Proportion of Incorrect Labels')
    plt.title('Proportions of Incorrect Labels Across Datasets')
    plt.xticks(index + bar_width, label_names, rotation=90)        # Rotate labels by 90 degrees
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'error_proportions_across_datasets.png'))
    plt.show()


# Fonction pour générer un graphique des proportions d'erreurs par classe
def plot_classwise_error_proportions(dataset_type):  
    
    if not incorrect_labels:
        print(f"No incorrect labels to plot for {dataset_type} dataset.")
        return
    
    incorrect_df = pd.DataFrame(incorrect_labels)
    
    # S'assurer que la colonne 'label' contient des entiers
    incorrect_df['label'] = incorrect_df['label'].astype(int)
    # Filtrer par type de dataset si nécessaire
    incorrect_df = incorrect_df[incorrect_df['dataset_type'] == dataset_type]
    
    label_counts = incorrect_df['label'].value_counts()
    total_count = len(incorrect_df)
    proportions = label_counts / total_count
    
    # Récupérer les noms des labels directement depuis labels_dict
    label_names = [labels_dict[label] for label in proportions.index if label in labels_dict]
    
    plt.figure(figsize=(12, 6))
    plt.bar(label_names, proportions, color='lightcoral')
    plt.xlabel('Class')
    plt.ylabel('Proportion of Incorrect Labels')
    plt.title(f'Proportion of Incorrect Labels by Class ({dataset_type.capitalize()})')
    plt.xticks(rotation=90)                                                                     # Incliner les labels de 90 degrés
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, f'classwise_error_proportions_{dataset_type}.png'))
    plt.show()


def plot_error_heatmap():
    if not incorrect_labels:
        print("No incorrect labels to plot in heatmap.")
        return

    incorrect_df = pd.DataFrame(incorrect_labels)
    # Convertir les labels du dataframe en entier pour correspondre à labels_dict
    incorrect_df['label'] = incorrect_df['label'].astype(int)
    
    # vérifions si le type de data est présent (train, test, val)
    if 'dataset_type' not in incorrect_df.columns:
        print("Error: 'dataset_type' column missing in incorrect labels data.")
        return
    
    # Filtrer les erreurs par dataset
    train_errors = incorrect_df[incorrect_df['dataset_type'] == 'train']
    val_errors = incorrect_df[incorrect_df['dataset_type'] == 'val']
    test_errors = incorrect_df[incorrect_df['dataset_type'] == 'test']
    
    # Vérifier qu'il y a des erreurs pour chaque dataset
    if train_errors.empty:
        print("No errors found for train dataset.")
    if val_errors.empty:
        print("No errors found for val dataset.")
    if test_errors.empty:
        print("No errors found for test dataset.")
    # Créer la liste des labels valides
    valid_labels = sorted(set(incorrect_df['label']).intersection(labels_dict.keys()))
    label_names = [labels_dict[label] for label in valid_labels]
    
    # Fonction pour créer une matrice de confusion
    def compute_confusion_matrix(df, labels):
        matrix = np.zeros((len(labels), len(labels)))
        for _, row in df.iterrows():
            true_label = row['label']
            if true_label in labels:
                index = labels.index(true_label)
                matrix[index, index] += 1
        return matrix
    
    train_matrix = compute_confusion_matrix(train_errors, valid_labels)
    val_matrix = compute_confusion_matrix(val_errors, valid_labels)
    test_matrix = compute_confusion_matrix(test_errors, valid_labels)

    plt.figure(figsize=(18, 10))

    def plot_matrix(matrix, title, pos):
        plt.subplot(1, 3, pos)
        plt.title(title)
        plt.imshow(matrix, cmap='Blues', interpolation='nearest')
        plt.xticks(np.arange(len(label_names)), label_names, rotation=90)               # Incliner les labels de 90 degrés
        plt.yticks(np.arange(len(label_names)), label_names)
        plt.colorbar()

    plot_matrix(train_matrix, 'Train Data', 1)
    plot_matrix(val_matrix, 'Validation Data', 2)
    plot_matrix(test_matrix, 'Test Data', 3)

    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'error_heatmap.png'))
    plt.show()

# Fonction pour générer un graphique des erreurs par dataset
def plot_error_counts_per_dataset():
    
    if not incorrect_labels:
        print("No incorrect labels to plot error counts per dataset.")
        return
    
    incorrect_df = pd.DataFrame(incorrect_labels)
    
     # Compter les erreurs par dataset
    train_error_count = incorrect_df[incorrect_df['dataset_type'] == 'train']['label'].value_counts()
    val_error_count = incorrect_df[incorrect_df['dataset_type'] == 'val']['label'].value_counts()
    test_error_count = incorrect_df[incorrect_df['dataset_type'] == 'test']['label'].value_counts()

    all_labels = sorted(set(train_error_count.index).union(val_error_count.index).union(test_error_count.index))
    label_names = [labels_dict.get(label, 'unknown') for label in all_labels]

    train_counts = train_error_count.reindex(all_labels, fill_value=0)
    val_counts = val_error_count.reindex(all_labels, fill_value=0)
    test_counts = test_error_count.reindex(all_labels, fill_value=0)

    bar_width = 0.25
    index = np.arange(len(all_labels))

    plt.figure(figsize=(14, 7))
    plt.bar(index, train_counts, bar_width, label='Train')
    plt.bar(index + bar_width, val_counts, bar_width, label='Val')
    plt.bar(index + 2 * bar_width, test_counts, bar_width, label='Test')

    plt.xlabel('Class')
    plt.ylabel('Number of Incorrect Labels')
    plt.title('Number of Incorrect Labels per Dataset')
    plt.xticks(index + bar_width, label_names, rotation=90)                                      # Incliner les labels de 90 degrés
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'error_counts_per_dataset.png'))
    plt.show()

def main():
    global incorrect_labels  # Déclaration pour utiliser dans la fonction d'affichage

    # Lire les fichiers de labels
    train_df, val_df, test_df = read_labels_files()

    # Vérification des labels
    verifier_labels(train_df, labels_dict)
    verifier_labels(val_df, labels_dict)
    verifier_labels(test_df, labels_dict)

    # Charger les DataFrames en vérifiant la colonne 'verified'
    processed_train_df = load_with_verified_check(os.path.join(data_dir, 'processed_train.csv'))
    processed_val_df = load_with_verified_check(os.path.join(data_dir, 'processed_val.csv'))
    processed_test_df = load_with_verified_check(os.path.join(data_dir, 'processed_test.csv'))

    # Afficher des informations pour le diagnostic
    print(f"Total train images: {len(train_df)}")
    print(f"Total processed train images: {len(processed_train_df)}")
    print(f"Total val images: {len(val_df)}")
    print(f"Total processed val images: {len(processed_val_df)}")
    print(f"Total test images: {len(test_df)}")
    print(f"Total processed test images: {len(processed_test_df)}")

    # Si les fichiers CSV n'existent pas ou sont vides, les créer en traitant les images
    if len(processed_train_df) < len(train_df):
        print("Processing new train images...")
        # Utiliser processed_train_df pour filtrer les images déjà traitées
        sampled_train_df = sample_images(train_df, 30, processed_train_df)                               # définition de la taille d'echantillon de travail sur train
        print(f"Sampled {len(sampled_train_df)} new train images to process")
        processed_train_df = pd.concat([processed_train_df, process_images_in_parallel(sampled_train_df, images_dir)])
        processed_train_df.to_csv(os.path.join(data_dir, 'processed_train.csv'), index=False)
        print(f"Processed and saved processed_train.csv with {len(processed_train_df)} images.")

    if len(processed_val_df) < len(val_df):
        print("Processing new validation images...")
        sampled_val_df = sample_images(val_df, 2, processed_val_df)                                     # définition de la taille d'echantillon de travail sur val
        print(f"Sampled {len(sampled_val_df)} new validation images to process")
        processed_val_df = pd.concat([processed_val_df, process_images_in_parallel(sampled_val_df, images_dir)])
        processed_val_df.to_csv(os.path.join(data_dir, 'processed_val.csv'), index=False)
        print(f"Processed and saved processed_val.csv with {len(processed_val_df)} images.")

    if len(processed_test_df) < len(test_df):
        print("Processing new test images...") 
        sampled_test_df = sample_images(test_df, 2, processed_test_df)                                   # définition de la taille d'echantillon du travail sur test
        print(f"Sampled {len(sampled_test_df)} new test images to process")
        processed_test_df = pd.concat([processed_test_df, process_images_in_parallel(sampled_test_df, images_dir)])
        processed_test_df.to_csv(os.path.join(data_dir, 'processed_test.csv'), index=False)
        print(f"Processed and saved processed_test.csv with {len(processed_test_df)} images.")

    # Afficher les images pour validation
    print("Displaying images for validation...")
    display_images(processed_train_df, images_dir, 'train')
    display_images(processed_val_df, images_dir, 'val')
    display_images(processed_test_df, images_dir, 'test')

    # Visualisations
    plot_incorrect_labels('train')
    plot_incorrect_labels('val')
    plot_incorrect_labels('test')

    plot_error_proportions_across_datasets()
    plot_error_heatmap()
    plot_error_counts_per_dataset()
    plot_classwise_error_proportions('train')
    plot_classwise_error_proportions('val')
    plot_classwise_error_proportions('test')

    print("Le processus de vérification des images et de visualisation est terminé.")

if __name__ == "__main__":
    main()


















