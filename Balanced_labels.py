import pandas as pd
import os

def combine_datasets():
    """
    Combine deux datasets Excel en un seul dataset final
    """
    
    # Chemins des fichiers
    file1 = r"C:\Users\ADmiN\Desktop\video_project\final_dataset_final_precise.xlsx"
    file2 = r"C:\Users\ADmiN\Desktop\video_project\final_dataset_final_precise_B.xlsx"
    output_file = r"C:\Users\ADmiN\Desktop\video_project\dataset_final.xlsx"
    
    try:
        # Vérification de l'existence des deux fichiers
        if not os.path.exists(file1):
            print(f"Erreur: {file1} introuvable!")
            return
        
        if not os.path.exists(file2):
            print(f"Erreur: {file2} introuvable!")
            return
        
        # Lecture des deux fichiers Excel
        print("Lecture des datasets...")
        df1 = pd.read_excel(file1)
        df2 = pd.read_excel(file2)
        
        # Affichage des informations de base sur les datasets
        print(f"\nDataset 1 ({file1}):")
        print(f"Dimensions: {df1.shape}")
        print(f"Colonnes: {list(df1.columns)}")
        
        print(f"\nDataset 2 ({file2}):")
        print(f"Dimensions: {df2.shape}")
        print(f"Colonnes: {list(df2.columns)}")
        
        # Vérification de l'existence des colonnes requises
        required_columns = ['clip_name', 'clip_text']
        
        for col in required_columns:
            if col not in df1.columns:
                print(f"Avertissement: Colonne '{col}' introuvable dans {file1}")
            if col not in df2.columns:
                print(f"Avertissement: Colonne '{col}' introuvable dans {file2}")
        
        # Combinaison des datasets
        print("\nCombinaison des datasets...")
        combined_df = pd.concat([df1, df2], ignore_index=True)
        
        # Tri par clip_name pour assurer l'ordre correct (1-9)
        try:
            # Extraction des numéros de vidéo pour un tri approprié
            if 'clip_name' in combined_df.columns:
                # Création d'une colonne temporaire avec les numéros de vidéo extraits pour le tri
                combined_df['temp_video_num'] = combined_df['clip_name'].str.extract(r'video(\d+)', expand=False).astype(float)
                # Tri par numéro de vidéo, puis par clip_name
                combined_df = combined_df.sort_values(['temp_video_num', 'clip_name'], na_position='last')
                # Suppression de la colonne temporaire
                combined_df = combined_df.drop('temp_video_num', axis=1)
                print("✓ Vidéos triées dans l'ordre correct (1-9)")
            else:
                combined_df = combined_df.sort_values('clip_name')
        except Exception as e:
            print(f"Impossible de trier par numéro de vidéo: {e}, conservation de l'ordre original")
        
        # Affichage des informations sur le dataset combiné
        print(f"\nDataset combiné:")
        print(f"Dimensions: {combined_df.shape}")
        print(f"Total clips: {len(combined_df)}")
        
        # Affichage d'un échantillon des données
        if not combined_df.empty:
            print(f"\nÉchantillon de données:")
            print(combined_df.head())
        
        # Sauvegarde du dataset combiné
        print(f"\nSauvegarde dans {output_file}...")
        combined_df.to_excel(output_file, index=False)
        print(f"Dataset combiné sauvegardé avec succès dans {output_file}")
        
        # Affichage de quelques statistiques
        if 'clip_name' in combined_df.columns:
            print(f"\nStatistiques:")
            print(f"Clips uniques: {combined_df['clip_name'].nunique()}")
            if combined_df['clip_name'].str.contains('video', case=False, na=False).any():
                print("Distribution des vidéos:")
                video_counts = combined_df['clip_name'].str.extract(r'(video\d+)', expand=False).value_counts().sort_index()
                print(video_counts)
        
    except Exception as e:
        print(f"Erreur survenue: {str(e)}")
        print("Veuillez vérifier que les fichiers Excel sont correctement formatés et non corrompus.")

if __name__ == "__main__":
    combine_datasets()