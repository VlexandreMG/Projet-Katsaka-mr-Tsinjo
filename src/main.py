# main.py
from features import generer_dataset_complet

if __name__ == "__main__":
    print("⏳ Analyse des feuilles de maïs en cours...")
    
    # On indique le chemin de notre dossier de test
    chemin_data = "data"
    
    # On lance l'extraction globale
    df_resultat = generer_dataset_complet(chemin_data)
    
    print("\n--- RÉSULTATS DU TEST ---")
    if df_resultat.empty:
        print("❌ Le tableau est vide. Vérifie que :")
        print("   1. Le dossier 'data/' existe à la racine.")
        print("   2. Les sous-dossiers 'saines/' et 'malades/' contiennent des images (.jpg, .png).")
    else:
        print(f"✅ Succès ! {len(df_resultat)} images ont été analysées.")
        print("\nAperçu des données extraites (Pandas DataFrame) :")
        # Affiche les premières lignes du tableau
        print(df_resultat.to_string())