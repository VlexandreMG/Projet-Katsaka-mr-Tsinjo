import os
import cv2
import numpy as np
import pandas as pd

def extraire_features_image(chemin_image):
    """
    Extrait les caractéristiques physiques d'une image de feuille de maïs.
    Retourne un dictionnaire avec 'pct_rouille' et 'rugosite'.
    """
    # 1. Chargement de l'image en BGR (format par défaut d'OpenCV)
    img = cv2.imread(chemin_image)
    if img is None:
        raise ValueError(f"Impossible de charger l'image : {chemin_image}")
        
    # --- ÉTAPE PRÉALABLE : Isoler la feuille du fond (Segmentation simple) ---
    # Pour calculer un ratio par rapport à la FEUILLE et non à toute l'image,
    # on doit savoir quels pixels appartiennent à la feuille.
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # [cite: 24]
    
    # Masque générique pour le VERT de la feuille saine + JAUNE/MARRON de la rouille
    # On exclut ainsi les fonds blancs, noirs ou neutres.
    borne_basse_feuille = np.array([5, 20, 20])
    borne_haute_feuille = np.array([85, 255, 255])
    masque_feuille_totale = cv2.inRange(img_hsv, borne_basse_feuille, borne_haute_feuille)
    
    # Nombre total de pixels composants la feuille
    nb_pixels_feuille = np.sum(masque_feuille_totale == 255)
    
    # Sécurité : si l'image est vide ou sans feuille détectée
    if nb_pixels_feuille == 0:
        nb_pixels_feuille = img.shape[0] * img.shape[1] # On rabat sur la taille de l'image
        
    # --- ÉTAPE 1 : Calcul de X1 - pct_rouille --- 
    # Définition des plages de couleur HSV pour la rouille (marrons, jaunes, oranges) [cite: 26]
    # H (Teinte): 5 à 25 couvre l'orange et le marron/jaune chaud
    # S (Saturation): 40 à 255 pour éviter les teintes trop grises/ternes
    # V (Valeur/Luminosité): 40 à 255 pour éviter les ombres trop noires [cite: 25]
    borne_basse_rouille = np.array([5, 40, 40])
    borne_haute_rouille = np.array([25, 255, 255])
    
    # Création du masque binaire : 255 (blanc) là où c'est de la rouille, 0 (noir) ailleurs
    masque_rouille = cv2.inRange(img_hsv, borne_basse_rouille, borne_haute_rouille)
    
    # On s'assure de ne compter que la rouille *qui est sur la feuille*
    masque_rouille_strict = cv2.bitwise_and(masque_rouille, masque_feuille_totale)
    
    # Compter le nombre de pixels de rouille
    nb_pixels_rouille = np.sum(masque_rouille_strict == 255)
    
    # Calcul du ratio X1 
    pct_rouille = nb_pixels_rouille / nb_pixels_feuille
    
    # --- ÉTAPE 2 : Calcul de X2 - rugosite (Filtre de Sobel) --- [cite: 29, 32]
    # On passe en niveaux de gris pour analyser les textures/contours
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calcul des gradients horizontaux (Scharr ou Sobel) et verticaux
    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3) # [cite: 31]
    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3) # [cite: 31]
    
    # Magnitude du gradient (intensité combinée des variations brusques)
    magnitude_sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # On applique le masque de la feuille pour ne calculer la rugosité que sur la feuille
    magnitude_feuille = magnitude_sobel[masque_feuille_totale == 255]
    
    # X2 = Variance de l'intensité des gradients (plus il y a de pustules, plus ça varie) [cite: 30, 32]
    rugosite = np.var(magnitude_feuille) if len(magnitude_feuille) > 0 else 0.0
    
    

    return {
        "pct_rouille": float(pct_rouille),
        "rugosite": float(rugosite)
    }

def generer_dataset_complet(dossier_racine):
    """
    Parcourt automatiquement les dossiers 'saines' et 'malades' [cite: 13, 22]
    et extrait les caractéristiques pour construire le DataFrame final.
    """
    categories = {
        "saines": 0,   # Label 0 pour Saine [cite: 12]
        "malades": 1   # Label 1 pour Malade [cite: 12]
    }
    
    donnees_combinees = []
    
    # Boucle de parcours des dossiers 
    for nom_dossier, label in categories.items():
        chemin_dossier = os.path.join(dossier_racine, nom_dossier)
        
        # Vérification de l'existence du sous-dossier
        if not os.path.exists(chemin_dossier):
            print(f"⚠️ Attention : Le dossier {chemin_dossier} n'existe pas. On passe.")
            continue
            
        print(f"📁 Analyse du dossier : {nom_dossier} (Label: {label})...")
        
        # Parcours de toutes les images du dossier
        for nom_fichier in os.listdir(chemin_dossier):
            if nom_fichier.lower().endswith(('.png', '.jpg', '.jpeg')):
                chemin_complet = os.path.join(chemin_dossier, nom_fichier)
                
                try:
                    # Extraction des descripteurs numériques
                    features = extraire_features_image(chemin_complet)
                    
                    # Structuration de la ligne
                    ligne = {
                        "ID_Image": nom_fichier,
                        "pct_rouille": features["pct_rouille"],
                        "rugosite": features["rugosite"],
                        "votre_variable": 0.0, # À remplacer par ta variable personnelle [cite: 33, 34]
                        "label_malade": label
                    }
                    donnees_combinees.append(ligne)
                    
                except Exception as e:
                    print(f"❌ Erreur lors du traitement de {nom_fichier} : {e}")

    # Transformation de la liste de dictionnaires en DataFrame Pandas 
    df = pd.DataFrame(donnees_combinees)
    return df

# --- BLOC DE TEST DE NOTRE MVP ---
if __name__ == "__main__":
    # Spécifie ici le chemin vers ton dossier de données
    # Structure attendue : data/saines/ et data/malades/ [cite: 13, 14]
    DOSSIER_DATA = "data" 
    
    print("🚀 Démarrage de l'extraction des caractéristiques...")
    df_features = generer_dataset_complet(DOSSIER_DATA)
    
    # Affichage du résultat final
    if not df_features.empty:
        print("\n✅ Extraction réussie ! Aperçu du DataFrame obtenu :")
        print(df_features.head())
        
        # Optionnel : Sauvegarder en CSV pour la suite (Partie 3)
        # df_features.to_csv("dataset_maïs_features.csv", index=False)
    else:
        print("\n⚠️ Aucun DataFrame généré. Vérifie que tes images sont bien dans 'data/saines/' ou 'data/malades/'.")