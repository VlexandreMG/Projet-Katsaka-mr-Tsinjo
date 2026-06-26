# generer_test_images.py
import os
import cv2
import numpy as np

def créer_image_test(type_feuille, chemin_sortie):
    # 1. Créer un fond noir pour l'image (400x400 pixels, 3 canaux de couleur BGR)
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # 2. Dessiner une forme de "feuille" (un grand rectangle ou une ellipse verte)
    # En BGR : (Vert=180, Bleu=40, Rouge=40) -> Un beau vert agricole
    cv2.rectangle(img, (50, 50), (350, 350), (40, 180, 40), -1)
    
    if type_feuille == "malade":
        # 3. Ajouter des fausses pustules de rouille (points marrons/oranges)
        # En BGR : (Bleu=20, Vert=70, Rouge=150) -> Une couleur rouille/brune
        np.random.seed(42) # Pour que les points soient toujours au même endroit
        for _ in range(120): # On dessine 120 pustules
            x = np.random.randint(60, 340)
            y = np.random.randint(60, 340)
            rayon = np.random.randint(2, 5)
            cv2.circle(img, (x, y), rayon, (20, 70, 150), -1)
            
    # Sauvegarder l'image sur le disque
    cv2.imwrite(chemin_sortie, img)

if __name__ == "__main__":
    print("📁 Création des dossiers de test...")
    os.makedirs("data/saines", exist_ok=True)
    os.makedirs("data/malades", exist_ok=True)
    
    print("🖼️ Génération des fausses images de feuilles...")
    
    # Générer 2 feuilles saines
    créer_image_test("saine", "data/saines/feuille_saine_1.png")
    créer_image_test("saine", "data/saines/feuille_saine_2.png")
    
    # Générer 2 feuilles malades
    créer_image_test("malade", "data/malades/feuille_malade_1.png")
    créer_image_test("malade", "data/malades/feuille_malade_2.png")
    
    print("✨ C'est prêt ! Tu as maintenant 4 images de simulation dans ton dossier 'data/'.")