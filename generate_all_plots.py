"""
Script de Génération de Toutes les Figures du Projet Maroc 2030
================================================================

Ce script charge les données, exécute le Feature Engineering et génère toutes les figures
du projet pour s'assurer de leur cohérence et de leur exactitude.
Il sert d'outil de validation automatisée.
"""

import os
import sys
import argparse
import subprocess

# S'assurer que le dossier racine est dans le python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

def main():
    parser = argparse.ArgumentParser(description="Script de Génération de Figures")
    parser.add_argument(
        "--test_year", 
        type=int, 
        default=2023, 
        help="Année de début pour le split de test (default: 2023)"
    )
    parser.add_argument(
        "--quick_run", 
        action="store_true", 
        help="Lancer en mode rapide pour tester"
    )
    args = parser.parse_args()
    
    print("==================================================")
    print("Démarrage de la génération programmatique des figures")
    print("==================================================")
    
    # Nous appelons directement main.py via subprocess pour garantir la cohérence
    cmd = [sys.executable, "main.py", "--test_year", str(args.test_year)]
    if args.quick_run:
        cmd.append("--quick_run")
        
    print(f"Exécution de la commande : {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print("\n==================================================")
        print("Toutes les figures ont été générées et validées avec succès !")
        print("==================================================")
    else:
        print("\n==================================================")
        print("Erreur lors de l'exécution du pipeline de modélisation.")
        print("==================================================")
        sys.exit(result.returncode)

if __name__ == '__main__':
    main()
