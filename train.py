from ultralytics import YOLO
import torch
import argparse

def main():
    parser = argparse.ArgumentParser(description="Entrainement AI SONIA Vision")
    # Choix du modèle
    parser.add_argument('--model-name', type=str, default='yolov8n.pt', help='Nom du modèle')
    parser.add_argument('--load-model', type=str, default=None, help='Chemin du modèle à charger')
    parser.add_argument('--dataset-yaml', type=str, default=None, help='Chemin du fichier de configuration du dataset')

    # Entraînement
    parser.add_argument('--batch-size', type=int, default=8, metavar='N', help='Taille des batchs (défaut: 8)')
    parser.add_argument('--image-size', type=int, default=640, metavar='N', help='Taille des images (défaut: 640)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='Nombre d\'epochs pour l\'entrainement (défaut: 10)')
    parser.add_argument('--start-epoch', type=int, default=0, metavar='N', help='Epoch de départ (défaut:0)')
    parser.add_argument('--save-period', type=int, default=1, help='Période de sauvegarde du modèle (défaut:1)')
    # parser.add_argument('--augment', action='store_true', default=False, help='Augmentation des données et entraînement sur ces données augmentées')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("Cuda disponible :", args.cuda)

    if args.load_model is not None:
        model = YOLO(args.load_model)
    else:
        model = YOLO(args.model_name)

    model.train(data=args.dataset_yaml, epochs=args.epochs, imgsz=args.image_size, save_period=args.save_period, batch=args.batch_size, plots=True)

if __name__ == "__main__":
   main()