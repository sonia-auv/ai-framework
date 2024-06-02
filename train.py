from ultralytics import YOLO
import torch
import argparse
import yaml

def main():
    parser = argparse.ArgumentParser(description="Entrainement AI SONIA Vision")
    # Choix du modèle
    parser.add_argument('--project', type=str, default='Tests', help='Nom du projet')
    parser.add_argument('--model-name', type=str, default='yolov8n.pt', help='Nom du modèle')
    parser.add_argument('--load-model', type=str, default=None, help='Chemin du modèle à charger')
    parser.add_argument('--dataset', type=str, default=None, help='Chemin du fichier de configuration du dataset')

    # Entraînement
    parser.add_argument('--batch-size', type=int, default=8, metavar='N', help='Taille des batchs (défaut: 8)')
    parser.add_argument('--image-size', type=int, default=640, metavar='N', help='Taille des images (défaut: 640)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='Nombre d\'epochs pour l\'entrainement (défaut: 10)')
    parser.add_argument('--start-epoch', type=int, default=0, metavar='N', help='Epoch de départ (défaut:0)')
    parser.add_argument('--save-period', type=int, default=1, help='Période de sauvegarde du modèle (défaut:1)')
    parser.add_argument('--resume', action='store_true', default=False, help='Poursuit un entraînement déjà commencé du modèle (défaut:False)')
    parser.add_argument('--time', type=int, default=10, help='Nombre d\'heure maximum pour un entraînement (défaut:10)')
    parser.add_argument('--patience', type=int, default=10, help='Nombre d\'epochs sans amélioration avant d\'arrêter (défaut:10)')

    # Augmentation des données
    parser.add_argument('--augment', action='store_true', default=False, help='Augmentation des données et entraînement sur ces données augmentées')
    parser.add_argument('--perspective', type=float, default=0, help='Augmentation par changement de perspective (défaut:0, plage:[0, 0.001])')
    parser.add_argument('--translate', type=float, default=0, help='Augmentation par translation (défaut:0, plage:[0, 1])')
    parser.add_argument('--scale', type=float, default=0, help='Augmentation par changement d\'échelle (défaut:0, plage:[0, inf])')
    parser.add_argument('--degrees', type=float, default=0, help='Augmentation par rotation (défaut:0, plage:[-180, 180])')
    parser.add_argument('--hsv_h', type=float, default=0, help='Augmentation par changement de luminosité (défaut:0, plage:[0, 1])')
    parser.add_argument('--hsv_s', type=float, default=0, help='Augmentation par changement de colorimétrie (défaut:0, plage:[0, 1])')
    parser.add_argument('--hsv_v', type=float, default=0, help='Augmentation par changement de colorimétrie (défaut:0, plage:[0, 1])')
    parser.add_argument('--p', type=float, default=0, help=' (défaut:0, plage:[0, 1])')

    # Export
    parser.add_argument('--plot', type=bool, default=True, help='Enregistre les courbes de l\'entraînement (défaut:True)')



    args = parser.parse_args(namespace=None)
    if torch.cuda.is_available():
        print('Cuda disponible')
    else:
        print('Cuda non disponible')

    if args.load_model is not None:
        model = YOLO('models/'+args.load_model)
    else:
        model = YOLO(args.model_name+'.pt')

    if args.augment:
        augment_settings = yaml.full_load(open('augment.yaml'))
        args.perspective = augment_settings['perspective']
        args.translate = augment_settings['translate']
        args.scale = augment_settings['scale']
        args.degrees = augment_settings['degrees']
        args.hsv_h = augment_settings['hsv_h']
        args.hsv_v = augment_settings['hsv_v']
        args.hsv_s = augment_settings['hsv_s']


    model.train(project=args.project,
                name=args.model_name,
                data='./datasets/'+args.dataset+'/data.yaml', 
                epochs=args.epochs, 
                imgsz=args.image_size, 
                save_period=args.save_period, 
                batch=args.batch_size, 
                plots=args.plot,
                resume=args.resume,
                time=args.time,
                patience=args.patience,
                perspective=args.perspective,
                translate=args.translate,
                scale=args.scale,
                degrees=args.degrees,
                hsv_h=args.hsv_h,
                hsv_v=args.hsv_v,
                hsv_s=args.hsv_s,
                cfg='./default.yaml')

if __name__ == "__main__":
   main()