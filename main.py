import argparse
from ai_sonia import AiSonia
from labeling_sonia import LabelingSonia
from dataset_sonia import DatasetSonia
from utils import mix_datasets
        
def parse():
    parser = argparse.ArgumentParser(description="AI SONIA Vision")
    # Choix de la tâche réalisée
    parser.add_argument('--task', 
                        type=str, 
                        required=True, 
                        choices=['train', 'test', 'init_dataset', 'load_labels', 'mix_dataset'], 
                        help='Tache réalisé')
    # Init dataset
    parser.add_argument('--src', 
                        type=str, 
                        default=None,
                        help='Source dataset path (required)')
    parser.add_argument('--dataset-name', 
                        type=str, 
                        default='on_working_directory', 
                        help='Dataset name (default:on_working_directory)')
    parser.add_argument('--train-proba', 
                        type=float, 
                        default=0.7, 
                        help='Proportion of train (default:0.7)')
    parser.add_argument('--val-proba', 
                        type=float, 
                        default=0.2, 
                        help='Proportion of val (default:0.2)')
    # Load labels
    parser.add_argument('--project-id', 
                        type=str, 
                        default=None,
                        help='Id du projet LabelBox')
    # Mix dataset
    # Choix du dataset
    parser.add_argument('--new-dataset', 
                        type=str, 
                        default=None,
                        help='Nom du dataset')
    parser.add_argument('--dataset-list',
                        '--list', 
                        nargs='+', 
                        default=None,
                        help='Datasets à mélanger')
    # Train or test models
    # Choix du modèle
    parser.add_argument('--project', 
                        type=str, 
                        default=None, 
                        help='Nom du projet (défaut:None)')
    parser.add_argument('--model', 
                        type=str, 
                        default=None,
                        choices=[None, 'yolo'], 
                        help='Modèle choisi (défaut:yolo)')
    parser.add_argument('--name', 
                        type=str, 
                        default=None, 
                        help='Nom du modèle (défaut:modèle choisi)')
    parser.add_argument('--load-model', 
                        type=str, 
                        default=None, 
                        help='Chemin du modèle à charger')
    parser.add_argument('--dataset', 
                        type=str, 
                        default=None, 
                        help='Chemin du fichier de configuration du dataset')
    # Entraînement
    parser.add_argument('--resume', 
                        action='store_true', 
                        default=False, 
                        help='Poursuit un entraînement déjà commencé du modèle (défaut:False)')
    # Augmentation des données
    parser.add_argument('--augment', 
                        action='store_true', 
                        default=False, 
                        help='Augmentation des données et entraînement sur ces données augmentées (défaut:False)')
    # Inférence
    parser.add_argument('--inf-confidence', 
                        type=float, 
                        default=0.5, 
                        help='Confiance minimum pour l\'inférence (défaut:0.5)')
    parser.add_argument('--inf-img-size', 
                        type=int, 
                        default=[600, 400], 
                        help='Taille de l\'image pour l\'inférence (défaut:[600, 400])')
    # Parsing des arguments
    return parser.parse_args(namespace=None)

def main():
    args = parse()
    if args.task == 'train' or args.task == 'test':
        assert not args.dataset is None
        assert not args.model is None
        assert not args.load_model is None
        sonia_ai = AiSonia(args)
        if args.task == 'train':
            sonia_ai.train()
        elif args.task == 'test':
            sonia_ai.predict()
    elif args.task == 'init_dataset':
        assert not args.src is None
        assert not args.dataset_name is None
        sonia_dataset = DatasetSonia(args.src, args.dataset_name, args.train_proba, args.val_proba)
        sonia_dataset.create()
    elif args.task == 'load_labels':
        assert not args.dataset is None
        assert not args.project_id is None
        labels = LabelingSonia(args.dataset, args.project_id)
        labels.convert_labels()
    elif args.task == 'mix_dataset':
        assert not args.new_dataset is None
        assert not args.dataset_list is None
        assert len(args.dataset_list) > 1
        mix_datasets(args.new_dataset, args.dataset_list)


if __name__ == "__main__":
   main()