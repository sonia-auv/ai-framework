import labelbox
import json
import argparse

API_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjazNnNHZrd3czeTMyMDcyNTdyb201dG0xIiwib3JnYW5pemF0aW9uSWQiOiJjamRmODljNGxxdnNmMDEwMHBvdnFqeWppIiwiYXBpS2V5SWQiOiJjbHluc21sMGgwYWtlMDd3ajF3b2gwM2dwIiwic2VjcmV0IjoiM2UyNGM2NDJmYzAxZDYwZTI4MzFlOWQ3MzNiNWFlNmYiLCJpYXQiOjE3MjEwOTY4NTAsImV4cCI6MjM1MjI0ODg1MH0.P61ZVls7up7WM_mmmTmGXDT06eEBhk-vDC2ZcEbws14'
PROJECT_ID = 'clyn59tv9063e07y07nku1bwx'
DATASET_DIR = 'datasets/'

parser = argparse.ArgumentParser(description="Entraînement AI SONIA Vision")
# Choix du dataset
parser.add_argument('--dataset', 
                    type=str, 
                    required=True,
                    help='Nom du dataset')
parser.add_argument('--project-id', 
                    type=str, 
                    default=PROJECT_ID,
                    help='ID du projet')
args = parser.parse_args(namespace=None)

client = labelbox.Client(api_key=API_KEY)
params = {}

project = client.get_project(args.project_id)
export_task = project.export_v2(params=params)

export_task.wait_till_done()
if export_task.errors:
	print(export_task.errors)
else:
	with open(DATASET_DIR+args.dataset+'/data.json', 'w', encoding='utf-8') as f:
		json.dump(export_task.result, f, ensure_ascii=False, indent=4)