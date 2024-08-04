DATASET_DIR = 'datasets/'
DATASET_YAML = '/data.yaml'
def make_data_yaml(dataset_name):
    text = 'path: ../datasets/{}\ntrain: train/images/\nval: val/images/\ntest: test/images/\n\nnames:\n0: bin\n1: bin_bleue\n2: bin_rouge\n3: gate\n4: gate_rouge\n5: gate_bleue\n6: torpille\n7: torpille_target\n8: buoy\n9: samples_table\n10: path'.format(dataset_name)
    with open(DATASET_DIR+dataset_name+DATASET_YAML, "a") as f:
        f.write(text)