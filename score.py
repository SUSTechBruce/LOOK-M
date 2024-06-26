import json
import argparse
import os
import pandas as pd
from collections import defaultdict

def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = round(sum(d[key] for d in dict_list) / len(dict_list), 5)*100
        
    return mean_dict

def main(args):
    
    ########################## Set Dataset Taxonomy ##########################
    dataset_list={
        'Realistic Temporal': [
            'ActionLocalization', 'ActionPrediction', 'ActionSequence', 'CharacterOrder', 
            'CounterfactualInference', 'EgocentricNavigation', 'MovingAttribute', 'MovingDirection', 
            'ObjectExistence', 'ObjectInteraction', 'ObjectShuffle', 'SceneTransition', 'StateChange'
        ],
        'Realistic Semantic': [
            'ALFRED','CLEVR-Change','DocVQA','IEdit','MMCoQA','MultiModalQA',
            'nuscenes','OCR-VQA','SlideVQA','Spot-the-Diff','TQA','WebQA','WikiVQA'
        ],
        'Diagnostic': ['TextNeedleInAHaystack', 'ImageNeedleInAHaystack', 'GPR1200']
    }

    ########################## Collect Evaluation Result ##########################
    result_dir = args.result_dir

    result = {}
    for model_name in args.models:
        print(f'Collecting result of {model_name}...')
        model_result = {}
        for task_name, dataset_names in dataset_list.items():
            task_result = {}
            if not dataset_names:   # TODO
                continue
            for dataset in dataset_names:
                try:
                    eval_path = os.path.join(result_dir, model_name, dataset, 'eval.json')
                    if not os.path.exists(eval_path):
                        print(f'\t{model_name}--{dataset}  No evaluation file found')
                        task_result[dataset] = {}
                        continue
                    dataset_result = json.load(open(eval_path))
                except Exception as e:
                    print(eval_path)
                task_result[dataset] = dataset_result
            model_result[task_name] = task_result
            
        result[model_name] = model_result
    
    ########################## Save Result ##########################
    json.dump(
        result, 
        open(os.path.join(result_dir, model_name, 'result.json'), 'w'), 
        ensure_ascii=False, indent=4)

    # Function to parse JSON and create a dataset for DataFrame
    def parse_json_to_df(data):
        parsed_data = []
        try:
            for model, tasks in data.items():
                model_data = {'Model': model}
                for task, datasets in tasks.items():
                    for dataset, metrics in datasets.items():
                        for metric, value in metrics.items():
                            if metric not in [
                                "image_quantity_level-Accuracy", 
                                "image_quantity_level-Result", 
                                "Diff-Accuracy"]:  # Ignore
                                model_data[f"{dataset} ({metric})"] = round(value*100, 2)
                parsed_data.append(model_data)
        except Exception as e:
            print(e, value)
        return pd.DataFrame(parsed_data)

    # Convert JSON to DataFrame & Save to CSV
    df = parse_json_to_df(result)
    df.to_csv(os.path.join(result_dir, model_name, 'result.csv'), index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', type=str, required=True)
    parser.add_argument("--models", type=str, nargs='+', help="list of models")
    args = parser.parse_args()
    main(args)

'''
python score.py \
    --result-dir outputs \
    --models llava-v1.5-7b
'''