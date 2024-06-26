
from datetime import datetime
from argparse import ArgumentParser

import json, os
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
import torch

from utils import get_worker_class, MileBenchDataset
from omegaconf import OmegaConf
from tqdm import tqdm
import gc


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default='data/MileBench')
    parser.add_argument('--dataset_name', default='data/sample.json')
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--bsz', default=1, type=int)
    parser.add_argument('--batch-image', default=1, type=int)
    parser.add_argument('--combine_image', default=None, type=int, help='Use combined N images for evaluation.')
    parser.add_argument('--model_configs', default='configs/model_configs.yaml')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--kv_mode', default="origin", choices=["origin", "h2o", "weighted_merge", "pivot_merge", "text_prior_h2o", "text_prior_weighted_merge", "text_prior_pivot_merge", "snapkv", "avg_merge", "mean_h2o", "text_prior_avg_merge"])
    parser.add_argument('--hh_ratio', default=0.1, type=float)
    parser.add_argument('--recent_ratio', default=0.1, type=float)
    args = parser.parse_args()

    args.output_pth = os.path.join(args.output_dir, f"{args.model_name}/{args.dataset_name}/pred.json")
    os.makedirs(os.path.dirname(args.output_pth), exist_ok=True)
    return args

def split_data(data):
    '''
    Split the data by the images number
    ex: {
        2: [sample1, ...]
        3: [sample2, ...]
    }
    '''
    data_dict = {}
    for d in data:
        n_img = len(d['task_instance']['images_path'])
        if n_img in data_dict:
            data_dict[n_img].append(d)
        else:
            data_dict[n_img] = [d]
    return data_dict

def save(results, accelerator, args):
    if accelerator.is_main_process:
        if os.path.exists(args.output_pth):
            if not args.overwrite:
                print(f'{args.output_pth} exists. Please pass `overwrite=True` to avoid unwanted overwriting.')
                exit(0)
        json.dump(results, open(args.output_pth, 'w'), ensure_ascii=False, indent=4)

def main(args):
    import torch.distributed as dist
    accelerator = Accelerator()
    accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.bsz
    accelerator.state.deepspeed_plugin.deepspeed_config['train_batch_size'] = args.bsz * dist.get_world_size()
    accelerator.print(f'{datetime.now()}: Generation of {args.model_name} to {args.dataset_name}')

    ######################### Loading Data #########################
    data_dir = args.data_dir
    dataset_name = args.dataset_name
    combine_image = args.combine_image
    dataset_dir = os.path.join(data_dir, dataset_name)
    img_dir = os.path.join(dataset_dir, 'images')
    model_name = args.model_name
    core_annotation = json.load(
        open(os.path.join(dataset_dir, 
        f'{dataset_name}_combined_{combine_image}.json' 
            if combine_image and combine_image!=1 else f'{dataset_name}.json')))
    # split data by images number
    data_dict = split_data(core_annotation['data'])
    ################################################################

    #################### Initializing Worker ######################
    worker_class = get_worker_class(args.model_name)
    # breakpoint()
    models_configs = OmegaConf.load(args.model_configs)
    if not models_configs.get(args.model_name):
        # raise ValueError
        print(args.model_name)
    config = list(models_configs.values())[0]
    config.hh_ratio = args.hh_ratio
    config.recent_ratio = args.recent_ratio
    config.device = str(accelerator.device)
    config.kv_mode = args.kv_mode
    worker = worker_class.from_config(config=config)
    # prepare model for accelerator
    worker.model = accelerator.prepare(worker.model)

    ################################################################

    ###################### Start Generating ########################

    print('Initialization Finished')
    print(f'Predicting {dataset_name} Using {model_name}')
    prediction_results = []
    for n_img, sub_data in data_dict.items():
        print(f'Proceeding {n_img}-length images samples | Num: {len(sub_data)}')
        lc_dataset = MileBenchDataset(
                        annotation=sub_data,
                        task_instructions=core_annotation['meta_data']['task_instruction'],
                        img_dir=img_dir,
                        max_context_len=config.max_context_len,
                        n_tokens_per_image=config.n_tokens_per_image,
                        tokenizer=worker.tokenizer,
                        dataset_name=dataset_name,
                        combine_image=combine_image,
                        )
        lc_dataloader = DataLoader(dataset=lc_dataset,
                                batch_size=max(int(args.batch_image/n_img),1),
                                shuffle=False,
                                num_workers=8,
                                collate_fn=lc_dataset.collate_fn)
        lc_dataloader = accelerator.prepare_data_loader(lc_dataloader, device_placement=False)

        # start inference
        for batch in tqdm(lc_dataloader) if accelerator.is_main_process else lc_dataloader:
            outputs = worker(device=accelerator.device, **batch) # list[dict], with the key "answer" added to each item
            all_predictions = accelerator.gather_for_metrics(outputs)
            prediction_results.extend(all_predictions)
        # gather all results
        accelerator.wait_for_everyone()
        # remove the repetition
        prediction_results = list({item['sample_id']: item for item in prediction_results}.values())
        print(f'Generation done {len(prediction_results)}')
        gc.collect(); torch.cuda.empty_cache()
    ################################################################

    ######################### Save Result ##########################
    save(prediction_results, accelerator, args)
    ################################################################


if __name__ == '__main__':
    args = parse_args()
    main(args)
