from typing import Any


class BaseWorker():
    def __init__(self, config) -> None:
        self.init_components(config)
        self.gen_kwargs = config.get('gen_kwargs', {})
        self.model_id = config.model_name

    def init_components(self) -> None:
        '''
        Initialize model and processor, and anything needed in forward
        '''
        raise NotImplementedError

    @classmethod
    def from_config(cls, **kwargs):
        return cls(**kwargs)

    def forward(self, questions: list[str], image_paths: list[str, list], device, gen_kwargs) -> list[str]:

        raise NotImplementedError

    def __call__(self, device, **kwargs: Any) -> Any:
        for k in ['question', 'image_path']:
            assert k in kwargs, f'the key {k} is missing'
        questions = kwargs['question']
        image_paths = kwargs['image_path']
        answers = self.forward(
            questions=questions,
            image_paths=image_paths,
            device=device,
            gen_kwargs=self.gen_kwargs,
        )
        outputs = self.collate_batch_for_output(kwargs, answers=answers, prompts=questions)
        return outputs

    def collate_batch_for_output(self, batch, answers, prompts):

        ret = []
        len_batch = len(batch['id'])
        assert len(answers) == len(prompts) == len_batch

        for i in range(len_batch):
            new = {
                'sample_id': batch['id'][i], # modify the key
                'image': batch['image_path'][i],
                **{
                    k: v[i]
                    for k, v in batch.items() if k not in ('id', 'image_path')
                },
                'gen_model_id': self.model_id,
                'pred_response': answers[i],
                'gen_kwargs': dict(self.gen_kwargs), # omegaconf -> dict
            }

            ret.append(new)

        return ret
