import logging
import relbert

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

for t, name in zip(['c', 'relbert_output/ckpt/auto_d922/epoch_2', 'relbert_output/ckpt/auto_c932/epoch_2'],
                   ['Manual', 'AutoPrompt', 'P-tuning']):
    trainer = relbert.Trainer(
        template_type=t,
        epoch=2,
        export='examples/analysis_data/ckpt/{}'.format(name),
        exclude_relation="Class Inclusion"
    )
    trainer.train()


