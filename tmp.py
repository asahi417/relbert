import logging
import relbert

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')

if __name__ == '__main__':
    prompter = relbert.prompt.GradientTriggerSearch(
        model='roberta-large',
        export_name='test',
        n_trigger_i=4,
        n_trigger_b=2,
        n_trigger_e=2,
        n_iteration=1000,
        topk=100)
    prompter.get_prompt()


