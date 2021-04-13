import logging
import relbert

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')

if __name__ == '__main__':
    prompter = relbert.prompt.GradientTriggerSearch(
        model='roberta-base', export_name='test', batch=4, topk=2)
    prompter.get_prompt(0)

