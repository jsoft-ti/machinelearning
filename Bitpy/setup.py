import yaml
from resources.scripts.functions import setupData, generatePickleNextBTC
if __name__ == '__main__':

    with open('resources/scripts/main_variables.yaml') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        #setupData()
        #generatePickleNextBTC(data)
        #generatePickleNextPTR4(data)
        #generatePikleRecomendation(data)