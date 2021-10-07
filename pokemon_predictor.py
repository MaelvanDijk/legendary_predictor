import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle



# df = pd.read_csv('Pokemon wk 3.csv',delimiter=';', decimal=",")
#
# features = ["defense", "hp", "sp_attack", "sp_defense", "experience_growth"]
#
# rfclf = RandomForestClassifier(max_depth=30, random_state=0)
# rfclf.fit(df[features], df["is_legendary"])
#
# filename = 'legendary_rfclf.sav'
# pickle.dump(rfclf, open(filename, 'wb'))



def make_predictions(defense, hp, sp_attack, sp_defense, experience_growth):

    defense_norm = (int(defense) - 5) / (230 - 5)
    hp_norm = (int(hp) - 1) / (255 - 1)
    sp_attack_norm = (int(sp_attack) - 10) / (194 - 10)
    sp_defense_norm = (int(sp_defense) - 20) / (230 - 20)
    experience_growth_norm = (int(experience_growth) - 600000) / (1640000 - 600000)

    features_input = [defense_norm, hp_norm, sp_attack_norm, sp_defense_norm, experience_growth_norm]
    features_input = np.array(features_input)

    loaded_model = pickle.load(open("legendary_rfclf.sav", 'rb'))
    result = loaded_model.predict(features_input.reshape(1, -1))
    print(result[0])