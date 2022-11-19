import pandas as pd
import numpy as np

def devideDataset(dataset):
    LEARNING_DATASET_FRACTION = 0.87
    VALIDATING_DATASET_FRACTION = 0.09

    countRowsInDataset = dataset.shape[0]
    countRowsInLearningDataset = int(countRowsInDataset * LEARNING_DATASET_FRACTION)
    countRowsInValidatingDataset = int(countRowsInDataset * VALIDATING_DATASET_FRACTION)

    return np.split(dataset,
                    [countRowsInLearningDataset, countRowsInLearningDataset + countRowsInValidatingDataset])

dataset = pd.read_csv(f'heart.csv')

#ставим в конец столбец HeartDiseaseorAttack
columns = list(dataset.columns)
idxColFirst, idxColLast = columns.index('HeartDiseaseorAttack'), columns.index('Income')
columns[idxColFirst], columns[idxColLast] = columns[idxColLast], columns[idxColFirst]
dataset = dataset[columns]

#выравниваем множество
nullDataset = dataset.loc[dataset['HeartDiseaseorAttack'] == 0.0]
notNullDataset = dataset.loc[dataset['HeartDiseaseorAttack'] == 1.0]

countRowsInNotNullDataset = notNullDataset.shape[0]
#notNullDataset = notNullDataset.sample(500)
nullDataset = nullDataset.sample(countRowsInNotNullDataset)#nullDataset = nullDataset.sample(500)#

(learningNullDataset, validatingNullDataset, testingNullDataset) = devideDataset(nullDataset)
(learningNotNullDataset, validatingNotNullDataset, testingNotNullDataset) = devideDataset(notNullDataset)

learningDataset = pd.concat([learningNullDataset, learningNotNullDataset])
validatingDataset = pd.concat([validatingNullDataset, validatingNotNullDataset])
testingDataset = pd.concat([testingNullDataset, testingNotNullDataset])

columnsForNeurosumulator = list(map(lambda x: "X"+str(x), range(1, len(dataset.columns))))
columnsForNeurosumulator.append("D1")

learningDataset.columns = columnsForNeurosumulator
validatingDataset.columns = columnsForNeurosumulator
testingDataset.columns = columnsForNeurosumulator

# learningDataset.to_csv('learningDataset.csv', sep=",", index=False)
# validatingDataset.to_csv('validatingDataset.csv', sep=",", index=False)
# testingDataset.to_csv('testingDataset.csv', sep=",", index=False)

learningDataset = learningDataset.sample(learningDataset.shape[0])
validatingDataset = validatingDataset.sample(validatingDataset.shape[0])

learningDataset.to_excel('learningDataset1.xlsx', sheet_name='Обучающая выборка', index=False)
validatingDataset.to_excel('validatingDataset1.xlsx', sheet_name='Валидирующая выборка', index=False)
testingDataset.to_excel('testingDataset1.xlsx', sheet_name='Тестовая выборка', index=False)