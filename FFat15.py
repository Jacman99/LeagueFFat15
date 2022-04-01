'''
======STATS AND ML ON LoL DATA======

Name: Jason Combs
Student ID: 301352433
Student Email: jcombs@sfu.ca
Class: CMPT353 - Computational Data Science
Professor: Dr. Greg Baker
'''

# Basic import
import pandas as pd

# For cleaning data
from sklearn.pipeline import make_pipeline

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

# Imports for analysis
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier


# League data from EUW1 taken from: https://www.kaggle.com/bobbyscience/league-of-legends-diamond-ranked-games-10-min
data = pd.read_csv('high_diamond_ranked_10min.csv')
print(data)

# Creating these result columns for more clear data visualization later
blueTeamdict = {1 : "Victory", 0 : "Defeat"}
redTeamdict = {0 : "Victory", 1 : "Defeat"}
data['blueResult'] = data['blueWins'].map(blueTeamdict)
data['redResult'] = data['blueWins'].map(redTeamdict)

# Count number of games and wins
numGames = data['blueWins'].count()
numWins = data['blueWins'].sum()
print("Number of games in data:", numGames)
print("Number of blue team wins in data:", numWins)

# Split data up into two sets, if team blue won or lost
blueWins = data[data['blueWins'] == 1]
redWins = data[data['blueWins'] == 0]

# Plotting: Kills
sns.jointplot(x=data['blueKills'], y=data['redKills'], kind="reg")
plt.suptitle('Blue vs Red Team Kills', x=0.7)
plt.xlim(-.5, 23)
plt.ylim(-1, 24)
plt.show()

sns.jointplot(x=blueWins['blueKills'], y=blueWins['redKills'], kind="reg")
plt.suptitle('Blue vs Red Team Kills\nfor Games Blue Wins', x=0.7)
plt.xlim(-.5, 23)
plt.ylim(-1, 24)
plt.show()

sns.jointplot(x=redWins['blueKills'], y=redWins['redKills'], kind="reg")
plt.suptitle('Blue vs Red Team Kills\nfor Games Red Wins', x=0.7)
plt.xlim(-.5, 23)
plt.ylim(-1, 24)
plt.show()

# Plotting: TotalGold
sns.jointplot(x=data['blueTotalGold'], y=data['redTotalGold'], kind="reg")
plt.suptitle('Blue vs Red Team TotalGold', x=0.7)
plt.xlim(10000, 24000)
plt.ylim(11000, 23000)
plt.show()

sns.jointplot(x=blueWins['blueTotalGold'], y=blueWins['redTotalGold'], kind="reg")
plt.suptitle('Blue vs Red Team TotalGold\nfor Games Blue Wins', x=0.7)
plt.xlim(10000, 24000)
plt.ylim(11000, 23000)
plt.show()

sns.jointplot(x=redWins['blueTotalGold'], y=redWins['redTotalGold'], kind="reg")
plt.suptitle('Blue vs Red Team TotalGold\nfor Games Red Wins', x=0.7)
plt.xlim(10000, 24000)
plt.ylim(11000, 23000)
plt.show()

# Remove redundant or unnecessary features
data = data.drop(columns=['gameId']) # Game ID is not needed for our analysis

data = data.drop(columns=['blueDeaths', 'redDeaths', 'blueAvgLevel', 'redAvgLevel', 'blueTotalGold', 'redTotalGold', \
    'blueTotalExperience', 'redTotalExperience', 'blueGoldPerMin', 'redGoldPerMin', \
    'blueEliteMonsters', 'redEliteMonsters', 'blueTotalMinionsKilled', 'redTotalMinionsKilled', \
    'blueTotalJungleMinionsKilled', 'redTotalJungleMinionsKilled', 'blueCSPerMin', 'redCSPerMin', \
    'redFirstBlood', 'redGoldDiff', 'redExperienceDiff']) # Redundant features

# Get column names
columnNames = data.columns
print(columnNames)

# Machine Learning
X = data.drop(columns=['blueWins', 'blueResult', 'redResult']) # Result y values not needed for training
y = data['blueWins']

X_train, X_test, y_train, y_test = train_test_split(X, y)

gauss_model = make_pipeline(
        MinMaxScaler(),
        GaussianNB()
)
gauss_model.fit(X_train, y_train)
print("Gaussian Model Train and Test scores:")
print(gauss_model.score(X_train, y_train))
print(gauss_model.score(X_test, y_test))

rf_model = make_pipeline(
        MinMaxScaler(),
        RandomForestClassifier(n_estimators=100, max_depth=4)
)
rf_model.fit(X_train, y_train)
print("Random Forest Classifier Train and Test scores:")
print(rf_model.score(X_train, y_train))
print(rf_model.score(X_test, y_test))

svc_model = make_pipeline(
        MinMaxScaler(),
        SVC(kernel='linear', C=0.01)
)
svc_model.fit(X_train, y_train)
print("Support Vector Classifier Train and Test scores:")
print(svc_model.score(X_train, y_train))
print(svc_model.score(X_test, y_test))

gradient_model = GradientBoostingClassifier(n_estimators=50, max_depth=2, min_samples_leaf=0.1)
gradient_model.fit(X_train, y_train)
print("Gradient Boosting Classifier Train and Test scores:")
print(gradient_model.score(X_train, y_train))
print(gradient_model.score(X_test, y_test))

mlp_model1 = make_pipeline(
    MinMaxScaler(),
    MLPClassifier(max_iter=10000, activation='identity', solver='sgd', hidden_layer_sizes=(8, 5))
)
mlp_model1.fit(X_train, y_train)
print("Multi-Layer Perceptron Model 1 Train and Test scores:")
print(mlp_model1.score(X_train, y_train))
print(mlp_model1.score(X_test, y_test))

mlp_model2 = make_pipeline(
    MinMaxScaler(),
    MLPClassifier(max_iter=10000, activation='identity', solver='sgd', hidden_layer_sizes=(8, 6, 4))
)
mlp_model2.fit(X_train, y_train)
print("Multi-Layer Perceptron Model 2 Train and Test scores:")
print(mlp_model2.score(X_train, y_train))
print(mlp_model2.score(X_test, y_test))

mlp_model3 = make_pipeline(
    MinMaxScaler(),
    MLPClassifier(max_iter=10000, activation='identity', solver='adam', hidden_layer_sizes=(5, 4, 3))
)
mlp_model3.fit(X_train, y_train)
print("Multi-Layer Perceptron Model 3 Train and Test scores:")
print(mlp_model3.score(X_train, y_train))
print(mlp_model3.score(X_test, y_test))

model = VotingClassifier(estimators=[
    ('gauss', gauss_model),
    ('rf', rf_model),
    ('svc', svc_model),
    ('grad', gradient_model),
    ('mlp1', mlp_model1),
    ('mlp2', mlp_model2),
    ('mlp3', mlp_model3)
    ],
    voting='hard')
model.fit(X_train, y_train)
print("Voting Classifier consisting of all previous models Train and Test scores:")
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))
