import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

#data = pd.read_csv(r'C:\Users\Adam-PC\Desktop\Homework\MSCA 31008 - Data Mining\Project\fighter_level_dataset.csv')
for f in os.listdir('../fun/'): exec(open('../fun/'+f).read())
load(r"C:\Users\Adam-PC\Documents\GitHub\msca31008\out\c2-fighter-level-fillna.pkl")

## Select the relevant data
fighter_columns = ['fighter', 'age', 'Height_cms', 'Reach_cms', 'Weight_lbs', 'Stance', 'date']

fighter_history_columns = history_columns= ['current_lose_streak', 'current_win_streak', 'draw', 'longest_win_streak', 'losses', 'total_rounds_fought', 'total_time_fought(seconds)', 'total_title_bouts', 'win_by_Decision_Majority', 'win_by_Decision_Split', 'win_by_Decision_Unanimous', 'win_by_KO/TKO', 'win_by_Submission', 'win_by_TKO_Doctor_Stoppage', 'wins']

relevant_data = fighter_dataset[fighter_columns + fighter_history_columns]

## Remove duplicates of fighters, only keep the most up to date entry
relevant_data.sort_values(by = 'date', ascending = False, inplace = True)
relevant_data = relevant_data.drop_duplicates(subset = 'fighter')
relevant_data.drop(columns = 'date', inplace = True)
relevant_data.drop(columns = 'Stance', inplace = True) # Removed Stance for dataset simplicity, consider adding in later
relevant_data.fillna(0, inplace = True)

x = relevant_data.iloc[:, 1:]
y = relevant_data.iloc[:, 0]
#y = y.astype('category').cat.codes

model = TSNE(learning_rate = 100)

tsne_transformed = model.fit_transform(x)

xs = tsne_transformed[:, 0]
ys = tsne_transformed[:, 1]

## Plot the clusters
#plt.scatter(xs, ys, c = y)
#plt.ylim(-50, 50)
#plt.xlim(-50,50)
#plt.show()

fig, ax = plt.subplots(figsize = (8, 6))
ax.set_ylim(-50, 50)
ax.set_xlim(-50,50)
ax.scatter(xs, ys)

total_dataset = pd.DataFrame(np.column_stack([y, tsne_transformed]))
total_dataset.rename(columns = {0: 'fighter', 1: 'xs', 2: 'ys'}, inplace = True)

outlier3 = total_dataset[(total_dataset['xs'] <= 5) & (total_dataset['ys'] <= -20)]


#Alessandro Ricci
#Alexander Volkov
#Ali Bagautinov
#Alptekin Ozkilic
#Andrea Lee
#Andy Enz
#Anton Kuivanen
#Artem Lobov
#Ashkan Mokhtarian
#Ashley Yoder
#Bartosz Fabinski
#Benson Henderson
#Blagoy Ivanov
#Bojan Velickovic
#Brad Katona
#Brandon Davis
#Brendan O'Reilly
#Christian Colombo
#Claudia Gadelha
#Claudio Silva
#Clifford Starks
#Daichi Abe
#Danielle Taylor
#Danny Martinez
#Danny Mitchell
#David Michaud
#David Mitchell
#David Zawada
#Demetrious Johnson
#Desmond Green
#Devin Powell
#Diego Nunes
#Dominick Cruz
#Drakkar Klose
#Elias Silverio
#Elizabeth Phillips
#Elvis Mutapcic
#Emil Meek
#Eric Shelton
#Eryk Anders
#Frankie Edgar
#Garreth McLellan
#Gasan Umalatov
#Georges St-Pierre
#Gilbert Melendez
#Hacran Dias
#Holly Holm
#Ian Heinisch
#Ian McCall
#Irene Aldana
#Isaac Vallie-Flagg
#Ivan Jorge
#JJ Aldrich
#Jake Shields
#Jason Miller
#Jeff Hougland
#Jenel Lausa
#Jeremy Kennedy
#Jesse Ronson
#Jessica Aguilar
#Jessica-Rose Clark
#Jessin Ayari
#Jimmy Crute
#Joanna Jedrzejczyk
#Jocelyn Jones-Lybarger
#Jodie Esquibel
#John Gunderson
#Jonathan Martinez
#Jordan Johnson
#Jorge Gurgel
#Jose Aldo
#Josh Clopton
#Josh Copeland
#Kamaru Usman
#Katlyn Chookagian
#Keith Wisniewski
#Kevin Holland
#Konstantin Erokhin
#Kyle Bochniak
#Lauren Murphy
#Leonardo Augusto Leleco
#Levan Makashvili
#Lucie Pudilova
#Lukasz Sajewski
#Luke Cummo
#Magomed Bibulatov
#Marcin Held
#Marcin Tybura
#Masanori Kanehara
#Megan Anderson
#Merab Dvalishvili
#Michael Kuiper
#Michihiro Omigawa
#Michinori Tanaka
#Mickael Lebout
#Miguel Torres
#Mike Rhodes
#Mike Rodriguez
#Milton Vieira
#Mizuto Hirota
#Montel Jackson
#Nam Phan
#Nasrat Haqparast
#Nick Pace
#Nick Ring
#Nicolas Dalby
#Nina Ansaroff
#Norifumi Yamamoto
#Ramazan Emeev
#Rashid Magomedov
#Rick Glenn
#Ricky Simon
#Riki Fukuda
#Rodney Wallace
#Roger Hollett
#Ruslan Magomedov
#Ryan LaFlare
#Ryan Spann
#Sai Wang
#Said Nurmagomedov
#Sean O'Malley
#Sean Sherk
#Seohee Ham
#Sergio Pettis
#Sheymon Moraes
#Tarec Saffiedine
#Terrion Ware
#Tim Elliott
#Tom DeBlass
#Valentina Shevchenko
#Vince Morales
#Viviane Pereira
#Will Campuzano
#Yan Xiaonan
#Zach Makovsky
