#features (1 = sim , 2 = não)
#asa?
#chifre?
#pelo?

cavalo1 = [2,2,1]
cavalo2 =[2,2,2]
cavalo3 = [2,1,2]

unicornio1 = [1,1,1]
unicornio2 = [1,1,2]
unicornio3 = [1,2,2]

#cavalo = 1 e unicornio = 2
treino_x = [cavalo1,cavalo2,cavalo3,unicornio1,unicornio2,unicornio3]
treino_y = [1,1,1,2,2,2]

from sklearn.svm import LinearSVC
model = LinearSVC()
model.fit(treino_x, treino_y)


animal_misterioso = [1,1,2]
model.predict([animal_misterioso])

misterio1 = [2,2,2]
misterio2 = [1,2,1]
misterio3 = [1,1,1]

testes = [misterio1, misterio2, misterio3]
previsoes = model.predict(testes)

testes_classes = [2,1,1]


corretos = (previsoes == testes_classes).sum()
total = len(testes)
taxa_de_acerto = corretos/total
print("Taxa de acerto ", taxa_de_acerto)


from sklearn.metrics import accuracy_score

taxa_de_acerto = accuracy_score(testes_classes, previsoes)
print("Taxa de acerto", taxa_de_acerto * 100)




