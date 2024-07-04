# ANN-from-scratch

réseau de neurones pour reconnaître des chiffres, codé en pur python et numpy

le réseau de neurones et le code nécessaire à son entraînement correspond aux fichiers "code2.py" et "code3.py" (il y en a deux versions, la deuxième contenant une couche de neurones supplémentaire)

les dossiers "weights_and_biais" et "weights_and_biais2" contiennent les poids des réseaux de neurone respectifs

les captures d'écrans correspondent aux courbes d'entraînememnt des réseaux de neurones

le code "test_neural_network.py" permet de tester les résultats du réseau de neurones sur quelques images du jeu de données

le code "draw_number.py" permet de tester le réseau de neurone sur des images dessinées en temps réel

Je n'ai pas pu ajouter le jeu de donnée par trop volumineux pour github, il s'agit du célèbre MNIST trouvable sur le site http://yann.lecun.com/exdb/mnist/ et ici convertit au format csv





paramètres et performances lors du dernier entraînement:
    train_score: ≃88%
    test_score: ≃88%
    temps d'entraînement: ≃40min
    nombre d'images: 20000
    nombre d'itérations: 50000
    pas: 0.05



avec 2 hidden layers, 60 neurones pour le 1e et 30 pour le 2e:
    train_score: 0.9481
    test_score: 0.913
    temps d'entraînement: ≃1h40min
    nombre d'images: 20000
    nombre d'itérations: 50000
    pas: 0.05
