// 0-8 = Classificacao
// 9 - 16 = Regressao
// 17 - 27 = Clustering
// 28 - 38 = RD

let dados = [
    {
        nome: 'Linear SVC',
        descricao: 'Variante da Máquina de Vetores de Suporte (SVM) que utiliza uma função de perda hinge e busca um hiperplano que melhor separa as classes.',
        link: 'https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html',
        vantagens: [
            'Eficiente para grandes conjuntos de dados',
            'Interpretabilidade relativamente alta (comparado a outros SVMs)',
            'Boa performance em dados linearmente separáveis'
        ],
        desvantagens: [
            'Pode ter dificuldades com dados não linearmente separáveis',
            'Sensível à escala dos dados'
        ],
        hiperparametros: [
            'C: Parâmetro de regularização que controla o trade-off entre a margem e o erro de classificação',
            'loss: Função de perda utilizada (hinge, squared_hinge)',
            'penalty: Norma utilizada para a regularização (l1 ou l2)'
        ],
        classe: 'classificacao'
    },
    {
        nome: 'Naive Bayes',
        descricao: 'Algoritmo baseado no teorema de Bayes que assume que as features são independentes entre si. É eficiente e simples de implementar.',
        link: 'https://scikit-learn.org/stable/modules/naive_bayes.html',
        vantagens: [
            'Eficiente em termos de tempo de treinamento e predição',
            'Requer poucos dados de treinamento',
            'Boa performance em problemas de texto e alta dimensionalidade'
        ],
        desvantagens:[
            'A suposição de independência entre as features pode não ser realista em muitos casos',
            'Sensível à presença de dados faltantes'
        ],
        hiperparametros: [
            'alpha: Parâmetro de suavização que evita a probabilidade zero'
        ]
        ,classe: 'classificacao'
    },
    {
        nome: 'KNeighbors Classifier',
        descricao: 'Classificador que atribui a uma nova instância a classe mais comum entre seus k vizinhos mais próximos.',
        link: 'https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html',
        vantagens: [
            'Simples e intuitivo',
            'Não faz suposições sobre a distribuição dos dados',
            'Bom desempenho em pequenos conjuntos de dados com bordas claramente definidas'
        ],
        desvantagens: [
            'Computacionalmente intensivo para grandes conjuntos de dados',
            'Sensível a dados ruidosos e irrelevantes',
            'Performance depende da escolha de k e da métrica de distância'
        ],
        hiperparametros: [
            'n_neighbors: Número de vizinhos a serem considerados',
            'weights: Função de ponderação utilizada (uniform, distance)',
            'metric: Métrica utilizada para calcular a distância (euclidean, manhattan, etc.)'
        ]
        ,classe: 'classificacao'
    },
    {
        nome: 'SVC',
        descricao: 'Máquina de Vetores de Suporte (SVM) que encontra o hiperplano que maximiza a margem entre as classes, podendo utilizar diferentes kernels para lidar com dados não linearmente separáveis.',
        link: 'https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html',
        vantagens: [
            'Boa performance em problemas de alta dimensionalidade',
            'Eficaz em conjuntos de dados não linearmente separáveis com kernels apropriados',
            'Flexível, pois suporta diferentes funções de kernel'
        ],
        desvantagens: [
            'Computacionalmente intensivo, especialmente para grandes conjuntos de dados',
            'Difícil de interpretar os resultados',
            'Sensível à escolha dos hiperparâmetros'
        ],
        hiperparametros: [
            'C: Parâmetro de regularização',
            'kernel: Função de kernel utilizada (linear, poly, rbf, sigmoid)',
            'gamma: Parâmetro de kernel para rbf, poly, e sigmoid'
        ]
        ,classe: 'classificacao'
    },
    {
        nome: 'Ensemble Classifiers',
        descricao: 'Combinação de múltiplos modelos de classificação para melhorar a precisão e a robustez do modelo final. Exemplos: Random Forest, Gradient Boosting.',
        link: 'https://scikit-learn.org/stable/modules/ensemble.html',
        vantagens: [
            'Alta precisão e robustez',
            'Redução do overfitting comparado a modelos individuais',
            'Versatilidade em diversos tipos de problemas'
        ],
        desvantagens: [
            'Pode ser computacionalmente caro e mais lento para treinar',
            'Difícil de interpretar',
            'Requer mais recursos de memória'
        ],
        hiperparametros: [
            'n_estimators: Número de estimadores no ensemble',
            'max_features: Número máximo de features consideradas para divisão de nós',
            'learning_rate: Taxa de aprendizado para métodos de boosting'
        ]
        ,classe: 'classificacao'
    },
    {
        nome: 'SGD Classifier',
        descricao: 'Classificador que utiliza o algoritmo de descida de gradiente estocástico para otimizar a função de perda. É eficiente para grandes conjuntos de dados.',
        link: 'https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html',
        vantagens: [
            'Eficiente em termos de memória e tempo para grandes conjuntos de dados',
            'Flexível com suporte a múltiplas funções de perda',
            'Pode ser usado com dados em streaming'
        ],
        desvantagens: [
            'Requer ajuste cuidadoso dos hiperparâmetros, especialmente a taxa de aprendizado',
            'Pode convergir lentamente se os dados não forem normalizados',
            'Sensível ao ruído nos dados'
        ],
        hiperparametros: [
            'loss: Função de perda utilizada (hinge, log, modified_huber, etc.)',
            'alpha: Parâmetro de regularização',
            'learning_rate: Esquema de atualização da taxa de aprendizado (constant, optimal, invscaling, adaptive)'
        ]
        ,classe: 'classificacao'
    },
    {
        nome: 'Kernel Approximation',
        descricao: 'Técnica utilizada para aproximar o cálculo do kernel em SVMs, tornando-os mais escaláveis para grandes conjuntos de dados. Exemplos: Kernel aproximado, Nyström.',
        link: 'https://scikit-learn.org/stable/modules/kernel_approximation.html',
        vantagens: [
            'Reduz a complexidade computacional em SVMs',
            'Permite a aplicação de SVMs em grandes conjuntos de dados',
            'Mantém boa precisão em comparação com kernels exatos'
        ],
        desvantagens: [
            'Aproximações podem introduzir erro, reduzindo a precisão',
            'Escolha do método de aproximação pode ser crítica',
            'Pode não ser tão eficaz em alguns tipos de dados'
        ],
        hiperparametros: [
            'n_components: Número de componentes na aproximação',
            'gamma: Coeficiente do kernel para RBF',
            'degree: Grau do polinômio em kernels polinomiais'
        ]
        ,classe: 'classificacao'
    },
    {
        nome: 'Random Forest',
        descricao: 'Modelo de ensemble que combina múltiplas árvores de decisão para realizar classificações ou regressões. Cada árvore é construída em um subconjunto aleatório dos dados e com um subconjunto aleatório das features, reduzindo o overfitting e aumentando a precisão.',
        link: 'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html',
        vantagens: [
            'Alta precisão em diversos tipos de dados',
            'Robusto a outliers e ruído',
            'Capacidade de lidar com dados faltantes',
            'Importante ferramenta para feature importance',
        ],
        desvantagens: [
            'Pode ser computacionalmente caro para grandes conjuntos de dados',
            'Pode ser difícil interpretar as decisões individuais de cada árvore'
        ],
        hiperparametros: [
            'n_estimators: Número de árvores na floresta',
            'max_depth: Profundidade máxima de cada árvore',
            'min_samples_split: Número mínimo de amostras necessárias para dividir um nó',
            'min_samples_leaf: Número mínimo de amostras necessárias em cada folha'
        ]
        ,classe: 'classificacao'
    },
    {
        nome: 'Logistic Regression',
        descricao: 'Modelo probabilístico utilizado para problemas de classificação binária ou multiclasse. Calcula a probabilidade de uma instância pertencer a uma determinada classe.',
        link: 'https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html',
        vantagens: [
            'Simples e fácil de interpretar',
            'Rápido para treinar e fazer predições',
            'Funciona bem em problemas lineares'
        ],
        desvantagens: [
            'Pode não funcionar bem em problemas não lineares sem engenharia de features',
            'Sensível a outliers'
        ],
        hiperparametros: [
            'C: Parâmetro de regularização que controla o trade-off entre a precisão do treinamento e a simplicidade do modelo',
            'penalty: Norma utilizada para regularização (l1, l2, elasticnet)',
            'solver: Algoritmo usado para otimizar a função de perda (liblinear, saga, etc.)'
        ]
        ,classe: 'regressao'
    },
    {
        nome: 'Linear Regression',
        descricao: 'Modelo que estabelece uma relação linear entre uma variável dependente e uma ou mais variáveis independentes.',
        link: 'https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html',
        vantagens: [
            'Fácil de entender e interpretar',
            'Eficiente para problemas de regressão simples e múltipla',
            'Computacionalmente eficiente para treinamento e inferência'
        ],
        desvantagens: [
            'Assume uma relação linear entre as variáveis',
            'Sensível a outliers',
            'Pode não capturar bem relações complexas entre as variáveis'
        ],
        hiperparametros: [
            'fit_intercept: Se deve calcular a interceptação para o modelo',
            'normalize: Se as variáveis independentes devem ser normalizadas',
            'n_jobs: Número de jobs a serem usados para o cálculo'
        ]
        ,classe: 'regressao'
    },
    {
        nome: 'Polinomial Regression',
        descricao: 'Extensão da regressão linear que permite modelar relações não lineares entre as variáveis através da introdução de termos polinomiais.',
        link: 'https://scikit-learn.org/stable/auto_examples/linear_model/plot_polynomial_interpolation.html',
        vantagens: [
            'Capaz de capturar relações não lineares',
            'Pode modelar dados complexos com múltiplas interações',
            'Simples de implementar usando transformações polinomiais'
        ],
        desvantagens: [
            'Maior risco de overfitting, especialmente com graus elevados de polinômios',
            'Pode ser difícil de interpretar',
            'Sensível a outliers'
        ],
        hiperparametros: [
            'degree: Grau do polinômio',
            'interaction_only: Se deve considerar apenas interações entre termos',
            'include_bias: Se deve incluir o termo de bias (interceptação) na transformação'
        ]
        ,classe: 'regressao'
    },
    {
        nome: "SGD Regressor",
        descricao: 'Um modelo linear que usa o algoritmo de descida de gradiente estocástico para otimizar a função de perda, eficiente para grandes conjuntos de dados.',
        link: 'https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html',
        vantagens: [
            'Eficiente para grandes conjuntos de dados',
            'Suporta várias funções de perda',
            'Pode ser usado com dados de streaming'
        ],
        desvantagens: [
            'Requer ajuste cuidadoso dos hiperparâmetros, como a taxa de aprendizado',
            'Pode ser sensível ao ruído nos dados',
            'Convergência lenta se os dados não forem normalizados'
        ],
        hiperparametros: [
            'loss: Função de perda utilizada (squared_loss, huber, epsilon_insensitive, etc.)',
            'penalty: Termo de regularização (l2, l1, elasticnet)',
            'alpha: Parâmetro de regularização'
        ]
        ,classe: 'regressao'
    },
    {
        nome: "Lasso",
        descricao: 'Modelo de regressão linear que utiliza regularização L1 para promover sparsidade no modelo, reduzindo o número de features.',
        link: 'https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html',
        vantagens: [
            'Promove sparsidade, selecionando automaticamente as features mais relevantes',
            'Pode melhorar a interpretabilidade do modelo',
            'Bom para problemas onde muitas features são irrelevantes'
        ],
        desvantagens: [
            'Pode descartar features que são, na verdade, relevantes',
            'Pode ser sensível ao ajuste do parâmetro de regularização',
            'Não lida bem com colinearidade entre as variáveis'
        ],
        hiperparametros: [
            'alpha: Parâmetro de regularização que controla a penalização L1',
            'max_iter: Número máximo de iterações para a convergência',
            'tol: Tolerância para a convergência'
        ]
        ,classe: 'regressao'
    },
    {
        nome: "Elastic Net",
        descricao: 'Combinação de regularização L1 e L2, útil quando há muitas features correlacionadas.',
        link: 'https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html',
        vantagens: [
            'Combina as vantagens da Lasso e da Ridge',
            'Útil quando há muitas features correlacionadas',
            'Pode melhorar a generalização do modelo'
        ],
        desvantagens: [
            'Pode ser mais difícil de interpretar do que Lasso ou Ridge isoladamente',
            'Requer ajuste cuidadoso dos hiperparâmetros',
            'Pode ser computacionalmente caro'
        ],
        hiperparametros: [
            'alpha: Parâmetro de regularização global',
            'l1_ratio: Mix de penalização L1 e L2',
            'max_iter: Número máximo de iterações para a convergência'
        ]
        ,classe: 'regressao'
    },
    {
        nome: "Ridge Regression",
        descricao: 'Modelo de regressão linear que utiliza regularização L2 para reduzir a complexidade do modelo, útil para lidar com multicolinearidade.',
        link: 'https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html',
        vantagens: [
            'Reduz o overfitting em modelos lineares',
            'Funciona bem em presença de multicolinearidade',
            'Mantém todas as features no modelo, mas com coeficientes reduzidos'
        ],
        desvantagens: [
            'Não reduz o número de features',
            'Menor interpretabilidade em comparação com Lasso',
            'Requer ajuste do parâmetro de regularização'
        ],
        hiperparametros: [
            'alpha: Parâmetro de regularização que controla a penalização L2',
            'solver: Algoritmo usado para otimizar a função de perda (auto, svd, cholesky, etc.)',
            'max_iter: Número máximo de iterações para a convergência'
        ]
        ,classe: 'regressao'
    },
    {
        nome: "SVR",
        descricao: 'Suporte a vetores de regressão (SVR) é um modelo que busca encontrar uma função que se aproxime dos dados observados dentro de uma margem de tolerância, utilizando diferentes kernels.',
        link: 'https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html',
        vantagens: [
            'Suporta regressão não linear com kernel rbf',
            'Robusto a outliers com o uso de margens flexíveis',
            'Pode lidar bem com alta dimensionalidade dos dados'
        ],
        desvantagens: [
            'Computacionalmente caro para grandes conjuntos de dados',
            'Sensível à escolha dos hiperparâmetros, especialmente o kernel e gamma',
            'Interpretação mais difícil do que regressão linear'
        ],
        hiperparametros: [
            'C: Parâmetro de regularização',
            'kernel: Tipo de kernel a ser utilizado (linear, poly, rbf, sigmoid)',
            'gamma: Coeficiente do kernel rbf'
        ]
        ,classe: 'regressao'
    },
    {
        nome: "Ensemble Regressors",
        descricao: 'Modelo que combina as predições de vários regressões base para melhorar a precisão do modelo final. Exemplos incluem Random Forest Regressor, Gradient Boosting Regressor.',
        link: 'https://scikit-learn.org/stable/modules/ensemble.html',
        vantagens: [
            'Alta precisão e robustez',
            'Reduz o overfitting ao combinar múltiplos modelos',
            'Pode capturar diferentes padrões nos dados que modelos individuais não conseguem'
        ],
        desvantagens: [
            'Pode ser computacionalmente caro e lento para treinar',
            'Mais difícil de interpretar do que modelos individuais',
            'Requer ajuste de múltiplos hiperparâmetros, o que pode ser complexo'
        ],
        hiperparametros: [
            'n_estimators: Número de modelos base no ensemble',
            'max_depth: Profundidade máxima das árvores em Random Forest ou Gradient Boosting',
            'learning_rate: Taxa de aprendizado para modelos como Gradient Boosting',
            'subsample: Proporção dos dados a serem usados para treinar cada modelo base'
        ]
        ,classe: 'regressao'
    },
    {
        nome: 'K-means',
        descricao: 'Algoritmo de partição que agrupa dados em k clusters. Cada observação pertence ao cluster com a média mais próxima.',
        link: 'https://en.wikipedia.org/wiki/K-means_clustering',
        vantagens: [
            'Simples e rápido para grandes conjuntos de dados',
            'Facilmente interpretável e implementável',
            'Bom para clusters esféricos e de tamanhos similares'
        ],
        desvantagens: [
            'Número de clusters precisa ser definido a priori',
            'Sensível a outliers e inicialização dos centroides',
            'Pode não funcionar bem com clusters de formas ou densidades variáveis'
        ],
        hiperparametros: [
            'n_clusters: Número de clusters a serem formados',
            'init: Método de inicialização dos centroides (k-means++, random, etc.)',
            'max_iter: Número máximo de iterações do algoritmo'
        ]
        ,classe: 'clustering'
    },
    {
        nome: 'DBSCAN',
        descricao: 'Algoritmo baseado em densidade que agrupa pontos de dados com base em sua densidade. Define clusters como regiões de alta densidade separadas por regiões de baixa densidade.',
        link: 'https://en.wikipedia.org/wiki/DBSCAN',
        vantagens: [
            'Não requer especificar o número de clusters a priori',
            'Identifica clusters de qualquer forma, ao contrário do K-means',
            'Robusto a outliers e ruído'
        ],
        desvantagens: [
            'Escolha sensível dos parâmetros eps e min_samples',
            'Pode falhar em detectar clusters de densidade variável',
            'Não funciona bem em dados de alta dimensionalidade'
        ],
        hiperparametros: [
            'eps: A distância máxima entre dois pontos para que sejam considerados vizinhos',
            'min_samples: Número mínimo de pontos para formar um cluster',
            'metric: A métrica de distância utilizada (euclidean, manhattan, etc.)'
        ]
        ,classe: 'clustering'
    },
    {
        nome: "Spectral Clustering",
        descricao: 'Método de clustering que utiliza técnicas de álgebra linear, como a decomposição espectral do grafo de afinidade dos dados, para realizar a clusterização.',
        link: 'https://en.wikipedia.org/wiki/Spectral_clustering',
        vantagens: [
            'Pode capturar estruturas complexas em dados',
            'Bom para dados que não se separam bem com clusters esféricos',
            'Versátil, pois pode ser usado com diferentes métricas de similaridade'
        ],
        desvantagens: [
            'Computacionalmente intensivo para grandes conjuntos de dados',
            'Requer ajuste cuidadoso dos parâmetros de afinidade',
            'Número de clusters deve ser conhecido a priori'
        ],
        hiperparametros: [
            'n_clusters: Número de clusters a serem formados',
            'affinity: Tipo de afinidade usada (nearest_neighbors, rbf, etc.)',
            'assign_labels: Método usado para rotular clusters (kmeans, discretize)'
        ]
        ,classe: 'clustering'
    },
    {
        nome: "GMM",
        descricao: '(Gaussian Mixture Model) Modelo probabilístico que assume que os dados são gerados a partir de uma mistura de várias distribuições Gaussianas com parâmetros desconhecidos.',
        link: 'https://en.wikipedia.org/wiki/Mixture_model',
        vantagens: [
            'Capaz de modelar clusters elípticos ou de formas mais complexas',
            'Fornece probabilidades de pertencimento ao cluster, útil para incerteza',
            'Pode lidar com variância diferente em diferentes clusters'
        ],
        desvantagens: [
            'Requer especificar o número de clusters a priori',
            'Pode convergir para um mínimo local, dependendo da inicialização',
            'Computacionalmente intensivo, especialmente para grandes dados'
        ],
        hiperparametros: [
            'n_components: Número de clusters (componentes gaussianos)',
            'covariance_type: Tipo de matriz de covariância (full, tied, diag, spherical)',
            'max_iter: Número máximo de iterações para o algoritmo EM'
        ]
        ,classe: 'clustering'
    },
    {
        nome: "MiniBatch KMeans",
        descricao: 'Versão do K-means que utiliza mini-lotes de dados para acelerar o processo de clustering, especialmente em grandes conjuntos de dados.',
        link: 'https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html',
        vantagens: [
            'Muito mais rápido que o K-means tradicional',
            'Eficiente em termos de memória, adequado para grandes conjuntos de dados',
            'Pode ser atualizado de forma online com novos dados'
        ],
        desvantagens: [
            'Pode ter menor precisão em comparação ao K-means tradicional',
            'Ainda sofre com limitações como sensibilidade a outliers e número fixo de clusters',
            'A qualidade dos clusters pode variar dependendo do tamanho do mini-batch'
        ],
        hiperparametros: [
            'n_clusters: Número de clusters a serem formados',
            'batch_size: Tamanho dos mini-lotes de dados',
            'max_iter: Número máximo de iterações do algoritmo'
        ]
        ,classe: 'clustering'
    },
    {
        nome: "Mean Shift",
        descricao: 'Algoritmo de clustering baseado em estimativa de densidade que busca os modos da distribuição de dados.',
        link: 'https://en.wikipedia.org/wiki/Mean_shift',
        vantagens: [
            'Não requer especificar o número de clusters a priori',
            'Pode identificar clusters de qualquer forma',
            'Robusto a outliers'
        ],
        desvantagens: [
            'Computacionalmente caro para grandes conjuntos de dados',
            'Pode falhar em detectar clusters em dados esparsos',
            'A escolha do bandwidth é crítica e pode ser difícil de ajustar'
        ],
        hiperparametros: [
            'bandwidth: A largura de banda usada na estimativa de densidade',
            'max_iter: Número máximo de iterações',
            'bin_seeding: Se o seeding dos clusters deve ser feito por binning'
        ]
        ,classe: 'clustering'
    },
    {
        nome: "VBGMM",
        descricao: '(Variational Bayesian Gaussian Mixture Model) Extensão do GMM que usa inferência variacional bayesiana para determinar automaticamente o número de componentes (clusters).',
        link: 'https://scikit-learn.org/stable/modules/mixture.html#variational-bayesian-gaussian-mixture',
        vantagens: [
            'Determina automaticamente o número de clusters',
            'Incorpora incerteza nas estimativas dos parâmetros',
            'Pode capturar estruturas mais complexas nos dados'
        ],
        desvantagens: [
            'Mais complexo e computacionalmente caro que o GMM tradicional',
            'Pode ser difícil de interpretar',
            'Requer ajuste cuidadoso dos hiperparâmetros'
        ],
        hiperparametros: [
            'n_components: Número máximo de componentes',
            'covariance_type: Tipo de matriz de covariância (full, tied, diag, spherical)',
            'max_iter: Número máximo de iterações para o algoritmo EM'
        ]
        ,classe: 'clustering'
    },
    {
        nome: 'Agglomerative Clustering',
        descricao: 'Algoritmo de clustering hierárquico que forma clusters de forma bottom-up, ou seja, cada ponto começa em seu próprio cluster e os clusters são iterativamente fundidos com base na proximidade.',
        link: 'https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html',
        vantagens: [
            'Não requer o número de clusters a priori',
            'Pode capturar hierarquias de clusters',
            'Bom para dados em que os clusters não são esféricos'
        ],
        desvantagens: [
            'Computacionalmente caro para grandes conjuntos de dados',
            'Escolha do método de linkagem pode afetar significativamente os resultados',
            'Menos eficiente para dados de alta dimensionalidade'
        ],
        hiperparametros: [
            'n_clusters: Número de clusters para encontrar',
            'linkage: Método de linkagem (ward, complete, average, single)',
            'affinity: Métrica usada para calcular a proximidade entre os clusters'
        ]
    },
    {
        nome: 'Birch',
        descricao: 'Algoritmo de clustering hierárquico que é eficiente para grandes conjuntos de dados. Ele constrói uma árvore com clusters em cada nó, facilitando a clusterização rápida e com menos memória.',
        link: 'https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html',
        vantagens: [
            'Eficiente em termos de memória e tempo para grandes conjuntos de dados',
            'Pode lidar com clusters de forma complexa',
            'Bom para dados que não se encaixam bem em outras formas de clusters'
        ],
        desvantagens: [
            'Pode não funcionar bem em dados de alta dimensionalidade',
            'A performance pode ser sensível ao ajuste dos parâmetros',
            'Menos interpretável do que outros métodos'
        ],
        hiperparametros: [
            'threshold: Limiar para controlar a condensação dos clusters',
            'branching_factor: Número máximo de clusters filhos por nó',
            'n_clusters: Número de clusters após o processo de condensação'
        ]
        ,classe: 'clustering'
    },
    {
        nome: 'Affinity Propagation',
        descricao: 'Algoritmo de clustering que identifica exemplares (representantes) entre as amostras e forma clusters em torno desses exemplares. A comunicação entre pontos de dados é usada para determinar a estrutura do cluster.',
        link: 'https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html',
        vantagens: [
            'Não requer a definição do número de clusters a priori',
            'Pode identificar exemplares (representantes) de clusters',
            'Funciona bem com dados ruidosos'
        ],
        desvantagens: [
            'Computacionalmente caro e lento para grandes conjuntos de dados',
            'A escolha do parâmetro de preferência pode ser desafiadora',
            'Pode resultar em muitos clusters, especialmente em grandes conjuntos de dados'
        ],
        hiperparametros: [
            'damping: Fator de amortecimento entre 0.5 e 1',
            'preference: Controle sobre o número de clusters ao ajustar a preferência',
            'max_iter: Número máximo de iterações para o algoritmo'
        ]
        ,classe: 'clustering'
    },
    {
        nome: 'OPTICS',
        descricao: 'Algoritmo de clustering baseado em densidade semelhante ao DBSCAN, mas que pode detectar clusters de densidade variável, ordenando os pontos de dados de acordo com sua densidade local.',
        link: 'https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html',
        vantagens: [
            'Identifica clusters de densidade variável',
            'Não requer número de clusters a priori',
            'Robusto a outliers e ruído'
        ],
        desvantagens: [
            'Computacionalmente intensivo para grandes conjuntos de dados',
            'Pode ser difícil de interpretar e ajustar parâmetros',
            'Menos eficiente em dados de alta dimensionalidade'
        ],
        hiperparametros: [
            'min_samples: Número mínimo de pontos para formar um cluster',
            'max_eps: Distância máxima entre dois pontos para que sejam considerados vizinhos',
            'metric: Métrica de distância usada para cálculo (euclidean, manhattan, etc.)'
        ]
        ,classe: 'clustering'
    },
    {
        nome: 'Randomized PCA',
        descricao: 'Técnica que reduz a dimensionalidade de dados, identificando as direções de maior variância.',
        link: 'https://scikit-learn.org/1.3/modules/decomposition.html#principal-component-analysis-pca',
        vantagens: [
            'Rápido para grandes conjuntos de dados',
            'Mantém a maior parte da variância nos dados',
            'Efetivo para dados com alta dimensionalidade'
        ],
        desvantagens: [
            'Pode não capturar bem estruturas não lineares',
            'Resultado pode variar devido à aleatoriedade'
        ],
        hiperparametros: [
            'n_components: Número de componentes principais a serem mantidos',
            'random_state: Semente para gerar números aleatórios',
        ]
        ,classe: 'rd'
    },
    {
        nome: 'TSNE',
        descricao: 'Algoritmo não linear que preserva a vizinhança local dos dados em um espaço de menor dimensão.',
        link: 'https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html',
        vantagens: [
            'Excelente para visualização de dados complexos em 2D ou 3D',
            'Preserva bem a estrutura local dos dados',
        ],
        desvantagens: [
            'Computacionalmente caro para grandes conjuntos de dados',
            'Difícil de interpretar em termos de componentes lineares',
        ],
        hiperparametros: [
            'n_components: Dimensão do espaço de saída',
            'perplexity: Número de vizinhos mais próximos usados para calcular as similaridades',
            'learning_rate: Taxa de aprendizado para otimização',
            'n_iter: Número de iterações para otimização'
        ]
        ,classe: 'rd'
    },
    {
        nome: "Isomap",
        descricao: "Técnica de redução de dimensionalidade baseada em geometria que preserva a estrutura geodésica dos dados, transformando-os em um espaço de menor dimensão.",
        link: "https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html",
        vantagens: [
            'Bom para dados que se distribuem em uma variedade de formas não lineares',
            'Mantém a estrutura global dos dados',
        ],
        desvantagens: [
            'Pode ser sensível a ruídos e outliers',
            'Pode ser computacionalmente caro para grandes conjuntos de dados',
        ],
        hiperparametros: [
            'n_neighbors: Número de vizinhos para considerar ao construir o grafo de similaridade',
            'n_components: Dimensão do espaço de saída'
        ]
        ,classe: 'rd'
    },
    {
        nome: "Spectral Embedding",
        descricao: "Algoritmo que utiliza o espectro do gráfico de similaridade dos dados para realizar a redução de dimensionalidade, capturando a estrutura dos dados em um espaço de menor dimensão.",
        link: "https://scikit-learn.org/stable/modules/generated/sklearn.manifold.SpectralEmbedding.html",
        vantagens: [
            'Útil para detectar a estrutura subjacente dos dados',
            'Preserva bem as relações de proximidade',
        ],
        desvantagens: [
            'Menos interpretável do que métodos lineares',
            'Pode ser sensível à escolha dos parâmetros',
        ],
        hiperparametros: [
            'n_components: Dimensão do espaço de saída',
            'affinity: Tipo de afinidade usado para construir o gráfico de similaridade (nearest_neighbors, rbf)',
        ]
        ,classe: 'rd'
    },
    {
        nome: "LLE",
        descricao: "Técnica de redução de dimensionalidade não linear que preserva a estrutura local dos dados, mapeando-os em um espaço de menor dimensão.",
        link: "https://scikit-learn.org/stable/modules/generated/sklearn.manifold.LocallyLinearEmbedding.html",
        vantagens: [
            'Preserva a geometria local dos dados',
            'Bom para visualização em 2D ou 3D',
        ],
        desvantagens: [
            'Pode falhar em capturar a estrutura global dos dados',
            'Menos eficiente para dados muito ruidosos',
        ],
        hiperparametros: [
            'n_neighbors: Número de vizinhos para considerar ao construir o grafo de similaridade',
            'n_components: Dimensão do espaço de saída',
            'method: Método de mapeamento (standard, modified, etc.)'
        ]
        ,classe: 'rd'
    },
    {
        nome: "Kernel Approximation",
        descricao: "Técnica usada para aproximar o mapeamento de um kernel não linear em um espaço de menor dimensão, facilitando a aplicação de modelos lineares em dados não lineares.",
        link: "https://scikit-learn.org/stable/modules/kernel_approximation.html",
        vantagens: [
            'Escalável para grandes conjuntos de dados',
            'Permite aplicar técnicas lineares a dados não lineares',
        ],
        desvantagens: [
            'Pode perder precisão dependendo da aproximação',
            'Escolha do kernel e parâmetros pode ser desafiadora',
        ],
        hiperparametros: [
            'n_components: Dimensão do espaço de saída',
            'kernel: Tipo de kernel usado (linear, rbf, etc.)',
            'gamma: Parâmetro de kernel para alguns tipos de kernels (rbf, poly, etc.)'
        ]
        ,classe: 'rd'
    },
    {
        nome: 'PCA',
        descricao: '(Principal Component Analysis) Técnica linear que transforma as variáveis originais em componentes principais que capturam a maior variância possível dos dados.',
        link: 'https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html',
        vantagens: [
            'Simples e eficiente para reduzir a dimensionalidade em problemas lineares',
            'Útil para interpretação e visualização de dados',
            'Reduz ruído e melhora a performance de modelos'
        ],
        desvantagens: [
            'Pode não capturar bem a estrutura não linear dos dados',
            'Componentes principais podem ser difíceis de interpretar'
        ],
        hiperparametros: [
            'n_components: Número de componentes principais a serem mantidos',
            'svd_solver: Algoritmo usado para a decomposição (auto, full, arpack, randomized)'
        ]
        ,classe: 'rd'
    },
    {
        nome: 'Factor Analysis',
        descricao: 'Método estatístico usado para descrever variabilidade entre variáveis observadas e correlacionadas em termos de um menor número de variáveis latentes (fatores).',
        link: 'https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FactorAnalysis.html',
        vantagens: [
            'Útil para identificar a estrutura subjacente dos dados',
            'Reduz a dimensionalidade enquanto retém a variância explicada'
        ],
        desvantagens: [
            'Requer um grande número de observações para ser eficaz',
            'Pode ser sensível à escolha do número de fatores'
        ],
        hiperparametros: [
            'n_components: Número de fatores a serem mantidos',
            'svd_method: Método de decomposição de valores singulares (lapack, randomised, etc.)'
        ]
        ,classe: 'rd'
    },
    {
        nome: 'UMAP',
        descricao: '(Uniform Manifold Approximation and Projection) Algoritmo de redução de dimensionalidade não linear que preserva tanto a estrutura global quanto local dos dados em um espaço de menor dimensão.',
        link: 'https://umap-learn.readthedocs.io/en/latest/',
        vantagens: [
            'Excelente para visualização em 2D ou 3D',
            'Mantém bem a estrutura global e local dos dados',
            'Escalável para grandes conjuntos de dados'
        ],
        desvantagens: [
            'Requer ajustes de parâmetros para resultados ideais',
            'Pode ser mais lento que outros métodos em conjuntos de dados extremamente grandes'
        ],
        hiperparametros: [
            'n_neighbors: Controla a escala local da estrutura dos dados',
            'min_dist: Controla a compactação da projeção em espaços de menor dimensão',
            'metric: Métrica de distância usada para construir o gráfico de similaridade'
        ]
        ,classe: 'rd'
    },
    {
        nome: 'MDS',
        descricao: '(Multidimensional Scaling) Método que tenta representar as distâncias entre todos os pares de pontos em um espaço de menor dimensão de forma que as distâncias sejam preservadas tanto quanto possível.',
        link: 'https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html',
        vantagens: [
            'Bom para visualizar distâncias relativas entre pontos',
            'Preserva a estrutura global dos dados'
        ],
        desvantagens: [
            'Computacionalmente caro para grandes conjuntos de dados',
            'Pode ser sensível a ruídos e outliers'
        ],
        hiperparametros: [
            'n_components: Número de dimensões no espaço de saída',
            'metric: Se a escala multidimensional é métrica ou não métrica',
            'max_iter: Número máximo de iterações para otimização'
        ]
        ,classe: 'rd'
    },
    {
        nome: 'Autoencoder',
        descricao: 'Rede neural usada para aprender uma representação compacta (codificação) dos dados, geralmente em menor dimensionalidade, que pode ser usada para reconstrução dos dados originais.',
        link: 'https://keras.io/examples/autoencoders/',
        vantagens: [
            'Capaz de capturar estruturas não lineares complexas',
            'Escalável para grandes conjuntos de dados',
            'Pode ser ajustado para diferentes tipos de dados (imagens, texto, etc.)'
        ],
        desvantagens: [
            'Requer maior poder computacional para treinar',
            'Pode ser difícil de ajustar para resultados ótimos'
        ],
        hiperparametros: [
            'layers: Estrutura da rede (número e tamanho das camadas)',
            'activation: Função de ativação usada nas camadas ocultas',
            'learning_rate: Taxa de aprendizado usada no otimizador'
        ]
        ,classe: 'rd'
    }
];