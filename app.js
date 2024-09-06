
function toggleResultados(sliceStart, sliceEnd) {
    let section = document.getElementById("resultados");
    let container = document.getElementById('movable-container');

    // Verifica se a seção está exibindo os dados do botão clicado
    if (section.dataset.active === `${sliceStart}-${sliceEnd}`) {
        // Se já estiver exibindo, oculta a seção e reseta a posição do container
        section.classList.remove('visible');
        setTimeout(() => {
            section.style.display = "none";
            section.innerHTML = "";
            section.dataset.active = ""; // Reseta o estado ativo
            container.classList.remove('moved');
            clearButtonSelection();
        }, 500); // Aguarda a transição antes de esconder
    } else {
        // Caso contrário, gera e exibe o novo conteúdo
        let temp = "";
        for (let dado of dados.slice(sliceStart, sliceEnd)) {
            let textoExtra = "";
            textoExtra += '<h2 class="topics">Vantagens</h2>';
            if (Array.isArray(dado.vantagens)) {
                textoExtra += dado.vantagens.map(linha => `<p>-${linha}</p>`).join('');
            } else {
                textoExtra += "<p>Texto genérico para ser substituído pelo texto da sua base de dados.</p>";
            }
            textoExtra += '<h2 class="topics">Desvantagens</h2>';
            if (Array.isArray(dado.desvantagens)) {
                textoExtra += dado.desvantagens.map(linha => `<p>-${linha}</p>`).join('');
            } else {
                textoExtra += "<p>Texto genérico para ser substituído pelo texto da sua base de dados.</p>";
            }
            textoExtra += '<h2 class="topics">Hiperparâmetros</h2>';
            if (Array.isArray(dado.hiperparametros)) {
                textoExtra += dado.hiperparametros.map(linha => `<p>-${linha}</p>`).join('');
            } else {
                textoExtra += "<p>Texto genérico para ser substituído pelo texto da sua base de dados.</p>";
            }

            temp += `
                <div class="item-resultado">
                    <h2>
                        <a class="no_grif">${dado.nome}</a>
                        <button class="expandir" onclick="expandContent(this)">+</button>
                    </h2>
                    <p class="descricao-meta">${dado.descricao}</p>
                    <div class="extra-content" style="display: none;">
                        <p>${textoExtra}</p>
                        <a href=${dado.link} target="_blank">Mais informações</a>
                    </div>
                </div>
            `;
        }
        section.innerHTML = temp;
        section.style.display = "block";
        section.dataset.active = `${sliceStart}-${sliceEnd}`;
        container.classList.add('moved');
        setTimeout(() => {
            section.classList.add('visible');
        }, 50); // Adiciona um pequeno atraso para garantir que a transição funcione
    }
}

function pesquisarModelo() {
    let input = document.getElementById('campo-pesquisa').value.toLowerCase();

    // Verifica se o campo de entrada está vazio
    if (input.trim() === "") {
        mostrarAlerta("Por favor, digite o nome de um modelo.");
        return; // Sai da função se o campo estiver vazio
    }

    let resultados = dados.filter(dado => dado.nome.toLowerCase().includes(input));

    if (resultados.length > 0) {
        let section = document.getElementById("resultados");
        let container = document.getElementById('movable-container');
        let temp = `
            <button class="fechar" onclick="fecharPesquisa()">x</button>
        `;

        resultados.forEach(resultado => {
            let textoExtra = "";
            textoExtra += '<h2 class="topics">Vantagens</h2>';
            if (Array.isArray(resultado.vantagens)) {
                textoExtra += resultado.vantagens.map(linha => `<p>-${linha}</p>`).join('');
            } else {
                textoExtra += "<p>Texto genérico para ser substituído pelo texto da sua base de dados.</p>";
            }
            textoExtra += '<h2 class="topics">Desvantagens</h2>';
            if (Array.isArray(resultado.desvantagens)) {
                textoExtra += resultado.desvantagens.map(linha => `<p>-${linha}</p>`).join('');
            } else {
                textoExtra += "<p>Texto genérico para ser substituído pelo texto da sua base de dados.</p>";
            }
            textoExtra += '<h2 class="topics">Hiperparâmetros</h2>';
            if (Array.isArray(resultado.hiperparametros)) {
                textoExtra += resultado.hiperparametros.map(linha => `<p>-${linha}</p>`).join('');
            } else {
                textoExtra += "<p>Texto genérico para ser substituído pelo texto da sua base de dados.</p>";
            }

            temp += `
                <div class="item-resultado">
                    <h2>
                        <a class="no_grif">${resultado.nome}</a>
                        <button class="expandir" onclick="expandContent(this)">+</button>
                    </h2>
                    <p class="descricao-meta">${resultado.descricao}</p>
                    <div class="extra-content" style="display: none;">
                        <p>${textoExtra}</p>
                        <a href=${resultado.link} target="_blank">Mais informações</a>
                    </div>
                </div>
            `;
        });

        section.innerHTML = temp;
        section.style.display = "block";
        container.classList.add('moved');
        setTimeout(() => {
            section.classList.add('visible');
        }, 50); // Adiciona um pequeno atraso para garantir que a transição funcione

        selectButton('search');
    } else {
        mostrarAlerta("Modelo não encontrado.");
    }
}

function mostrarAlerta(mensagem) {
    let alerta = document.getElementById('alerta');
    alerta.innerText = mensagem;
    alerta.style.display = 'block';

    setTimeout(() => {
        alerta.style.display = 'none';
    }, 3000); // Alerta desaparecerá após 3 segundos
}

function fecharPesquisa() {
    let section = document.getElementById("resultados");
    let container = document.getElementById('movable-container');
    
    section.classList.remove('visible');
    setTimeout(() => {
        section.style.display = "none";
        section.innerHTML = "";
        container.classList.remove('moved');
        clearButtonSelection();
    }, 500); // Aguarda a transição antes de esconder
}

function Cla() {
    toggleResultados(0, 8);
    selectButton('Cla');
}

function Reg() {
    toggleResultados(9, 16);
    selectButton('Reg');
}

function Clu() {
    toggleResultados(17, 27);
    selectButton('Clu');
}

function Red() {
    toggleResultados(27, 38);
    selectButton('Red');
}


function expandContent(button) {
    const extraContent = button.parentElement.nextElementSibling.nextElementSibling;
    if (extraContent.style.display === "none") {
        extraContent.style.display = "block";
        button.innerText = "-";
    } else {
        extraContent.style.display = "none";
        button.innerText = "+";
    }
}

function selectButton(buttonClass) {
    clearButtonSelection();
    const button = document.querySelector(`.${buttonClass}`);
    if (button) {
        button.classList.add('selected');
    } else {
        console.error(`Button with class ${buttonClass} not found.`);
    }
}

function clearButtonSelection() {
    document.querySelectorAll('.Cla, .Reg, .Clu, .Red, .search').forEach(button => {
        button.classList.remove('selected');
    });
}

const questions = {
    start: {
        text: "Você tem mais de 50 amostras?",
        buttons: [
            { text: "Sim", next: "more_than_50_samples" },
            { text: "Não", next: "less_than_50_samples" }
        ]
    },
    less_than_50_samples: {
        text: "São necessários mais dados para treinar um modelo",
    },

    more_than_50_samples: {
        text: "Você está prevendo uma categoria?",
        buttons: [
            { text: "Sim", next: "predicting_category" },
            { text: "Não", next: "predicting_quantity" }
        ]
    },

    predicting_category: {
        text: "Você está lidando com dados rotulados?",
        buttons: [
            { text: "Sim", next: "labeled_data_yes" },
            { text: "Não", next: "labeled_data_no" }
        ]
    },

    predicting_quantity:{
        text: 'Você está prevendo um quantidade?',
        buttons: [
            { text: "Sim", next: "quantity_yes" },
            { text: "Não", next: "quantity_no" }
        ]

    },

    quantity_no: {
        text: 'Você quer dimensionar seus dados?',
        buttons: [
            { text: "Sim", next: "quantity_no_yes" },
            { text: "Não", next: "quantity_no_no" }
        ]
    },

    quantity_no_no:{
        text: 'Aparentemente você está prevendo uma estrutura, não existe nenhum modelo na base de dados para está situação'
    },

    //Clustering
    labeled_data_no: {
        text: "A quantidade de categorias é conhecida?",
        buttons: [
            { text: "Sim", next: "clu_label_quantity_yes" },
            { text: "Não", next: "clu_label_quantity_no" }
        ]
    },

    clu_label_quantity_yes: {
        text: "Quantos dados você tem?",
        buttons: [
            { text: "<10K", next: "clu_yes_lt10" },
            { text: ">10K", next: "clu_yes_mt10" }
        ]
    },

    clu_yes_mt10:{
        text: "O modelo de clustering MiniBatch KMeans pode ser uma boa tentativa"
    },

    clu_yes_lt10: {
        text: "O modelo de clustering Kmeans pode apresentar um bom desempenho, também são boas tentativas os modelos Spectral Clustering e GMM",
    },

    clu_label_quantity_no:{
        text: "Quantos dados você tem?",
        buttons: [
            { text: "<10K", next: "clu_no_lt10" },
            { text: ">10K", next: "clu_no_mt10" }
        ]
    },

    clu_no_lt10:{
        text: 'O modelo de clustering MeanShift e VBGMM podem apresentar um bom desempenho'
    },

    clu_no_mt10:{
        text:'Não existe nenhum modelo na base de dados que aparente ter bom desempenho com a sua descrição'
    },

    //Classification
    labeled_data_yes:{
        text: 'Quantos dados você tem?',
        buttons: [
            { text: "<100K", next: "cla_lt100" },
            { text: ">100K", next: "cla_mt100" }
        ]
    },

    cla_lt100:{
        text: 'O modelo de classificação Linear SVC pode apresentar um bom resultado. Caso não seja suficiente, o seus dados são em texto?',
        buttons: [
            { text: "Sim", next: "cla_lt100_yes" },
            { text: "Não", next: "cla_lt100_no" }
        ]
    },

    cla_lt100_yes:{
        text: 'O modelo de classificação Naive Bayes pode apresentar um bom resultado'
    },

    cla_lt100_no:{
        text: 'O modelo de classificação KNeighbors Classifier pode apresentar um bom resultado. Também são boas tentativas os modelos SVC e Ensemble Classifiers'
    },

    cla_mt100:{
        text: 'O modelo de classificação SGD Classifier pode apresentar um bom resultado. Também pode-se considerar o modelo Kernel Approximation'
    },

    //Regression
    quantity_yes: {
        text: 'Quantos dados você tem?',
        buttons: [
            { text: "<100K", next: "reg_lt100" },
            { text: ">100K", next: "reg_mt100" }
        ]
    },
    reg_mt100:{
        text:'O modelo de regressão SGD Regressor pode apresentar um bom resultado'
    },

    reg_lt100: {
        text: 'Poucas caractéristicas dos dados são importantes?',
        buttons: [
            { text: "Sim", next: "reg_lt100_yes" },
            { text: "Não", next: "reg_lt100_no" }
        ]
    },
    reg_lt100_yes: {
        text: 'Os modelos de regressão Lasso e ElasticNet podem apresentar bons resultados'
    },

    reg_lt100_no:{
        text: 'Os modelos de regressão RidgeRegression e SVR podem apresentar bons resultados. Também pode-se considerar o modelo EnsembleRegressors'
    },

    //Dimensionality Reduction
    quantity_no_yes: {
        text: 'O modelo de redução de dimensão Randomized PCA pode apresentar um bom resultado. Caso não funcione, quanto modelos você tem?',
        buttons: [
            { text: "<10K", next: "dr_lt10" },
            { text: ">10K", next: "dr_mt10" }
        ]
    },

    dr_lt10: {
        text: 'Os modelos de redução de dimensão Isomap e Spectral Embedding podem apresentar bons resultados. Também pode-se considerar o modelo LLE'
    },

    dr_mt10: {
        text: 'O modelo de redução de dimensão Kernel Approximation pode apresentar um bom resultado.'
    }
};

function toggleHelp() {
    const chatContainer = document.getElementById('chat-container');
    const btn = document.querySelector('.btn');

    // Verifica o estado atual de exibição
    if (chatContainer.style.display === "none" || chatContainer.style.display === "") {
        chatContainer.style.display = "flex";  // Mostra o chat
        btn.classList.add('active');           // Adiciona classe 'active' para controle de estilo
        startChat();
    } else {
        chatContainer.style.display = "none";  // Esconde o chat
        btn.classList.remove('active');        // Remove a classe 'active'
        document.getElementById('chat-content').innerHTML = '';  // Limpa o conteúdo do chat
    }
}

function startChat() {
    showQuestion("start");
}

function showQuestion(key) {
    const question = questions[key];
    const chatContent = document.getElementById('chat-content');

    // Limpa o conteúdo anterior
    chatContent.innerHTML = '';

    // Adiciona a nova pergunta
    const questionElement = document.createElement('div');
    questionElement.innerHTML = `<p>${question.text}</p>`;
    chatContent.appendChild(questionElement);

    // Adiciona os botões
    if (question.buttons) {
        question.buttons.forEach(button => {
            const buttonElement = document.createElement('button');
            buttonElement.className = 'chat-button';
            buttonElement.innerText = button.text;
            buttonElement.onclick = () => showQuestion(button.next);
            chatContent.appendChild(buttonElement);
        });
    }

    // Rolagem automática para o final
    chatContent.scrollTop = chatContent.scrollHeight;
}

