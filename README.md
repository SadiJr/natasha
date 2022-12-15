# Natasha
---

Repositório para armazenar o código da aplicação _web_ gerada durante meu projeto de TCC no curso de Sistemas de Informação, da Universidade Federal de Santa Catarina.

---

## Instalação
---

Os passos necessários para utilizar a interface _web_ criada consistem em:

1. Faça o clone do repositório:  
`git clone https://github.com/SadiJr/natasha.git`

2. Navegue até o repositório clonado:  
`cd natasha`

3. Crie um _virtualenv_:  
`python -m venv env`

4. Ative o _virtualenv_ criada:  
`source env/bin/activate`

5. Instale as dependências necessárias:  
`pip install -r requirements.txt`

6. Ainda com o _virtualenv_ ativo, execute os seguintes comandos:
```bash
-> % python
Python 3.10.8 (main, Nov  1 2022, 14:18:21) [GCC 12.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import nltk
>>> nltk.download('stopwords')
>>> nltk.download('punkt')
>>> nltk.download('wordnet')
>>> nltk.download('omw-1.4')
```

Esse passo é necessário para baixar algumas dependências necessárias da biblioteca `nltk`, usada para o processamento dos textos.

7. Saia do interpretador interativo do Python e, ainda com o _virtualenv_ ativo, execute o seguinte comando:
```bash
-> % python dashboard/app.py
```

Você deverá obter uma saída semelhante à essa:
```bash
Dash is running on http://0.0.0.0:8080/

 * Serving Flask app 'app'
 * Debug mode: on
```

8. Agora, basta acessar, via _browser_ de sua preferência, a aplicação _web_.

---

## Colaborando

Sinta-se livre para colaborar com esse projeto, seja reportando _bugs_ ou sugestões de _features_ através de _issues_, ou abrindo _pull requests_ com melhorias.

---
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/SadiJr/natasha/blob/main/LICENSE)

