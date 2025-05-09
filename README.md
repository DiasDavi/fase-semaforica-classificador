# fase-semaforica-classificador

## Dependências
A versão do python utilizado foi o Python 3.10.11

Recomendo criar um ambiente virtual para instalar as dependencias
```bash
python -m venv venv
```

para inicializar o ambiente virtual:
```bash
.\venv\Scripts\activate
```

Caso esteja utilizando GPU
```bash
pip install -r requirements/requirements-gpu.txt
```
Caso contrario
```bash
pip install -r requirements/requirements.txt
```

## Uso dos Scripts

### Treinamento do Modelo
O script `train.py` é responsável por treinar a rede neural com os dados processados.

### Previsão do imagens
O script `predict.py` é responsavel por realizar a previsão da classe da imagem
```bash
python predict.py --image CAMINHO_DA_IMAGEM
```

## Resultados
Os arquivos gerados após a execução dos scripts incluem:

* `output/class.h5`: Modelo treinado.
* `output/plot.png`: Gráficos do treinamento.