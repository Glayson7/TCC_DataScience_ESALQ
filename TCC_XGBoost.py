# Importação de Bibliotecas
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import learning_curve

# Supressão de Avisos Futuros
warnings.simplefilter(action="ignore", category=FutureWarning)

# ------------------------------
# Etapa 1: Carregamento dos Dados
# ------------------------------
print("Carregando o banco de dados...")
df = pd.read_excel("C:/Users/glays/Downloads/New Report (2) (1).xlsx")

# ----------------------------------------------
# Etapa 2: Pré-processamento e Limpeza de Dados
# ----------------------------------------------
print("Codificando variáveis categóricas...")
label_encoders = {}
relevant_columns = ["Defeito", "Objeto", "Mês/ano", "Prioridade"]
df_filtered = df[relevant_columns].dropna()

# Codificação de etiquetas para colunas categóricas
for col in ["Defeito", "Objeto", "Mês/ano"]:
    le = LabelEncoder()
    df_filtered[col] = le.fit_transform(df_filtered[col])
    label_encoders[col] = le

# Renomeando a coluna 'Prioridade' para 'Criticidade'
df_filtered.rename(columns={"Prioridade": "Criticidade"}, inplace=True)

# Codificando a variável alvo (Criticidade)
le = LabelEncoder()
y = le.fit_transform(df_filtered["Criticidade"])
label_encoders["Criticidade"] = le

# ------------------------------
# Etapa 3: Análise Exploratória de Dados
# ------------------------------
# Contagem de classes da variável alvo
criticidade_decoded = le.inverse_transform(y)
criticidade_count_decoded = (
    pd.Series(criticidade_decoded).value_counts().reset_index()
)
criticidade_count_decoded.columns = ["Criticidade", "Contagem"]
print("Tabela de Contagem de Criticidade:")
print(criticidade_count_decoded)

# Gráfico de Barras para Contagem de Criticidade
plt.figure(figsize=(8, 6))
plt.bar(
    criticidade_count_decoded["Criticidade"],
    criticidade_count_decoded["Contagem"],
    color="blue",
)
plt.xlabel("Níveis de Criticidade")
plt.ylabel("Contagem")
plt.title("Distribuição dos Níveis de Criticidade (C0, C1, C2)")
plt.xticks(rotation=0)
plt.show()
# Plotting the distribution of classes before SMOTE
plt.figure()
sns.countplot(y, palette="viridis")
plt.title("Class Distribution Before SMOTE")
plt.show()

# Etapa 3.5: Definição de Conjunto de Features
# -------------------------------------
# Separação das variáveis de características e alvo
print("Separando variáveis de características e alvo...")
X = df_filtered[["Defeito", "Objeto", "Mês/ano"]]

# Etapa 4: Balanceamento de Dados
# ------------------------------
# Aplicação de SMOTE para balanceamento de classes
print("Aplicando SMOTE...")
smote = SMOTE()
X_res, y_res = smote.fit_resample(X, y)

# Plotting the distribution of classes after SMOTE
plt.figure()
sns.countplot(y_res, palette="viridis")
plt.title("Class Distribution After SMOTE")
plt.show()

# Etapa 5: Separação de Dados
# ------------------------------
# Divisão do conjunto de dados em treinamento e teste
print("Dividindo dados em treinamento e teste...")
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, stratify=y_res, random_state=42
)

# ------------------------------
# Etapa 6: Otimização de Modelo
# ------------------------------
# Parâmetros para Grid Search
param_grid = {
    "learning_rate": [0.01, 0.1, 0.2],
    "n_estimators": [50, 100, 150],
}

# Inicializar o Grid Search com validação cruzada
print("Realizando Grid Search com validação cruzada...")
grid_search = GridSearchCV(
    XGBClassifier(
        objective="multi:softprob", num_class=len(np.unique(y_train))
    ),
    param_grid,
    cv=5,
)

# Ajustar aos dados
grid_search.fit(X_train, y_train)

# ------------------------------
# Etapa 7: Avaliação de Modelo
# ------------------------------
# Extração e impressão dos melhores parâmetros
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print(f"Melhores parâmetros para XGBoost: {best_params}")

# Fazendo previsões com o melhor modelo
print("Fazendo previsões com o melhor modelo...")
y_pred = best_model.predict(X_test)

# Decodificar as categorias para P0, P1 e P2 após a predição
y_pred_decoded = le.inverse_transform(y_pred)
print("Predições Decodificadas:")
print(y_pred_decoded)

# ------------------------------
# Etapa 8: Métricas e Visualizações
# ------------------------------
# Cálculo e impressão das métricas de desempenho
print(f"Acurácia: {accuracy_score(y_test, y_pred)}")
print(f"Matriz de Confusão: \n{confusion_matrix(y_test, y_pred)}")
print(f"Precisão: {precision_score(y_test, y_pred, average='weighted')}")
print(f"Recall: {recall_score(y_test, y_pred, average='weighted')}")
print(f"F1-Score: {f1_score(y_test, y_pred, average='weighted')}")

# Convertendo as previsões para os rótulos originais
y_pred_decoded = le.inverse_transform(y_pred)

# Contando o número de ocorrências para cada nível de criticidade previsto (C0, C1, C2)
predicted_criticidade_count = (
    pd.Series(y_pred_decoded).value_counts().reset_index()
)
predicted_criticidade_count.columns = ["Criticidade Prevista", "Contagem"]

# Gráfico de Barras para Contagem de Criticidade Prevista
plt.figure(figsize=(8, 6))
sns.barplot(
    x="Criticidade Prevista",
    y="Contagem",
    data=predicted_criticidade_count,
    palette="viridis",
)
plt.xlabel("Níveis de Criticidade Prevista")
plt.ylabel("Contagem")
plt.title("Distribuição dos Níveis de Criticidade Prevista (C0, C1, C2)")
plt.xticks(rotation=0)
plt.show()

# ------------------------------
# Etapa 9: Análise de Importância de Recursos
# ------------------------------
# Extração da importância dos recursos e plotagem
feature_importances = best_model.feature_importances_
feature_names = X.columns

plt.figure()
sns.barplot(x=feature_importances, y=feature_names, palette="viridis")
plt.title("Feature Importance")
plt.show()

# ------------------------------
# Etapa 10: Análise de Matriz de Confusão
# ------------------------------
# Heatmap da Matriz de Confusão
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="g")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# ------------------------------
# Etapa 11: Avaliação AUC-ROC
# ------------------------------
# Inclusão da métrica AUC-ROC para avaliação multiclasse
y_test_binarized = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_test_binarized.shape[1]

# Utilizando OneVsRestClassifier para treinar o modelo XGBoost
classifier = OneVsRestClassifier(best_model)
y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

# Calculando a curva ROC e a área ROC para cada classe
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plotando a curva ROC
plt.figure()
for i in range(n_classes):
    plt.plot(
        fpr[i],
        tpr[i],
        label=f"Curva ROC da classe {i} (área = {roc_auc[i]:.2f})",
    )
plt.plot([0, 1], [0, 1], "k--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Taxa de Falsos Positivos")
plt.ylabel("Taxa de Verdadeiros Positivos")
plt.title("Característica Operacional do Receptor")
plt.legend(loc="lower right")
plt.show()

print(f"AUC-ROC para multi-classe: {roc_auc}")

# ------------------------------
# Etapa 12: Gráfico de Aprendizado
# ------------------------------
# Plotagem do gráfico de aprendizado para avaliar o desempenho do modelo
print("Plotando o gráfico de aprendizado...")

train_sizes, train_scores, test_scores = learning_curve(
    best_model,
    X_res,
    y_res,
    cv=5,
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    verbose=0,
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.fill_between(
    train_sizes,
    train_mean - train_std,
    train_mean + train_std,
    alpha=0.1,
    color="r",
)
plt.fill_between(
    train_sizes,
    test_mean - test_std,
    test_mean + test_std,
    alpha=0.1,
    color="g",
)
plt.plot(train_sizes, train_mean, "o-", color="r", label="Training score")
plt.plot(
    train_sizes, test_mean, "o-", color="g", label="Cross-validation score"
)
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy Score")
plt.legend(loc="best")
plt.show()


# ------------------------------
# Etapa 13: Resumo Estatístico
# ------------------------------
# Distribuição de Classes Antes e Depois do SMOTE
class_distribution_before = pd.Series(y).value_counts().reset_index()
class_distribution_before.columns = ["Classe", "Contagem Antes do SMOTE"]

class_distribution_after = pd.Series(y_res).value_counts().reset_index()
class_distribution_after.columns = ["Classe", "Contagem Depois do SMOTE"]

class_distribution = pd.merge(
    class_distribution_before, class_distribution_after, on="Classe"
)
print("1. Distribuição de Classes Antes e Depois do SMOTE:")
print(class_distribution)

#  Melhores Parâmetros do Modelo
best_params_df = pd.DataFrame([best_params])
best_params_df.index = ["Melhores Parâmetros"]
print("\n2. Melhores Parâmetros do Modelo:")
print(best_params_df)

# Métricas de Avaliação do Modelo
metrics_df = pd.DataFrame(
    {
        "Métrica": ["Acurácia", "Precisão", "Recall", "F1-Score"],
        "Valor": [
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred, average="weighted"),
            recall_score(y_test, y_pred, average="weighted"),
            f1_score(y_test, y_pred, average="weighted"),
        ],
    }
)
print("\n3. Métricas de Avaliação do Modelo:")
print(metrics_df)

# Importância das Características
feature_importance_df = pd.DataFrame(
    {
        "Característica": X.columns,
        "Importância": best_model.feature_importances_,
    }
).sort_values(by="Importância", ascending=False)
print("\n4. Importância das Características:")
print(feature_importance_df)


# ------------------------------
# Etapa 14: Visualizações Finais
# ------------------------------
# Gráfico da Distribuição de Classes Antes e Depois do SMOTE
class_distribution.plot(
    x="Classe",
    y=["Contagem Antes do SMOTE", "Contagem Depois do SMOTE"],
    kind="bar",
    title="Distribuição de Classes Antes e Depois do SMOTE",
)
plt.show()

# Gráfico das Métricas de Avaliação do Modelo
metrics_df.plot(
    x="Métrica", y="Valor", kind="bar", title="Métricas de Avaliação do Modelo"
)
plt.show()

# Importância das Características
feature_importance_df.plot(
    x="Característica",
    y="Importância",
    kind="bar",
    title="Importância das Características",
)
plt.show()
