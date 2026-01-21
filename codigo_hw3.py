import os
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, pearsonr
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler, PowerTransformer, LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                           precision_score, recall_score, f1_score, roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings('ignore')

# ============================ CONFIGURAÇÃO ============================

RED_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
WHITE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
OUTDIR = "outputs_hw3"
FIGDIR = "figures_hw3"
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Tradução dos nomes das características
FEATURE_NAMES_PT = {
    'fixed acidity': 'Acidez Fixa',
    'volatile acidity': 'Acidez Volátil', 
    'citric acid': 'Ácido Cítrico',
    'residual sugar': 'Açúcar Residual',
    'chlorides': 'Cloretos',
    'free sulfur dioxide': 'Dióxido de Enxofre Livre',
    'total sulfur dioxide': 'Dióxido de Enxofre Total',
    'density': 'Densidade',
    'pH': 'pH',
    'sulphates': 'Sulfatos',
    'alcohol': 'Álcool',
    'quality': 'Qualidade'
}

FEATURE_NAMES_EN = list(FEATURE_NAMES_PT.keys())[:-1] 
FEATURE_NAMES_PT_LIST = [FEATURE_NAMES_PT[name] for name in FEATURE_NAMES_EN]

# ============================ FUNÇÕES AUXILIARES ============================

def load_and_combine_data():
    """Carrega e combina os datasets de vinho tinto e branco"""
    print("Carregando dados...")
    
    def download_if_needed(url, filename):
        if not os.path.exists(filename):
            urllib.request.urlretrieve(url, filename)
        return pd.read_csv(filename, sep=';')
    
    df_red = download_if_needed(RED_URL, "winequality-red.csv")
    df_white = download_if_needed(WHITE_URL, "winequality-white.csv")
    
    df_red['tipo_vinho'] = 'Tinto'
    df_white['tipo_vinho'] = 'Branco'
    df = pd.concat([df_red, df_white], ignore_index=True)
    df = df.rename(columns=FEATURE_NAMES_PT)
    
    print(f"✓ Dados carregados: {df.shape[0]} observações, {df.shape[1] - 2} características")
    return df

def transform_target_to_binary(df, threshold=7):
    """Transforma a variável qualidade em binária (1=alta qualidade, 0=baixa qualidade)"""
    df['Classe_Binaria'] = df['Qualidade'].apply(lambda x: 1 if x >= threshold else 0)
    
    print(f"\nTransformação da variável alvo (threshold={threshold}):")
    print(f"  • Classe 0 (Qualidade < {threshold}): {sum(df['Classe_Binaria'] == 0)} observações")
    print(f"  • Classe 1 (Qualidade ≥ {threshold}): {sum(df['Classe_Binaria'] == 1)} observações")
    print(f"  • Proporção: {sum(df['Classe_Binaria'] == 1)/len(df)*100:.2f}% de vinhos de alta qualidade")
    return df

def calculate_classification_metrics(y_true, y_pred, y_prob=None):
    """Calcula métricas de classificação"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_prob is not None and len(np.unique(y_true)) == 2:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, model_name, normalize=False):
    """Plota matriz de confusão"""
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=['Baixa (0)', 'Alta (1)'], 
                yticklabels=['Baixa (0)', 'Alta (1)'])
    plt.title(f'Matriz de Confusão - {model_name}')
    plt.ylabel('Classe Real')
    plt.xlabel('Classe Predita')
    
    filename = f'confusion_matrix_{model_name.lower().replace(" ", "_")}'
    if normalize:
        filename += '_normalized'
    plt.savefig(os.path.join(FIGDIR, f'{filename}.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_true, y_prob, model_name):
    """Plota curva ROC"""
    if len(np.unique(y_true)) != 2:
        print(f"  Não é possível plotar ROC para {model_name} (não é problema binário)")
        return
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title(f'Curva ROC - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    filename = f'roc_curve_{model_name.lower().replace(" ", "_")}'
    plt.savefig(os.path.join(FIGDIR, f'{filename}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return roc_auc

# ============================ ANÁLISE EXPLORATÓRIA ============================

def analise_exploratoria_classificacao(df):
    """Realiza análise exploratória para classificação"""
    print("\n" + "="*60)
    print("ANÁLISE EXPLORATÓRIA - CLASSIFICAÇÃO")
    print("="*60)
    
    X = df[FEATURE_NAMES_PT_LIST].values
    y = df['Classe_Binaria'].values
    
    print(f"\n1. Distribuição das classes:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"   • Classe {cls}: {count} observações ({count/len(y)*100:.2f}%)")
    
    print(f"\n2. Estatísticas por classe:")
    
    df_class_0 = df[df['Classe_Binaria'] == 0]
    df_class_1 = df[df['Classe_Binaria'] == 1]
    
    stats_by_class = []
    for col in FEATURE_NAMES_PT_LIST:
        stats_by_class.append([
            col,
            df_class_0[col].mean(), df_class_0[col].std(),
            df_class_1[col].mean(), df_class_1[col].std(),
            df_class_1[col].mean() - df_class_0[col].mean()
        ])
    
    stats_df = pd.DataFrame(stats_by_class, columns=[
        'Característica', 'Média (Classe 0)', 'DP (Classe 0)',
        'Média (Classe 1)', 'DP (Classe 1)', 'Diferença Médias'
    ])
    stats_df.to_csv(os.path.join(OUTDIR, 'estatisticas_por_classe.csv'), index=False, encoding='utf-8')
    
    stats_df['|Diferença|'] = abs(stats_df['Diferença Médias'])
    top5_diff = stats_df.nlargest(5, '|Diferença|')
    
    print(f"\n3. Top 5 características com maior diferença entre classes:")
    for i, row in top5_diff.iterrows():
        print(f"   {i+1}. {row['Característica']}: {row['Diferença Médias']:.3f}")
    
    # Visualizações
    print(f"\n4. Gerando visualizações...")
    
    # Boxplot das características por classe
    fig, axes = plt.subplots(4, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, col in enumerate(FEATURE_NAMES_PT_LIST):
        ax = axes[i]
        data = [df_class_0[col].values, df_class_1[col].values]
        ax.boxplot(data, labels=['Baixa (0)', 'Alta (1)'])
        ax.set_title(col)
        ax.set_ylabel('Valor')
        ax.grid(alpha=0.3)
    
    for i in range(len(FEATURE_NAMES_PT_LIST), len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle('Distribuição das Características por Classe de Qualidade', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'boxplot_por_classe.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Histograma das classes
    plt.figure(figsize=(8, 6))
    plt.bar(['Baixa (0)', 'Alta (1)'], [len(df_class_0), len(df_class_1)], 
            color=['lightcoral', 'lightgreen'], alpha=0.8)
    plt.title('Distribuição das Classes')
    plt.ylabel('Número de Observações')
    for i, v in enumerate([len(df_class_0), len(df_class_1)]):
        plt.text(i, v + max([len(df_class_0), len(df_class_1)])*0.01, 
                f'{v}\n({v/len(df)*100:.1f}%)', 
                ha='center', va='bottom')
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'distribuicao_classes.png'), dpi=150)
    plt.close()
    
    print("✓ Análise exploratória concluída")
    return X, y, stats_df

# ============================ PRÉ-PROCESSAMENTO ============================

def preprocessamento_classificacao(X_train, X_test, y_train, y_test):
    print("\n" + "="*60)
    print("PRÉ-PROCESSAMENTO DOS DADOS")
    print("="*60)
    
    preprocessor = Pipeline([
        ('power_transform', PowerTransformer(method='yeo-johnson')),
        ('scaler', StandardScaler())
    ])
    
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Análise após transformação
    skewness_after = skew(X_train_transformed, axis=0)
    skewness_before = skew(X_train, axis=0)
    
    reducao_media = np.mean(100 * (1 - np.abs(skewness_after) / np.abs(np.where(skewness_before != 0, skewness_before, 0.01))))
    
    print(f"\n1. Redução média de skewness: {reducao_media:.1f}%")
    print(f"2. Dados transformados: {X_train_transformed.shape}")
    
    return X_train_transformed, X_test_transformed, preprocessor


# ============================ MODELOS LINEARES ============================

def modelo_linear_logistic(X_train, X_test, y_train, y_test):
    """Modelo Linear: Regressão Logística"""
    print("\n" + "="*60)
    print("MODELO LINEAR: REGRESSÃO LOGÍSTICA")
    print("="*60)
    
    # Ajuste de hiperparâmetros
    param_grid = {
        'C': np.logspace(-3, 3, 10),
        'penalty': ['l2'],
        'solver': ['lbfgs']
    }
    
    print("1. Ajustando hiperparâmetros com validação cruzada...")
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    grid_search = GridSearchCV(
        LogisticRegression(random_state=RANDOM_SEED, max_iter=1000),
        param_grid,
        cv=kf,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print(f"   • Melhores parâmetros: C={best_params['C']:.4f}")
    
    # Treinar modelo final
    logreg = LogisticRegression(
        C=best_params['C'],
        penalty=best_params['penalty'],
        solver=best_params['solver'],
        random_state=RANDOM_SEED,
        max_iter=1000
    )
    logreg.fit(X_train, y_train)
    
    # Previsões
    y_pred = logreg.predict(X_test)
    y_prob = logreg.predict_proba(X_test)[:, 1]
    
    # Métricas
    metrics = calculate_classification_metrics(y_test, y_pred, y_prob)
    
    print(f"\n2. Resultados no conjunto de teste:")
    print(f"   • Acurácia: {metrics['accuracy']:.4f}")
    print(f"   • Precisão: {metrics['precision']:.4f}")
    print(f"   • Recall: {metrics['recall']:.4f}")
    print(f"   • F1-Score: {metrics['f1']:.4f}")
    if 'roc_auc' in metrics:
        print(f"   • AUC-ROC: {metrics['roc_auc']:.4f}")
    
    # Matriz de confusão
    plot_confusion_matrix(y_test, y_pred, "Regressão Logística")
    plot_confusion_matrix(y_test, y_pred, "Regressão Logística", normalize=True)
    
    # Curva ROC
    if 'roc_auc' in metrics:
        plot_roc_curve(y_test, y_prob, "Regressão Logística")
    
    # Importância das características
    if hasattr(logreg, 'coef_'):
        feature_importance = pd.DataFrame({
            'Característica': FEATURE_NAMES_PT_LIST,
            'Coeficiente': logreg.coef_[0],
            '|Coeficiente|': abs(logreg.coef_[0])
        })
        feature_importance = feature_importance.sort_values('|Coeficiente|', ascending=False)
        feature_importance.to_csv(os.path.join(OUTDIR, 'logreg_feature_importance.csv'), 
                                  index=False, encoding='utf-8')
        
        print(f"\n3. Top 5 características mais importantes:")
        for i, row in feature_importance.head(5).iterrows():
            print(f"   {i+1}. {row['Característica']}: {row['Coeficiente']:.4f}")
    
    resultados = {
        'model': logreg,
        'params': best_params,
        'metrics': metrics,
        'y_pred': y_pred,
        'y_prob': y_prob
    }
    
    return resultados

def modelo_linear_lda(X_train, X_test, y_train, y_test):
    """Modelo Linear: Linear Discriminant Analysis"""
    print("\n" + "="*60)
    print("MODELO LINEAR: ANÁLISE DISCRIMINANTE LINEAR (LDA)")
    print("="*60)
    
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    
    # Previsões
    y_pred = lda.predict(X_test)
    y_prob = lda.predict_proba(X_test)[:, 1]
    
    # Métricas
    metrics = calculate_classification_metrics(y_test, y_pred, y_prob)
    
    print(f"\n1. Resultados no conjunto de teste:")
    print(f"   • Acurácia: {metrics['accuracy']:.4f}")
    print(f"   • Precisão: {metrics['precision']:.4f}")
    print(f"   • Recall: {metrics['recall']:.4f}")
    print(f"   • F1-Score: {metrics['f1']:.4f}")
    if 'roc_auc' in metrics:
        print(f"   • AUC-ROC: {metrics['roc_auc']:.4f}")
    
    # Matriz de confusão
    plot_confusion_matrix(y_test, y_pred, "LDA")
    plot_confusion_matrix(y_test, y_pred, "LDA", normalize=True)
    
    # Curva ROC
    if 'roc_auc' in metrics:
        plot_roc_curve(y_test, y_prob, "LDA")
    
    # coeficientes do discriminante
    if hasattr(lda, 'coef_'):
        feature_importance = pd.DataFrame({
            'Característica': FEATURE_NAMES_PT_LIST,
            'Coeficiente': lda.coef_[0]
        })
        feature_importance['|Coeficiente|'] = abs(feature_importance['Coeficiente'])
        feature_importance = feature_importance.sort_values('|Coeficiente|', ascending=False)
        feature_importance.to_csv(os.path.join(OUTDIR, 'lda_feature_importance.csv'), 
                                  index=False, encoding='utf-8')
        
        print(f"\n2. Top 5 características mais importantes:")
        for i, row in feature_importance.head(5).iterrows():
            print(f"   {i+1}. {row['Característica']}: {row['Coeficiente']:.4f}")
    
    resultados = {
        'model': lda,
        'metrics': metrics,
        'y_pred': y_pred,
        'y_prob': y_prob
    }
    
    return resultados

# ============================ MODELOS NÃO LINEARES ============================

def modelo_nao_linear_svm(X_train, X_test, y_train, y_test):
    """Modelo Não Linear: Support Vector Machine"""
    print("\n" + "="*60)
    print("MODELO NÃO LINEAR: SUPPORT VECTOR MACHINE (SVM)")
    print("="*60)
    
    # Ajuste de hiperparâmetros
    param_dist = {
        'C': loguniform(1e-2, 1e3),
        'gamma': loguniform(1e-4, 1e1),
        'kernel': ['rbf']
}

    print("1. Ajustando hiperparâmetros com BUSCA ALEATÓRIA (RandomizedSearchCV)...")
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)

    random_search = RandomizedSearchCV(
        estimator=SVC(probability=True, random_state=RANDOM_SEED, cache_size=700),
        param_distributions=param_dist,
        n_iter=15,               
        cv=kf,
        scoring='accuracy',
        n_jobs=-1,
        random_state=RANDOM_SEED,
        verbose=0
    )
    random_search.fit(X_train, y_train)
    best_params = random_search.best_params_

    # Treinar modelo final
    svm = SVC(
        C=best_params['C'],
        gamma=best_params['gamma'],
        kernel=best_params['kernel'],
        probability=True,
        random_state=RANDOM_SEED
    )
    svm.fit(X_train, y_train)
    
    # Previsões
    y_pred = svm.predict(X_test)
    y_prob = svm.predict_proba(X_test)[:, 1]
    
    # Métricas
    metrics = calculate_classification_metrics(y_test, y_pred, y_prob)
    
    print(f"\n2. Resultados no conjunto de teste:")
    print(f"   • Acurácia: {metrics['accuracy']:.4f}")
    print(f"   • Precisão: {metrics['precision']:.4f}")
    print(f"   • Recall: {metrics['recall']:.4f}")
    print(f"   • F1-Score: {metrics['f1']:.4f}")
    if 'roc_auc' in metrics:
        print(f"   • AUC-ROC: {metrics['roc_auc']:.4f}")
    
    # Matriz de confusão
    plot_confusion_matrix(y_test, y_pred, "SVM (RBF)")
    plot_confusion_matrix(y_test, y_pred, "SVM (RBF)", normalize=True)
    
    # Curva ROC
    if 'roc_auc' in metrics:
        plot_roc_curve(y_test, y_prob, "SVM (RBF)")
    
    resultados = {
        'model': svm,
        'params': best_params,
        'metrics': metrics,
        'y_pred': y_pred,
        'y_prob': y_prob
    }
    
    return resultados

def modelo_nao_linear_knn(X_train, X_test, y_train, y_test):
    """Modelo Não Linear: k-Nearest Neighbors"""
    print("\n" + "="*60)
    print("MODELO NÃO LINEAR: k-NEAREST NEIGHBORS (k-NN)")
    print("="*60)
    
    # Ajuste de hiperparâmetros
    param_grid = {
        'n_neighbors': range(3, 21, 2),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
    print("1. Ajustando hiperparâmetros com validação cruzada...")
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    grid_search = GridSearchCV(
        KNeighborsClassifier(),
        param_grid,
        cv=kf,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print(f"   • Melhores parâmetros: k={best_params['n_neighbors']}, "
          f"weights={best_params['weights']}, metric={best_params['metric']}")
    
    # Treinar modelo final
    knn = KNeighborsClassifier(
        n_neighbors=best_params['n_neighbors'],
        weights=best_params['weights'],
        metric=best_params['metric']
    )
    knn.fit(X_train, y_train)
    
    # Previsões
    y_pred = knn.predict(X_test)
    y_prob = knn.predict_proba(X_test)[:, 1]
    
    # Métricas
    metrics = calculate_classification_metrics(y_test, y_pred, y_prob)
    
    print(f"\n2. Resultados no conjunto de teste:")
    print(f"   • Acurácia: {metrics['accuracy']:.4f}")
    print(f"   • Precisão: {metrics['precision']:.4f}")
    print(f"   • Recall: {metrics['recall']:.4f}")
    print(f"   • F1-Score: {metrics['f1']:.4f}")
    if 'roc_auc' in metrics:
        print(f"   • AUC-ROC: {metrics['roc_auc']:.4f}")
    
    # Matriz de confusão
    plot_confusion_matrix(y_test, y_pred, "k-NN")
    plot_confusion_matrix(y_test, y_pred, "k-NN", normalize=True)
    
    # Curva ROC
    if 'roc_auc' in metrics:
        plot_roc_curve(y_test, y_prob, "k-NN")
    
    resultados = {
        'model': knn,
        'params': best_params,
        'metrics': metrics,
        'y_pred': y_pred,
        'y_prob': y_prob
    }
    
    return resultados

def modelo_nao_linear_nn(X_train, X_test, y_train, y_test):
    """Modelo Não Linear: Neural Network"""
    print("\n" + "="*60)
    print("MODELO NÃO LINEAR: REDE NEURAL (MLP)")
    print("="*60)
    
    # Ajuste de hiperparâmetros simplificado
    param_grid = {
        'hidden_layer_sizes': [(50, 25), (100, 50), (100, 50, 25)],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01]
    }
    
    print("1. Ajustando hiperparâmetros com validação cruzada...")
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)  # Reduzido para velocidade
    
    grid_search = GridSearchCV(
        MLPClassifier(random_state=RANDOM_SEED, max_iter=500, early_stopping=True),
        param_grid,
        cv=kf,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print(f"   • Melhores parâmetros: hidden_layer_sizes={best_params['hidden_layer_sizes']}, "
          f"alpha={best_params['alpha']}, learning_rate_init={best_params['learning_rate_init']}")
    
    # Treinar modelo final
    mlp = MLPClassifier(
        hidden_layer_sizes=best_params['hidden_layer_sizes'],
        alpha=best_params['alpha'],
        learning_rate_init=best_params['learning_rate_init'],
        random_state=RANDOM_SEED,
        max_iter=500,
        early_stopping=True,
        verbose=False
    )
    mlp.fit(X_train, y_train)
    
    # Previsões
    y_pred = mlp.predict(X_test)
    y_prob = mlp.predict_proba(X_test)[:, 1]
    
    # Métricas
    metrics = calculate_classification_metrics(y_test, y_pred, y_prob)
    
    print(f"\n2. Resultados no conjunto de teste:")
    print(f"   • Acurácia: {metrics['accuracy']:.4f}")
    print(f"   • Precisão: {metrics['precision']:.4f}")
    print(f"   • Recall: {metrics['recall']:.4f}")
    print(f"   • F1-Score: {metrics['f1']:.4f}")
    if 'roc_auc' in metrics:
        print(f"   • AUC-ROC: {metrics['roc_auc']:.4f}")
    print(f"   • Épocas treinadas: {mlp.n_iter_}")
    
    # Matriz de confusão
    plot_confusion_matrix(y_test, y_pred, "Rede Neural")
    plot_confusion_matrix(y_test, y_pred, "Rede Neural", normalize=True)
    
    # Curva ROC
    if 'roc_auc' in metrics:
        plot_roc_curve(y_test, y_prob, "Rede Neural")
    
    # Curva de aprendizado
    if hasattr(mlp, 'loss_curve_'):
        plt.figure(figsize=(10, 6))
        plt.plot(mlp.loss_curve_, 'b-', linewidth=2)
        plt.xlabel('Época')
        plt.ylabel('Loss')
        plt.title('Curva de Aprendizado da Rede Neural')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGDIR, 'nn_loss_curve.png'), dpi=150)
        plt.close()
        print("   • Curva de aprendizado salva")
    
    resultados = {
        'model': mlp,
        'params': best_params,
        'metrics': metrics,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'n_iter': mlp.n_iter_
    }
    
    return resultados

# ============================ ANÁLISE COMPARATIVA ============================

def analise_comparativa_modelos(resultados, y_test_real):
    """Realiza análise comparativa entre todos os modelos"""
    print("\n" + "="*60)
    print("ANÁLISE COMPARATIVA FINAL")
    print("="*60)
    
    tabela_comparativa = []
    
    for nome, res in resultados.items():
        met = res['metrics']
        linha = [
            nome.upper(),
            f"{met['accuracy']:.4f}",
            f"{met['precision']:.4f}",
            f"{met['recall']:.4f}",
            f"{met['f1']:.4f}"
        ]
        
        if 'roc_auc' in met:
            linha.append(f"{met['roc_auc']:.4f}")
        else:
            linha.append("N/A")
        
        if 'params' in res:
            if nome == 'logistic':
                linha.append(f"C={res['params']['C']:.2f}")
            elif nome == 'svm':
                linha.append(f"C={res['params']['C']:.2f}, γ={res['params']['gamma']:.3f}")
            elif nome == 'knn':
                linha.append(f"k={res['params']['n_neighbors']}")
            elif nome == 'nn':
                linha.append(f"layers={res['params']['hidden_layer_sizes']}")
            else:
                linha.append("-")
        else:
            linha.append("-")
        
        tabela_comparativa.append(linha)
    
    tabela_comparativa.sort(key=lambda x: float(x[1]), reverse=True)
    
    headers = ['Modelo', 'Acurácia', 'Precisão', 'Recall', 'F1-Score', 'AUC-ROC', 'Parâmetros']
    print("\nCOMPARAÇÃO DE DESEMPENHO (conjunto de teste):")
    print(tabulate(tabela_comparativa, headers=headers, tablefmt='simple', stralign='center'))
    
    df_comparacao = pd.DataFrame(tabela_comparativa, columns=headers)
    df_comparacao.to_csv(os.path.join(OUTDIR, 'comparacao_modelos.csv'), index=False, encoding='utf-8')
    
    # Gráfico comparativo
    modelos = [linha[0] for linha in tabela_comparativa]
    acuracias = [float(linha[1]) for linha in tabela_comparativa]
    
    x = np.arange(len(modelos))
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(x, acuracias, color=['skyblue', 'lightgreen', 'salmon', 'gold', 'violet'][:len(modelos)])
    ax.set_xlabel('Modelo')
    ax.set_ylabel('Acurácia')
    ax.set_title('Comparação de Acurácia entre Modelos')
    ax.set_xticks(x)
    ax.set_xticklabels(modelos, rotation=45, ha='right')
    ax.set_ylim([0, 1])
    
    for bar, acc in zip(bars, acuracias):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'comparacao_acuracia.png'), dpi=150)
    plt.close()
    
    melhor_idx = np.argmax(acuracias)
    melhor_modelo = modelos[melhor_idx]
    melhor_acuracia = acuracias[melhor_idx]
    
    print(f"\n✓ MELHOR MODELO: {melhor_modelo} (Acurácia: {melhor_acuracia:.4f})")
    
    # ANÁLISE DOS TIPOS DE ERRO - CORRIGIDA
    print(f"\nANÁLISE DOS TIPOS DE ERRO (Matrizes de Confusão):")
    for nome, res in resultados.items():
        y_pred = res['y_pred']
        
        cm = confusion_matrix(y_test_real, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print(f"\n{nome.upper()}:")
        print(f"  • Verdadeiros Positivos (TP): {tp}")
        print(f"  • Falsos Positivos (FP): {fp}")
        print(f"  • Verdadeiros Negativos (TN): {tn}")
        print(f"  • Falsos Negativos (FN): {fn}")
        
        if fp > fn:
            print(f"  → O modelo tende a classificar vinhos ruins como bons (FP alto)")
        elif fn > fp:
            print(f"  → O modelo tende a classificar vinhos bons como ruins (FN alto)")
        else:
            print(f"  → O modelo tem equilíbrio entre FP e FN")
    
    return df_comparacao
    

# ============================ FUNÇÃO PRINCIPAL ============================


def main():
    print("="*60)
    print("HW3 - MODELOS DE CLASSIFICAÇÃO PARA QUALIDADE DO VINHO")
    print("="*60)
    
    # 1. Carregar dados
    df = load_and_combine_data()
    
    # 2. Transformar variável alvo em binária
    df = transform_target_to_binary(df, threshold=7)
    
    # 3. Análise exploratória
    X, y, stats_df = analise_exploratoria_classificacao(df)
    
    # 4. Divisão treino-teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y
    )
    print(f"\nDivisão dos dados (estratificada):")
    print(f"  • Treino: {X_train.shape[0]} observações")
    print(f"    - Classe 0: {sum(y_train == 0)} ({sum(y_train == 0)/len(y_train)*100:.1f}%)")
    print(f"    - Classe 1: {sum(y_train == 1)} ({sum(y_train == 1)/len(y_train)*100:.1f}%)")
    print(f"  • Teste:  {X_test.shape[0]} observações")
    print(f"    - Classe 0: {sum(y_test == 0)} ({sum(y_test == 0)/len(y_test)*100:.1f}%)")
    print(f"    - Classe 1: {sum(y_test == 1)} ({sum(y_test == 1)/len(y_test)*100:.1f}%)")
    
    # 5. Pré-processamento
    X_train_transformed, X_test_transformed, preprocessor = preprocessamento_classificacao(
        X_train, X_test, y_train, y_test
    )
    
    # 6. Modelagem
    print("\n" + "="*60)
    print("INICIANDO MODELAGEM DE CLASSIFICAÇÃO")
    print("="*60)
    
    todos_resultados = {}
    
    # Modelos lineares
    print("\n>>> MODELOS LINEARES <<<")
    resultados_logistic = modelo_linear_logistic(X_train_transformed, X_test_transformed, y_train, y_test)
    todos_resultados['logistic'] = resultados_logistic
    
    resultados_lda = modelo_linear_lda(X_train_transformed, X_test_transformed, y_train, y_test)
    todos_resultados['lda'] = resultados_lda
    
    # Modelos não lineares
    print("\n>>> MODELOS NÃO LINEARES <<<")
    
    # SVM
    resultados_svm = modelo_nao_linear_svm(X_train_transformed, X_test_transformed, y_train, y_test)
    todos_resultados['svm'] = resultados_svm
    
    # k-NN
    resultados_knn = modelo_nao_linear_knn(X_train_transformed, X_test_transformed, y_train, y_test)
    todos_resultados['knn'] = resultados_knn
    
    # Rede Neural
    resultados_nn = modelo_nao_linear_nn(X_train_transformed, X_test_transformed, y_train, y_test)
    todos_resultados['nn'] = resultados_nn
    
    # 7. Análise comparativa
    df_comparacao = analise_comparativa_modelos(todos_resultados, y_test)
    
    # 8. Conclusões
    print("\n" + "="*60)
    print("CONCLUSÕES")
    print("="*60)
    
    # Verificar se modelos não lineares melhoram performance
    melhor_linear_acuracia = max(
        todos_resultados['logistic']['metrics']['accuracy'],
        todos_resultados['lda']['metrics']['accuracy']
    )
    
    melhor_nao_linear_acuracia = max(
        todos_resultados['svm']['metrics']['accuracy'],
        todos_resultados['knn']['metrics']['accuracy'],
        todos_resultados['nn']['metrics']['accuracy']
    )
    
    if melhor_nao_linear_acuracia > melhor_linear_acuracia:
        diferenca = melhor_nao_linear_acuracia - melhor_linear_acuracia
        diferenca_percentual = 100 * diferenca / melhor_linear_acuracia
        print(f"✓ Os modelos não lineares são superiores:")
        print(f"  • Melhor modelo linear: {melhor_linear_acuracia:.4f}")
        print(f"  • Melhor modelo não linear: {melhor_nao_linear_acuracia:.4f}")
        print(f"  • Melhoria: +{diferenca_percentual:.2f}%")
        print(f"  → A estrutura não linear melhora o desempenho da classificação")
    else:
        print(f"✓ Os modelos lineares têm desempenho similar ou melhor")
        print(f"  → As relações podem ser predominantemente lineares")
    
    # Análise de balanceamento de classes
    proporcao_classe_1 = sum(y == 1) / len(y)
    print(f"\nAnálise de Balanceamento:")
    print(f"  • Proporção da classe minoritária (alta qualidade): {proporcao_classe_1:.2%}")
    if proporcao_classe_1 < 0.3 or proporcao_classe_1 > 0.7:
        print(f"  → As classes estão desbalanceadas, o que pode afetar a métrica de acurácia")
        print(f"  → Considere usar outras métricas como F1-Score ou AUC-ROC")
    
    print("\n" + "="*60)
    print("EXECUÇÃO CONCLUÍDA COM SUCESSO!")
    print("="*60)
    print(f"Arquivos salvos em:")
    print(f"  • {OUTDIR}/ (dados e tabelas)")
    print(f"  • {FIGDIR}/ (gráficos e matrizes de confusão)")


if __name__ == "__main__":
    main()