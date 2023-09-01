import os.path

import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from lime import lime_text
from text_classifier.models.base_model import BaseModel


def run_classification_report(y_test, y_pred, probas, classes):
    y_test_array = pd.get_dummies(y_test, drop_first=False).values

    # Accuracy, Precision, Recall
    accuracy = metrics.accuracy_score(y_test, y_pred)
    auc = metrics.roc_auc_score(y_test, probas, multi_class="ovr")
    print("Accuracy:", round(accuracy, 2))
    print("Auc:", round(auc, 2))
    print("Detail:")
    print(metrics.classification_report(y_test, y_pred))

    # Plot confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues,cbar=False)
    ax.set(xlabel="Pred", ylabel="True", xticklabels=classes, yticklabels=classes, title="Confusion matrix")
    plt.yticks(rotation=0)

    fig, ax = plt.subplots(nrows=1, ncols=2)
    # Plot roc
    for i in range(len(classes)):
        fpr, tpr, thresholds = metrics.roc_curve(y_test_array[:, i], probas[:, i])
        ax[0].plot(
            fpr, tpr, lw=3, label='{0} (area={1:0.2f})'.format(classes[i], metrics.auc(fpr, tpr))
        )
    ax[0].plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    ax[0].set(xlim=[-0.05, 1.0], ylim=[0.0, 1.05],
              xlabel='False Positive Rate',
              ylabel="True Positive Rate (Recall)",
              title="Receiver operating characteristic")
    ax[0].legend(loc="lower right")
    ax[0].grid(True)

    # Plot precision-recall curve
    for i in range(len(classes)):
        precision, recall, thresholds = metrics.precision_recall_curve(y_test_array[:, i], probas[:, i])
        ax[1].plot(
            recall, precision, lw=3, label='{0} (area={1:0.2f})'.format(classes[i], metrics.auc(recall, precision))
        )
    ax[1].set(
        xlim=[0.0, 1.05], ylim=[0.0, 1.05], xlabel='Recall', ylabel="Precision", title="Precision-Recall curve"
    )
    ax[1].legend(loc="best")
    ax[1].grid(True)
    plt.show()


def run_lime_explanation(
        df_test, y_test, y_pred, probas, i, classes, model, output_dir=None
):
    txt_instance = df_test["text_clean"].iloc[i]
    # check true value and predicted value
    print("True:", y_test[i], "--> Predicted:", y_pred[i], "| Prob:", round(np.max(probas[i]), 2))
    # show explanation
    explainer = lime_text.LimeTextExplainer(class_names=classes)
    explained = explainer.explain_instance(txt_instance, model.predict_proba, num_features=5, top_labels=len(classes))

    if output_dir:
        explained.save_to_file(os.path.join(output_dir, "lime_explanation.html"))
    else:
        explained.show_in_notebook(text=txt_instance, predict_proba=False)


def run_evaluation_report(model: BaseModel, df_test, y_test, output_dir=None):
    cleaned_texts = df_test["text_clean"].values
    y_pred, predicted_prob = model.predict(cleaned_texts)
    run_classification_report(y_test, y_pred, predicted_prob, model.classes)
    run_lime_explanation(
        df_test, y_test, y_pred, predicted_prob, 1, model.classes, model.model, output_dir
    )
