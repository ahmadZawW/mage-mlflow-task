import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import pandas as pd
import mlflow.models
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import logging
import matplotlib.pyplot as plt

@transformer
def train_model(df, *args, **kwargs):
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Set MLflow experiment with more descriptive name
    experiment_name = "telcoChurn_prediction"
    mlflow.set_experiment(experiment_name)
    
    # Prepare features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Define models with parameters
    models = {
        'random_forest': RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
        'logistic_regression': LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=42
        )
    }
    
    results = {}
    best_model_name = None
    best_accuracy = 0
    
    with mlflow.start_run(run_name="model_comparison") as parent_run:
        # Log dataset info
        mlflow.log_param("dataset_size", len(df))
        mlflow.log_param("feature_count", X.shape[1])
        mlflow.log_param("training_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        
        # Train and evaluate each model
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            with mlflow.start_run(nested=True, run_name=name) as child_run:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_test, y_pred, average='binary'
                )
                auc_roc = roc_auc_score(y_test, y_pred_proba)
                
                # Log parameters
                mlflow.log_params(model.get_params())
                
                # Log metrics
                metrics = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "auc_roc": auc_roc
                }
                mlflow.log_metrics(metrics)
                
                # Log feature importance if available
                if hasattr(model, 'feature_importances_'):
                    feature_importance = pd.DataFrame({
                        'feature': X.columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    # Save feature importance plot
                    plt.figure(figsize=(10, 6))
                    plt.bar(feature_importance['feature'][:10], 
                           feature_importance['importance'][:10])
                    plt.xticks(rotation=45)
                    plt.title(f'Top 10 Feature Importance - {name}')
                    plt.tight_layout()
                    plt.savefig(f'{name}_feature_importance.png')
                    mlflow.log_artifact(f'{name}_feature_importance.png')
                    plt.close()
                
                # Log model
                mlflow.sklearn.log_model(
                    model,
                    name,
                    input_example=X_test.iloc[:5],
                    signature=mlflow.models.infer_signature(X_test, y_pred)
                )
                
                # Track best model
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model_name = name
                
                results[name] = metrics
                
                # Save model locally
                joblib.dump(model, f'mage_pipeline/models/{name}_model.joblib')
        
        # Log best model information
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_metric("best_accuracy", best_accuracy)
    
    # Save best model separately
    best_model = models[best_model_name]
    joblib.dump(best_model, 'mage_pipeline/models/best_model.joblib')
    
    return {
        'results': results,
        'best_model': best_model_name,
        'test_data': X_test.head(25)
    }