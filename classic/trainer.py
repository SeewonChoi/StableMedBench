import wandb
import random
from argparse import ArgumentParser
import numpy as np
import joblib

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, average_precision_score

import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer

from mimic_icu import icu_loader
from mimic_mortality import mortality_loader
from mcmed import sepsis_loader, decomp_loader
from ehrshot import hyperkalemia_loader, hypolgycemia_loader 

def impute(X_train, X_test, X_val, method, i):
    features = X_train.columns
    if method == 'mean':
        imputer = SimpleImputer(strategy='mean')
    elif method == 'median':
        imputer = SimpleImputer(strategy='median')
    else:
        imputer = KNNImputer(n_neighbors=5)

    X_train = imputer.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=features)
    
    X_test = imputer.transform(X_test)
    X_val = imputer.transform(X_val)

    X_test = pd.DataFrame(X_test, columns=features)
    X_val = pd.DataFrame(X_val, columns=features)

    joblib.dump(imputer, f'results/{task}/{model_name}_{task}_{method}_{i}_imputer.pkl')
    return X_train, X_test, X_val

class XGBTrainer:
    def __init__(self, X, y, ids, learning_rate, save_model=False):
        self.X = X
        self.y = y
        self.ids = ids
        self.save_model = save_model
        self.learning_rate = learning_rate
        
        pos_count = np.sum(y.values == 1)
        neg_count = np.sum(y.values == 0)
        weight_ratio = neg_count / pos_count
        self.pos_weight = weight_ratio
        class_weight = {0: 1.0, 1: weight_ratio}

    def cross_validate_model(self, n_splits=2, n_repeats=1, random_state=42):
        metrics = {
            'acc': [], 'auc': [], 'f1': [], 'precision': [], 'recall': [], 'auprc': []
        }

        for i in range(n_repeats):
            X_train, X_temp, y_train, y_temp, ids_train, ids_temp = train_test_split(
                self.X, self.y, self.ids, test_size=0.4, random_state=random_state, stratify=self.y
            )

            # Step 2: Split temp into validation (20%) and test (20%) -> each is 50% of the remaining 40%
            X_val, X_test, y_val, y_test, ids_val, ids_test = train_test_split(
                X_temp, y_temp, ids_temp, test_size=0.5, random_state=random_state, stratify=y_temp
            )
            
            X_train, X_test, X_val = impute(X_train, X_test, X_val, method, i)

            model = XGBClassifier(
                learning_rate=self.learning_rate,
                n_estimators=100,
                max_depth=5,
                min_child_weight=1,
                gamma=0,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='binary:logistic',
                scale_pos_weight=self.pos_weight,
                seed=1234,
                use_label_encoder=False,
                eval_metric=['error', 'auc', 'logloss'],
            )

            model.fit(X_train, y_train, verbose=True)
            
            test_pred = model.predict(X_test)
            test_pred_prob = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            acc = accuracy_score(y_test, test_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, test_pred, average='binary')
            auc = roc_auc_score(y_test, test_pred_prob)
            auprc = average_precision_score(y_test, test_pred_prob)
            conf_matrix = confusion_matrix(y_test, test_pred) 

            # Save metrics
            metrics['acc'].append(acc)
            metrics['auc'].append(auc)
            metrics['f1'].append(f1)
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['auprc'].append(auprc)

            print(f"[Fold {i+1}] Acc: {acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}, AUPRC: {auprc:.4f}")
            print(conf_matrix)

            if self.save_model:
                ids_test['probs'] = test_pred_prob
                ids_test.to_csv(f"results/{task}/{model_name}_{task}_{method}_{i}.csv", index=False)

                model.save_model(f"results/{task}/{model_name}_{task}_{method}_{i}.json") 
            
            wandb.log({
                'acc': (acc),
                'auc': (auc),
                'f1': (f1),
                'precision': (precision),
                'recall': (recall),
                'auprc': (auprc),
            })

        print("\n=== Average Performance ===")
        for key in metrics:
            mean_val = np.mean(metrics[key])
            std_val = np.std(metrics[key])
            print(f"{key.upper()}: {mean_val:.4f} ± {std_val:.4f}")
            
            wandb.log({
                'final_mean': mean_val,
                'final_std': std_val,
                'final_name': key.upper()
            })

        return metrics

class TreeTrainer:
    def __init__(self, X, y, ids, model_type, save_model=False):
        self.X = X
        self.y = y
        self.ids = ids
        self.save_model = save_model

    def cross_validate_model(self, n_splits=2, n_repeats=1, random_state=42):
        metrics = {
            'acc': [], 'auc': [], 'f1': [], 'precision': [], 'recall': [], 'auprc': []
        }

        for i in range(n_repeats):
            X_train, X_temp, y_train, y_temp, ids_train, ids_temp = train_test_split(
                self.X, self.y, self.ids, test_size=0.4, random_state=random_state, stratify=self.y
            )

            X_val, X_test, y_val, y_test, ids_val, ids_test = train_test_split(
                X_temp, y_temp, ids_temp, test_size=0.5, random_state=random_state, stratify=y_temp
            )

            X_train, X_test, X_val = impute(X_train, X_test, X_val, method, i)

            pos_count = np.sum(y_train == 1)
            neg_count = np.sum(y_train == 0)
            weight_ratio = neg_count / pos_count
            class_weight = {0: 1.0, 1: weight_ratio}

            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=2,
                class_weight=class_weight,
                random_state=1234,
                n_jobs=-1  # Use all available cores
            )
            
            # Train
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_prob = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            acc = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
            auc = roc_auc_score(y_test, y_pred_prob)
            auprc = average_precision_score(y_test, y_pred_prob)

            # Save metrics
            metrics['acc'].append(acc)
            metrics['auc'].append(auc)
            metrics['f1'].append(f1)
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['auprc'].append(auprc)

            print(f"[Fold {i+1}] Acc: {acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}, AUPRC: {auprc:.4f}")

            if self.save_model:
                ids_test['probs'] = y_pred_prob
                ids_test.to_csv(f"results/{task}/{model_name}_{task}_{method}_{i}.csv", index=False)
                joblib.dump(model, f"results/{task}/{model_name}_{task}_{method}_{i}.pkl")
            
            wandb.log({
                'acc': (acc),
                'auc': (auc),
                'f1': (f1),
                'precision': (precision),
                'recall': (recall),
                'auprc': (auprc),
            })
        
        print("\n=== Average Performance ===")
        for key in metrics:
            mean_val = np.mean(metrics[key])
            std_val = np.std(metrics[key])
            print(f"{key.upper()}: {mean_val:.4f} ± {std_val:.4f}")
            
            wandb.log({
                'final_mean': mean_val,
                'final_std': std_val,
                'final_name': key.upper()
            })

        return metrics
    
if __name__ == "__main__":
    # Argument parser
    parser = ArgumentParser("classic")
    parser.add_argument("--learning-rate", type=float, default=0.1)  # Used for XGBoost
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--save-model", default=True)
    parser.add_argument("--model", type=str, choices=["xgb", "rf",], default="rf",
                         help="Model type: xgboost or random forest")
    parser.add_argument("--impute", type=str, choices=["mean", "median", "knn"], default="median")
    parser.add_argument("--task", type=str, choices=['sepsis', 'decomp', 'mortality', 'icu', 'hyperkalemia', 'hypoglycemia'], 
                         default="sepsis")
    args = parser.parse_args()
    
    # Setup random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    model_name = args.model
    task = args.task
    method = args.impute

    if args.task == 'sepsis':
        X, y, ids = sepsis_loader()
    elif args.task == 'decomp':
        X, y, ids = decomp_loader()
    elif args.task == 'mortality':
        X, y, ids = mortality_loader()
    elif args.task == 'icu':
        X, y, ids = icu_loader()
    elif args.task == 'hyperkalemia':
        X, y, ids = hyperkalemia_loader()
    elif args.task == 'hypoglycemia':
        X, y, ids = hypolgycemia_loader()

    # Initialize wandb
    config = vars(args)
    config['task'] = task
    config['model'] = model_name
    config['impute'] = method
    
    run = wandb.init(
        project=f"Classic",
        config=config
    )
    print(config)
    
    # Select trainer based on model type
    if model_name == "rf":
        trainer = TreeTrainer(
            X, y, ids, 'rf',
            save_model=args.save_model
        )
    elif model_name == 'xgb':
        trainer = XGBTrainer(
            X, y, ids, 
            args.learning_rate,
            save_model=args.save_model
        )

    # Train and evaluate
    trainer.cross_validate_model()
    
    # Finish wandb run
    run.finish()