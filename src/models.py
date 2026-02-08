from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier #needs pip install!
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,   QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier

from config.config import Config


class ModelFactory:

    def __init__(self):
        self.config = Config()


    def get_model(self, model_name):
        models = {
            "svm": self.get_svm(),
            "random_forest": self.get_random_forest(),
            "knn": self.get_knn(),
            "logistic_regression": self.get_logistic_regression(),
            "gradient_boosting": self.get_gradient_boosting(),
            "xgboost": self.get_xgboost(),
            "mlp": self.get_mlp(),

            #metafermena apo tou kwsti
            "gaussian_nb": self.get_gaussianNB(),
            "lda": self.get_lda(),
            "decision_tree": self.get_decision_tree(),
            "qda": self.get_qda()
        }

        if model_name not in models:
            raise ValueError(f"Invalid model name {model_name}")

        return models[model_name]

    def get_svm(self):
        return SVC(
            kernel = 'rbf',
            C = 10,
            gamma = "scale",
            random_state = self.config.RANDOM_STATE
        )

    def get_random_forest(self):
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            random_state=self.config.RANDOM_STATE,
            n_jobs=-1
        )

    def get_knn(self):
        return KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            n_jobs=-1
        )

    def get_logistic_regression(self):
        return LogisticRegression(
            max_iter=1000,
            random_state=self.config.RANDOM_STATE
        )

    def get_gradient_boosting(self):
        return GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=self.config.RANDOM_STATE
        )

    def get_xgboost(self):
        return XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=self.config.RANDOM_STATE,
            n_jobs=-1,
            objective="multi:softprob",
            num_class=self.config.NUM_CLASSES,
            eval_metric="mlogloss"
        )

    def get_mlp(self):
        return MLPClassifier(
            hidden_layer_sizes=(128,64),
            activation="relu",
            max_iter=500,
            random_state=self.config.RANDOM_STATE
        )

    def get_gaussianNB(self):
        return GaussianNB()

    def get_lda(self):
        return LinearDiscriminantAnalysis(
            solver="svd",
            tol=0.0001
        )
    
    def get_decision_tree(self):
        return DecisionTreeClassifier(
            criterion="gini",
            min_samples_leaf=1,
            min_samples_split=2
        )

    def get_qda(self):
        return QuadraticDiscriminantAnalysis(
            store_covariance=False,
            tol=0.0001
        )

    def get_all_models(self):

        return {
            "svm": self.get_svm(),
            "random_forest": self.get_random_forest(),
            "knn": self.get_knn(),
            "logistic_regression": self.get_logistic_regression(),
            "gradient_boosting": self.get_gradient_boosting(),
            "xgboost": self.get_xgboost(),
            "mlp": self.get_mlp(),
            "gaussian_nb": self.get_gaussianNB(),
            "lda": self.get_lda(),
            "decision_tree": self.get_decision_tree(),
            "qda": self.get_qda()
        }