"""Structure-only LightGBM vs TF-IDF+structure logistic regression: negative-tweet prediction."""
import io, json, os, sys, time
import numpy as np, pandas as pd, requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

SEED, SEEDS = 42, [42, 43, 44]
URL = "https://huggingface.co/datasets/osanseviero/twitter-airline-sentiment/resolve/main/Tweets.csv"
NUM = ["n_mentions", "n_hashtags", "n_urls", "retweet_count", "length"]
np.random.seed(SEED)


def download(url, tries=5):
    """GET with timeout and exponential backoff on 429/5xx."""
    for i in range(tries):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(str(r.status_code))
            r.raise_for_status()
            return r.content
        except requests.RequestException as e:
            print(f"download attempt {i + 1} failed: {e}")
            time.sleep(2 ** i)
    sys.exit("could not download data")


def load():
    raw = pd.read_csv(io.BytesIO(download(URL)))
    need = {"airline_sentiment", "airline", "retweet_count", "text"}
    if not need <= set(raw.columns):
        sys.exit(f"missing columns: {need - set(raw.columns)}")
    t = raw["text"].astype(str)
    df = pd.DataFrame({
        "airline": raw["airline"].astype(str), "text": t,
        "n_mentions": t.str.count(r"@\w+"), "n_hashtags": t.str.count(r"#\w+"),
        "n_urls": t.str.count(r"https?://"), "retweet_count": raw["retweet_count"].fillna(0),
        "length": t.str.len()})
    return df, (raw["airline_sentiment"] == "negative").astype(int).values


def structural():
    """Structural block shared by both models: one-hot brand + scaled log counts."""
    num = Pipeline([("log", FunctionTransformer(np.log1p)), ("sc", StandardScaler())])
    return [("brand", OneHotEncoder(handle_unknown="ignore"), ["airline"]), ("num", num, NUM)]


def baseline_pipe(seed):
    return Pipeline([("prep", ColumnTransformer(structural())),
                     ("clf", LGBMClassifier(learning_rate=0.05, random_state=seed, verbosity=-1, n_jobs=2))])


def improved_pipe(seed):
    text = [("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), "text"),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                                     sublinear_tf=True, max_features=100000), "text")]
    return Pipeline([("prep", ColumnTransformer(text + structural())),
                     ("clf", LogisticRegression(max_iter=2000, random_state=seed))])


def evaluate(model, X, y):
    p = model.predict_proba(X)[:, 1]
    pred = (p >= 0.5).astype(int)
    return {"auc": roc_auc_score(y, p), "acc": accuracy_score(y, pred),
            "f1": f1_score(y, pred, average="macro")}


df, y = load()
print(f"loaded {len(df)} tweets, negative share {y.mean():.3f}")
if len(df) < 1000 or y.min() == y.max():
    sys.exit("data not usable")
Xtr, Xte, ytr, yte = train_test_split(df, y, test_size=0.2, stratify=y, random_state=SEED)
print(f"train {len(Xtr)}, test {len(Xte)}")

# Hyper-parameter search with CV on the training set only (same folds for both models)
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
grids = {
    "baseline": (baseline_pipe, {"clf__n_estimators": [100, 300], "clf__num_leaves": [7, 31]}),
    "improved": (improved_pipe, {"clf__C": [0.3, 1, 3, 10, 30]}),
}
best, results = {}, {}
for name, (make, grid) in grids.items():
    gs = GridSearchCV(make(SEED), grid, scoring="roc_auc", cv=cv, n_jobs=1).fit(Xtr, ytr)
    best[name] = gs.best_params_
    print(f"{name}: best {gs.best_params_}, cv AUC {gs.best_score_:.4f}")
    runs = []
    for s in SEEDS:  # refit with each seed, evaluate once on test
        m = make(s).set_params(**gs.best_params_).fit(Xtr, ytr)
        runs.append(evaluate(m, Xte, yte))
    results[name] = {k: [r[k] for r in runs] for k in runs[0]}
    results[name]["cv_auc"] = gs.best_score_
    print(f"{name}: test AUC {np.mean(results[name]['auc']):.4f}")

b, i = results["baseline"], results["improved"]
metrics = {
    "baseline_score": float(np.mean(b["auc"])), "baseline_score_std": float(np.std(b["auc"])),
    "improved_score": float(np.mean(i["auc"])), "improved_score_std": float(np.std(i["auc"])),
    "baseline_accuracy": float(np.mean(b["acc"])), "improved_accuracy": float(np.mean(i["acc"])),
    "baseline_f1": float(np.mean(b["f1"])), "improved_f1": float(np.mean(i["f1"])),
    "baseline_cv_auc": float(b["cv_auc"]), "improved_cv_auc": float(i["cv_auc"]),
    "improved_best_C": float(best["improved"]["clf__C"]),
}
metrics["improvement"] = metrics["improved_score"] - metrics["baseline_score"]
if not all(np.isfinite(v) for v in metrics.values()):
    sys.exit("non-finite metrics")

os.makedirs("figures", exist_ok=True)
plt.bar(["structure-only LGBM", "TF-IDF + structure LR"],
        [metrics["baseline_score"], metrics["improved_score"]],
        yerr=[metrics["baseline_score_std"], metrics["improved_score_std"]], color=["gray", "tab:blue"])
plt.ylabel("Test ROC-AUC")
plt.title("Negative tweet prediction")
plt.tight_layout()
plt.savefig("figures/auc_comparison.png", dpi=120)

out = {
    "question": "On public tweets about competing airline brands, does adding tweet content to structural and engagement features predict negative customer sentiment better than a structure-only model?",
    "method": "Stratified 80/20 split (seed 42). Baseline: LightGBM on structural/engagement counts. Improved: logistic regression on word+char TF-IDF stacked with the same structural features. Hyper-parameters chosen by 5-fold CV on the training set only; test set used once.",
    "dataset": {"name": "Twitter US Airline Sentiment (osanseviero/twitter-airline-sentiment)",
                "source_url": URL, "n_rows": int(len(df)), "n_train": int(len(Xtr)),
                "n_test": int(len(Xte)), "same_as_paper": False, "baseline_is_papers_method": False},
    "metric_name": "ROC-AUC (test)", "higher_is_better": True,
    "baseline": "LightGBM on brand, #mentions, #hashtags, #URLs, retweet count and tweet length (structure only); n_estimators/num_leaves tuned by CV.",
    "improved": "Logistic regression on word (1-2gram) and char_wb (2-5gram) TF-IDF of the tweet text plus the same structural features; C tuned by CV on the training set.",
    "metrics": metrics,
    "notes": "Airline tweets are a stand-in for textile-brand data; the target is tweet-level negativity, not brand competitiveness. Single split; std is over seeds 42-44 only (logistic regression is near-deterministic), not over data splits. Accuracy and macro-F1 use a 0.5 threshold. Structural features are weak by design, so the gap mostly reflects the information in text.",
}
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print(json.dumps(metrics, indent=2))
